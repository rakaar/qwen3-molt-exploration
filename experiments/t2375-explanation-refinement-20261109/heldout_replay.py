#!/usr/bin/env python3
"""Prepare or execute a paired explanation comparison on the original labeled cases.

Preparation is offline. Execution requires --execute and an explicit reachable
judge URL. No rental, lifecycle, dashboard edits or credential storage occur.
Original row-level predictions come from the dashboard feature JSON, while
exact requests come from its revision-pinned public export. This does not
fabricate an original prompt audit or its missing raw judge responses.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT/'comparisons/2026-11-09-t2375-explanation'
OUTPUT = ROOT/'results/2026-11-09-t2375-explanation-refinement'
SOURCE = ROOT/'artifacts/molt-dashboard-vast-20261028/source'
MODEL = 'qwen3.8-27b-nvfp4'
METHODS = ('detection', 'fuzz', 'active_fuzz')
SUFFIX = '\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def confusion(rows):
    valid = [r for r in rows if r['prediction'] is not None]
    out = dict(total=len(rows), valid=len(valid), parse_failures=len(rows)-len(valid))
    for name, active, prediction in [('tp', True, True), ('tn', False, False),
                                     ('fp', False, True), ('fn', True, False)]:
        out[name] = sum(r['activating'] == active and r['prediction'] == prediction for r in valid)
    out['accuracy'] = (out['tp']+out['tn'])/len(valid) if valid else None
    out['balanced_accuracy'] = ((out['tp']/(out['tp']+out['fn'])+out['tn']/(out['tn']+out['fp']))/2
                                if out['tp']+out['fn'] and out['tn']+out['fp'] else None)
    return out


def prepare():
    protocol = json.loads((HERE/'protocol.json').read_text())
    candidate = (HERE/'candidate-explanation.txt').read_text().strip()
    baseline_text = (HERE/'baseline-explanation.txt').read_text().strip()
    assert hashlib.sha256(candidate.encode()).hexdigest() == protocol['candidate_sha256'], 'Frozen candidate changed'
    assert hashlib.sha256(baseline_text.encode()).hexdigest() == protocol['baseline_sha256'], 'Frozen baseline changed'
    feature_path = DATA/'original-campaign/features/2375.json'
    feature = json.loads(feature_path.read_text())
    assert feature['id'] == 2375 and feature['explanation'] == baseline_text
    prefixes = json.loads((DATA/'scoring_prefixes.json').read_text())
    requests = [json.loads(s) for s in (DATA/'published-requests-T2375.jsonl').read_text().splitlines()]
    index = {}
    for row in requests:
        assert row['layer'] == 29 and row['feature_id'] == 2375
        if row['task'] != 'binary':
            continue
        index.setdefault(row['suffix_messages'][-1]['content'], []).append(row)
    token_display = SOURCE/'crosslayer_transcoder/dashboard/token_display.py'
    spec = importlib.util.spec_from_file_location('pinned_token_display', token_display)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    escape = module.escape_added_tokens
    cases = []
    for method in METHODS:
        original = feature['methods'][method]
        assert original['status'] == 'complete' and len(original['rows']) == 100
        counts = confusion(original['rows'])
        assert all(original['metrics'][k] == v for k, v in counts.items()), 'Original metric replay failed'
        for number, row in enumerate(original['rows']):
            shown = row.get('shown_text', ''.join(row['str_tokens']))
            tail = '\n\nText examples:\n\nExample 0: '+shown+SUFFIX
            user = 'Latent explanation: '+baseline_text+tail
            matches = index.get(user, [])
            assert matches, f'No exact exported request: {method}/{number}'
            bodies = []
            for match in matches:
                messages = prefixes[match['prefix_id']]+match['suffix_messages']
                assert match['max_tokens'] == 32
                bodies.append(messages)
            assert len({digest(b) for b in bodies}) == 1, 'Ambiguous full historical request'
            historical = deepcopy(bodies[0])
            old = [dict(m, content=escape(m['content'])) for m in historical]
            new = deepcopy(old)
            new[-1]['content'] = escape('Latent explanation: '+candidate+tail)
            assert old[:-1] == new[:-1]
            cases.append(dict(method=method, row_index=number, activating=row['activating'],
                              historical_prediction=row['prediction'],
                              published_audit_lines=[r['audit_line'] for r in matches],
                              display_corrected=old != historical,
                              baseline_messages=old, candidate_messages=new))
    sources = [feature_path, DATA/'published-requests-T2375.jsonl', DATA/'scoring_prefixes.json',
               HERE/'candidate-explanation.txt', HERE/'baseline-explanation.txt', token_display]
    plan = dict(feature='L29-T2375', status='prepared_not_executed', judge_model=MODEL,
                baseline_explanation=baseline_text, candidate_explanation=candidate,
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                original_metrics={m:feature['methods'][m]['metrics'] for m in METHODS},
                historical_prediction_counts_verified=True, exact_request_matches=len(cases),
                display_corrected_cases=sum(c['display_corrected'] for c in cases),
                intruder='Not rerun: independent of explanation', cases=cases)
    save(OUTPUT/'heldout-plan.json', plan)
    return plan


def execute(plan, args):
    assert args.base_url, '--base-url is required for execution'
    assert args.model == MODEL, 'This comparison requires the original judge alias; do not silently substitute'
    assert 1 <= args.concurrency <= 16
    key = os.environ.get(args.api_key_env)
    headers = {'Content-Type':'application/json'}
    if key:
        headers['Authorization'] = 'Bearer '+key
    base = args.base_url.rstrip('/')
    req = urllib.request.Request(base+'/models', headers=headers)
    with urllib.request.urlopen(req, timeout=20) as response:
        models = json.load(response)
    assert MODEL in [r.get('id') for r in models['data']], 'Requested judge is not advertised'
    config = dict(base_url=base, model=MODEL, temperature=0, max_tokens=32,
                  chat_template_kwargs={'enable_thinking':False}, plan_sha256=digest(plan))
    run = OUTPUT/'heldout-scored'
    config_path = run/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Use a fresh run directory for changed settings'
    save(config_path, config)
    audit = run/'audit.jsonl'
    cached = {}
    if audit.exists():
        for line in audit.read_text().splitlines():
            row = json.loads(line)
            cached[row['request_sha256']] = row
    unique = {}
    case_keys = {}
    for case in plan['cases']:
        for arm in ['baseline', 'candidate']:
            body = dict(model=MODEL, messages=case[arm+'_messages'], temperature=0,
                        max_tokens=32, chat_template_kwargs={'enable_thinking':False})
            identity = digest(body)
            unique[identity] = body
            case_keys[(case['method'],case['row_index'],arm)] = identity

    def call(identity, body):
        row = dict(request_sha256=identity, body=body, prediction=None,
                   at_utc=datetime.now(timezone.utc).isoformat())
        try:
            req = urllib.request.Request(base+'/chat/completions', data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.load(response)
            row['response'] = result
            choice = result['choices'][0]
            text = choice['message'].get('content', '').strip()
            assert choice.get('finish_reason') != 'length', 'Output truncated'
            assert text in ['[0]', '[1]'], 'Invalid binary answer'
            row['prediction'] = text == '[1]'
        except Exception as error:
            row['error'] = type(error).__name__+': '+str(error)
        return row

    todo = [(k,b) for k,b in unique.items() if k not in cached]
    # A stored invalid request is retained; it is never silently retried for a better score.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        jobs = [pool.submit(call,k,b) for k,b in todo]
        for completed in as_completed(jobs):
            row = completed.result()
            with audit.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            cached[row['request_sha256']] = row
    summary = dict(feature='L29-T2375', scope='original_heldout_paired_explanation_comparison',
                   config=config, unique_requests=len(unique), new_requests=len(todo), methods={})
    for method in METHODS:
        cases = [c for c in plan['cases'] if c['method']==method]
        paired = []
        for case in cases:
            pair = dict(row_index=case['row_index'], activating=case['activating'])
            for arm in ['baseline','candidate']:
                pair[arm] = cached[case_keys[(method,case['row_index'],arm)]]['prediction']
            paired.append(pair)
        comparison = {}
        for arm in ['baseline','candidate']:
            comparison[arm] = confusion([dict(activating=p['activating'],prediction=p[arm]) for p in paired])
        valid = [p for p in paired if p['baseline'] is not None and p['candidate'] is not None]
        comparison.update(paired_valid=len(valid),
                          corrected=sum(p['baseline']!=p['activating'] and p['candidate']==p['activating'] for p in valid),
                          regressed=sum(p['baseline']==p['activating'] and p['candidate']!=p['activating'] for p in valid),
                          rows=paired)
        summary['methods'][method] = comparison
    summary['status'] = 'complete' if all(m[a]['parse_failures']==0 for m in summary['methods'].values() for a in ['baseline','candidate']) else 'incomplete'
    save(run/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='methods'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--base-url')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--api-key-env', default='vLLM_API_KEY')
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    plan = prepare()
    print(json.dumps({k:v for k,v in plan.items() if k in ['status','judge_model','exact_request_matches','display_corrected_cases','historical_prediction_counts_verified']}))
    if args.execute:
        execute(plan, args)
