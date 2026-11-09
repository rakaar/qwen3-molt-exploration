"""Independently join downloaded judge decisions to the frozen lookahead plan."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/2026-11-10-t2375-pre11-lookahead'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

protocol=json.loads((HERE/'protocol.json').read_text())
assert sha(HERE/'heldout-plan.json')==protocol['heldout_plan_sha256']
plan=json.loads((HERE/'heldout-plan.json').read_text())
summary=json.loads((OUT/'summary.json').read_text())
assert summary['status']=='complete' and summary['requests']==96
assert summary['protocol']==protocol
assert (OUT/'heldout-plan.json').read_bytes()==(HERE/'heldout-plan.json').read_bytes()
assert (OUT/'run.py').read_bytes()==(HERE/'run.py').read_bytes()
for arm in ('baseline','candidate'):
    text=(HERE/f'{arm}-explanation.txt').read_text().strip()
    assert hashlib.sha256(text.encode()).hexdigest()==protocol[arm+'_sha256']
    assert plan[arm+'_explanation']==text
audit=[json.loads(s) for s in (OUT/'audit.jsonl').read_text().splitlines()]
assert len(audit)==96
indexed={(r['method'],r['row_index'],r['arm']):r for r in audit}
assert len(indexed)==96
inputs=json.loads((OUT/'prepared-inputs.json').read_text())['tasks']
assert len(inputs)==96
tasks={(r['method'],r['row_index'],r['arm']):r for r in inputs}
comparison=json.loads((OUT/'judge-settings-comparison.json').read_text())
assert comparison['all_match'] is True
methods={}
for method in protocol['methods']:
    cases=[c for c in plan['cases'] if c['method']==method]
    pairs=[]
    for c in cases:
        pair=dict(row_index=c['row_index'],activating=c['activating'],role=c['role'],
                  prefix_only_predictions=c['prefix_only_predictions'])
        for arm in ('baseline','candidate'):
            key=(method,c['row_index'],arm)
            r=indexed[key]
            assert r['activating']==c['activating']
            assert r['messages']==c[arm+'_messages']==tasks[key]['messages']
            assert r['prompt_token_ids']==tasks[key]['prompt_token_ids']
            assert r['request_sha256']==tasks[key]['request_sha256']
            assert r['finish_reason']=='stop' and r['response'].strip() in ('[0]','[1]')
            assert r['prediction']==(r['response'].strip()=='[1]')
            pair[arm]=r['prediction']
        pairs.append(pair)
    positive=[p for p in pairs if p['activating']]
    negative=[p for p in pairs if not p['activating']]
    assert len(positive)==12
    arms={}
    for arm in ('baseline','candidate'):
        before=sum(p['prefix_only_predictions'][arm] for p in positive)
        after=sum(p[arm] for p in positive)
        arms[arm]=dict(prefix_only_positive_correct=before,lookahead_positive_correct=after,
                       positive_total=12,lookahead_positive_rate=after/12,
                       prefix_only_control_false_positives=sum(p['prefix_only_predictions'][arm] for p in negative),
                       lookahead_control_false_positives=sum(p[arm] for p in negative),control_total=len(negative))
        assert summary['methods'][method][arm]['tp']==after
        assert summary['methods'][method][arm]['fp']==arms[arm]['lookahead_control_false_positives']
    methods[method]=dict(arms=arms,pairs=pairs,
        new_description_corrected=sum(not p['baseline'] and p['candidate'] for p in positive),
        new_description_regressed=sum(p['baseline'] and not p['candidate'] for p in positive))
duplicates={}
for r in audit:
    duplicates.setdefault(r['request_sha256'],[]).append(r)
assert len(duplicates)==72
assert all(len({r['response'] for r in group})==1 for group in duplicates.values())
checks=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
    complete_requests=96,unique_request_bodies=72,exact_plan_messages_verified=True,
    exact_saved_input_tokens_verified=True,labels_unchanged=True,descriptions_unchanged=True,
    no_parse_failures=True,no_truncated_prompts=True,no_truncated_responses=True,
    runtime_matches_previous_local_judge=True,identical_requests_have_identical_outputs=True,
    lookahead_subject_tokens=16,posthoc=True,methods=methods)
assert max(len(r['prompt_token_ids']) for r in audit)+32<=4096
save(OUT/'verification.json',checks)
lines=['# T2375: 16-token lookahead diagnostic','',
       'The same 12 confirmed pre-eleven passages were rerun after appending the next 16 original subject-model source tokens. Both frozen descriptions, fixed few-shots, labels, highlights and judge settings were preserved.','',
       '| Test | Old: prefix only | Old: +16 tokens | New: prefix only | New: +16 tokens |',
       '|---|---:|---:|---:|---:|']
for method,m in methods.items():
    a,b=m['arms']['baseline'],m['arms']['candidate']
    lines.append(f"| {method} | {a['prefix_only_positive_correct']}/12 | {a['lookahead_positive_correct']}/12 | {b['prefix_only_positive_correct']}/12 | {b['lookahead_positive_correct']}/12 |")
lines += ['', 'These are positive-case identification rates, not accuracy on a balanced classification benchmark. Fuzz and Active Fuzz have identical positive prompts and produced identical positive predictions; they are not independent replications.','',
          '## Active Fuzz inactive-highlight controls','']
for arm in ('baseline','candidate'):
    a=methods['active_fuzz']['arms'][arm]
    lines.append(f"- {arm}: false positives {a['prefix_only_control_false_positives']}/{a['control_total']} before, {a['lookahead_control_false_positives']}/{a['control_total']} after lookahead.")
lines += ['', 'The control highlights remain at their original zero-gate positions. The newly appended continuation is unhighlighted. Because the subject model is causal, future context does not change the already measured gates at the original positions.','',
          '## Scope and provenance','',
          '- The shared 12 contexts comprise seven before list item 11, one before the date-range endpoint 11, two before November, and two before the 11 suffix in 2011. No XI or K cases are present.','- This is a post-hoc diagnostic on inspected cases, not a new blind held-out result. Showing a future continuation changes the original prefix-only evaluation task.','- Detection and Fuzz each use the 12 positive cases. Active Fuzz uses those same 12 positive cases plus their 12 original inactive-highlight controls. Two descriptions give 96 logical requests and 72 unique request bodies.','- Prefix-only counts come from the completed previous same-judge runs. They were not rerun for this diagnostic.','- Judge: vrfai/Qwen3.8-27B-NVFP4 at 1561c70b22784dba7c6943b3271fc5f185adda01; same verified cached weights, vLLM 0.33.0, Marlin W4A16 on RTX 3090, temperature 0, thinking disabled, max output 32.','- All judge responses were valid and untruncated. Source, request and response audits, runtime comparison and verified backup receipts are retained.','',
          '## Per-case positive decisions','', '| Test | Original row | Old +16 | New +16 |', '|---|---:|---:|---:|']
for method,m in methods.items():
    for p in m['pairs']:
        if p['activating']:
            lines.append(f"| {method} | {p['row_index']} | {int(p['baseline'])} | {int(p['candidate'])} |")
lines += ['', 'A decision of 1 is correct for these positive cases. See heldout-plan.json and audit.jsonl for the exact full messages and generated answers.','']
(OUT/'README.md').write_text('\n'.join(lines))
save(OUT/'local-manifest.json',dict(files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json']))
print(json.dumps({method:m['arms'] for method,m in methods.items()},indent=2))
