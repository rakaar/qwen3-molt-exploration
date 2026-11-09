"""Freeze a 16-token lookahead diagnostic on the 12 audited pre-eleven contexts."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/2026-11-10-t2375-pre11-lookahead'
OUT.mkdir(exist_ok=True)
SUFFIX = '\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint(row):
    return hashlib.sha256(json.dumps([row['str_tokens'],row['activations']],ensure_ascii=False).encode()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

spec = importlib.util.spec_from_file_location('token_display', ROOT / 'artifacts/molt-dashboard-vast-20261028/source/crosslayer_transcoder/dashboard/token_display.py')
display = importlib.util.module_from_spec(spec)
spec.loader.exec_module(display)
escape = display.escape_added_tokens
coverage_path = ROOT / 'results/2026-11-10-t2375-pre11-coverage/counts.json'
recovery_path = ROOT / 'results/2026-11-10-t2375-pre11-coverage/recovered-windows.json'
feature_path = ROOT / 'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json'
original_plan_path = ROOT / 'results/2026-11-09-t2375-explanation-refinement/heldout-plan.json'
coverage = json.loads(coverage_path.read_text())
recovery = json.loads(recovery_path.read_text())
feature = json.loads(feature_path.read_text())
original = json.loads(original_plan_path.read_text())
assert sha(feature_path) == coverage['source_feature_sha256'] == recovery['feature_sha256']
previous = json.loads((ROOT / 'results/2026-11-10-t2375-detection-fuzz-heldout/summary.json').read_text())
previous_active = json.loads((ROOT / 'results/2026-11-09-t2375-active-fuzz-local-judge/summary.json').read_text())
assert previous['status'] == previous_active['status'] == 'complete'
index = {(c['method'],c['row_index']):c for c in original['cases']}
cases = []
for method in ('detection','fuzz','active_fuzz'):
    targets = {c['fingerprint']:c for c in coverage['methods'][method]['cases']}
    assert len(targets) == 12
    source_rows = feature['methods'][method]['rows']
    selected = [(c['row_index'],'positive') for c in targets.values()]
    if method == 'active_fuzz':
        negatives = [(i,'inactive_highlight_control') for i,r in enumerate(source_rows)
                     if not r['activating'] and fingerprint(r) in targets]
        assert len(negatives) == 12
        selected += negatives
    for i, role in sorted(selected):
        row = source_rows[i]
        key = fingerprint(row)
        source = recovery['windows'][key]
        assert len(source['matches']) == 1
        last = next(a for a in source['matches'][0]['active_positions']
                    if a['window_position'] == len(row['str_tokens'])-1)
        following = last['following_tokens'][:16]
        assert len(following) == 16
        appended = escape(''.join(following))
        case = deepcopy(index[(method,i)])
        case.update(role=role, source_fingerprint=key,
                    continuation_token_ids=last['following_token_ids'][:16],
                    continuation_str_tokens=following,
                    original_highlighted_positions=row.get('highlighted_positions'))
        if role == 'positive':
            assert row['activating'] is True
        else:
            assert row['activating'] is False
            assert all(row['activations'][p] == 0 for p in row['highlighted_positions'])
        for arm in ('baseline','candidate'):
            messages = case[arm+'_messages']
            assert messages[-1]['content'].endswith(escape(row.get('shown_text',''.join(row['str_tokens'])))+SUFFIX)
            old = messages[-1]['content']
            messages[-1]['content'] = old[:-len(SUFFIX)] + appended + SUFFIX
            assert messages[-1]['content'].replace(appended+SUFFIX,SUFFIX) == old
            assert messages[:-1] == index[(method,i)][arm+'_messages'][:-1]
        assert case['baseline_messages'][:-1] == case['candidate_messages'][:-1]
        score = previous_active if method=='active_fuzz' else previous['methods'][method]
        pair = next(p for p in score['pairs'] if p['row_index']==i)
        case['prefix_only_predictions'] = {a:pair[a] for a in ('baseline','candidate')}
        cases.append(case)
assert len(cases) == 48 and sum(c['activating'] for c in cases) == 36
for arm in ('baseline','candidate'):
    src = ROOT / f'experiments/t2375-explanation-refinement-20261109/{arm}-explanation.txt'
    assert src.read_text().strip() == original[arm+'_explanation']
    shutil.copyfile(src,HERE/src.name)
plan = {k:original[k] for k in ('feature','baseline_explanation','candidate_explanation')}
plan.update(status='frozen_before_scoring', cases=cases, lookahead_subject_tokens=16)
save(HERE/'heldout-plan.json',plan)
runtime = ROOT / 'results/2026-11-09-t2375-active-fuzz-local-judge/runtime.json'
verification = ROOT / 'results/2026-11-09-t2375-active-fuzz-local-judge/judge-verification.json'
shutil.copyfile(runtime,HERE/'expected-active-fuzz-runtime.json')
shutil.copyfile(verification,HERE/'expected-judge-verification.json')
protocol = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    subject_feature='L29-T2375', judge_repo='vrfai/Qwen3.8-27B-NVFP4',
    judge_revision='1561c70b22784dba7c6943b3271fc5f185adda01',
    existing_judge_root='/workspace/t2375-active-fuzz-local-judge-20261109',
    budget_source='Reuse authorized SPAR instance 59903353 within existing $10 total cap',
    methods=['detection','fuzz','active_fuzz'], cases_per_method={'detection':12,'fuzz':12,'active_fuzz':24},
    positive_cases_per_method=12, active_fuzz_negative_controls=12,
    arms=['baseline','candidate'], requests=96, unique_request_bodies=72,
    lookahead_subject_tokens=16, maximum_runtime_minutes=25,
    temperature=0,max_tokens=32,enable_thinking=False,
    descriptions_unchanged=True, highlights_unchanged=True, prefix_labels_unchanged=True,
    only_prompt_change='Append 16 original subject-model source tokens before the final classification instruction; keep fixed few-shots, explanation text, labels and original highlights.',
    interpretation='Post-hoc lookahead diagnostic on previously inspected cases; not a new blind held-out score and not a causal predictor benchmark. Positive Fuzz and Active Fuzz requests are identical. Added tokens are context, not additional highlighted targets.',
    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (coverage_path,recovery_path,feature_path,original_plan_path)},
    heldout_plan_sha256=sha(HERE/'heldout-plan.json'),
    expected_active_fuzz_runtime_sha256=sha(HERE/'expected-active-fuzz-runtime.json'),
    expected_judge_verification_sha256=sha(HERE/'expected-judge-verification.json'),
    **{a+'_sha256':hashlib.sha256((HERE/(a+'-explanation.txt')).read_text().strip().encode()).hexdigest() for a in ('baseline','candidate')})
save(HERE/'protocol.json',protocol)
for name in ('protocol.json','heldout-plan.json'):
    shutil.copyfile(HERE/name,OUT/name)
print(json.dumps({k:protocol[k] for k in ['requests','unique_request_bodies','cases_per_method','lookahead_subject_tokens','descriptions_unchanged']},indent=2))
