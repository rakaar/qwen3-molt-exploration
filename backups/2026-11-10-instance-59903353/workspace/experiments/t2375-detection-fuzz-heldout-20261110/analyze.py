"""Verify fixed-plan decisions and summarize counts without displaying test text."""
import hashlib
import json
from math import comb
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'results/2026-11-10-t2375-detection-fuzz-heldout'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
summary=json.loads((OUT/'summary.json').read_text())
rows=[json.loads(line) for line in (OUT/'audit.jsonl').read_text().splitlines()]
protocol=json.loads((ROOT/'protocol.json').read_text())
assert sha(ROOT/'heldout-plan.json')==protocol['heldout_plan_sha256']
plan=json.loads((ROOT/'heldout-plan.json').read_text())
cases={(r['method'],r['row_index']):r for r in plan['cases'] if r['method'] in ['detection','fuzz']}
assert len(rows)==400 and len({r['request_sha256'] for r in rows})==400
assert len(cases)==200
for arm in ['baseline','candidate']:
    text=(ROOT/(arm+'-explanation.txt')).read_text().strip()
    assert hashlib.sha256(text.encode()).hexdigest()==protocol[arm+'_sha256']
    assert text==plan[arm+'_explanation']
for row in rows:
    case=cases[(row['method'],row['row_index'])]
    assert row['messages']==case[row['arm']+'_messages']
    assert row['activating']==case['activating']
    response=row['response'].strip()
    parsed=(response=='[1]') if response in ['[0]','[1]'] and row['finish_reason']!='length' else None
    assert parsed==row['prediction']
for method,result in summary['methods'].items():
    for arm in ['baseline','candidate']:
        selected=[r for r in rows if r['method']==method and r['arm']==arm]
        assert len(selected)==100 and sum(r['activating'] for r in selected)==50
        valid=[r for r in selected if r['prediction'] is not None]
        correct=sum(r['prediction']==r['activating'] for r in valid)
        assert correct==result[arm]['tp']+result[arm]['tn']
        assert len(valid)==result[arm]['valid']
        assert correct/len(valid)==result[arm]['accuracy']
    pairs=result['pairs']
    valid=[p for p in pairs if p['baseline'] is not None and p['candidate'] is not None]
    assert len(pairs)==100 and len(valid)==result['paired_valid']
    assert sum(p['baseline']!=p['activating'] and p['candidate']==p['activating'] for p in valid)==result['corrected']
    assert sum(p['baseline']==p['activating'] and p['candidate']!=p['activating'] for p in valid)==result['regressed']
runtime=json.loads((OUT/'runtime.json').read_text())
assert sha(OUT/'run.py')==runtime['source_sha256']['run.py']
settings=json.loads((OUT/'judge-settings-comparison.json').read_text());assert settings['all_match']
old_runtime=json.loads((ROOT/'expected-active-fuzz-runtime.json').read_text())
for field in settings['matched_fields']:assert runtime[field]==old_runtime[field]

# Post-scoring consistency check on repeated request hashes; do not inspect text.
prior=ROOT.parents[1]/'results/2026-11-09-t2375-active-fuzz-local-judge'
prior_rows={r['request_sha256']:r for r in [json.loads(line) for line in (prior/'audit.jsonl').read_text().splitlines()]}
shared=[r for r in rows if r['request_sha256'] in prior_rows]
agreement=sum(r['prediction']==prior_rows[r['request_sha256']]['prediction'] for r in shared)
verification=dict(status='verified',requests=400,cases_per_method=100,
    all_requests_match_frozen_plan=True,all_labels_match_saved_ground_truth=True,
    description_hashes_unchanged=True,judge_settings_match_active_fuzz=True,
    metrics_independently_recomputed=True,executed_source_hash_matches=True,
    manual_test_text_inspection_in_this_run_before_scoring=False,
    repeated_requests_from_active_fuzz=len(shared),repeated_predictions_matching=agreement)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
shutil.copy2(ROOT/'inference-code-comparison.json',OUT/'inference-code-comparison.json')

def correct(result,arm):return result[arm]['tp']+result[arm]['tn']
def row(name,result):return f"| {name} | {correct(result,'baseline')}/100 | {correct(result,'candidate')}/100 | {result['corrected']} | {result['regressed']} | {result['corrected']-result['regressed']:+d} |"
lines=['# T2375: complete held-out Detection and Fuzz comparison','',
       'Both frozen descriptions were evaluated on all 100 saved Detection cases and all 100 saved Fuzz cases, with the exact cached judge and inference settings used for the earlier local Active Fuzz run.','',
       '| Test | Original correct | Revised correct | Improved | Worsened | Net extra correct |',
       '|---|---:|---:|---:|---:|---:|',
       row('Detection',summary['methods']['detection']),row('Fuzz',summary['methods']['fuzz'])]
active=json.loads((prior/'summary.json').read_text())
lines += [row('Active Fuzz (previous run)',active),'',
    'Improved means the original description was judged incorrectly and the revised description correctly. Worsened means the reverse. Net is improved minus worsened.','',
    '## Evaluation integrity','',
    '- All 100 predefined cases per method were included: 50 activating and 50 inactive labels. Both arms were freshly evaluated, for 400 classifications total.',
    '- Both explanation strings match their original frozen SHA-256 hashes. No rewriting, case selection, or tuning followed test feedback.',
    '- No additional manual test-text inspection occurred before scoring in this run. Test content was read only programmatically for tokenization and judge inference; logs showed progress counts.',
    '- Historical example demonstrations and Active Fuzz feedback had already been viewed in earlier work. The candidate stayed unchanged throughout; these are saved benchmark cases, not a claim that their text has never been seen.',
    '- The native judge weights were re-hashed against the prior Active Fuzz model manifest before loading. The runtime versions, GPU, model revision and generation settings match the earlier run.',
    '- Both original and revised arms use identical example text, highlight masks, few-shot examples, rendering and decoding settings. Only the explanation differs.',
    '- No prompt truncation, forced classification repair, or selective response retry was used.',
    '- These metrics assess gate-condition descriptions, not the causal computation of the transform.','',
    '## Per-method details','']
statistics={}
for method,result in summary['methods'].items():
    b,c=result['baseline'],result['candidate']
    n=result['corrected']+result['regressed']
    p=min(1.,2*sum(comb(n,i) for i in range(min(result['corrected'],result['regressed'])+1))/2**n) if n else 1.
    statistics[method]=dict(corrected=result['corrected'],regressed=result['regressed'],discordant=n,exact_two_sided_p=p)
    lines += [f'### {method}','',
        f"Paired valid responses: {result['paired_valid']}/100. Parse failures: original {b['parse_failures']}, revised {c['parse_failures']}.",'',
        '| Count | Original | Revised |','|---|---:|---:|',
        f"| True positives | {b['tp']} | {c['tp']} |",f"| False negatives | {b['fn']} | {c['fn']} |",
        f"| True negatives | {b['tn']} | {c['tn']} |",f"| False positives | {b['fp']} | {c['fp']} |",'',
        f'Exact paired discordance test, two-sided and unadjusted: p = {p:.5g}. Treat this as an exploratory comparison on one saved set and judge.','']
lines += ['## Judge and artifacts','',
    f"- Judge: `{runtime['judge_repo']}` at `{runtime['judge_revision']}`.",
    f"- Runtime: vLLM {runtime['vllm']}, torch {runtime['torch']}, transformers {runtime['transformers']}.",
    '- RTX 3090, Marlin NVFP4 weight-only kernels with BF16 activations, greedy decoding, thinking disabled, 32 output-token limit.',
    '- Same 4,096-token context limit, one active sequence, seed 42, eager execution, disabled prefix caching and FlashInfer sampler as Active Fuzz.',
    '- The previous 32-case Detection/Fuzz gains were training diagnostics with the dashboard judge. They are separate from this 100-case comparison.',
    '- This is the same fresh local judge as our Active Fuzz result; historical dashboard judge scores remain separate.',
    f'- Repeated request consistency with Active Fuzz: {agreement}/{len(shared)} shared requests have matching predictions.',
    '- Raw requests and responses: audit.jsonl. Full tokenized inputs: prepared-inputs.json. Per-case decisions: summary.json.',
    '- Protocol, executed source, environment freeze, hash manifests and independent verification are included in this directory.','']
(OUT/'README.md').write_text('\n'.join(lines))
(OUT/'paired-statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json']
(OUT/'local-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps(dict(status=summary['status'],methods={m:{k:v[k] for k in ['baseline','candidate','corrected','regressed','paired_valid']} for m,v in summary['methods'].items()},verification=verification)))
