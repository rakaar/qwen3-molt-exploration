"""Report the paired local-judge result separately from dashboard scores."""
import hashlib
import json
from math import comb
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'results/2026-11-09-t2375-active-fuzz-local-judge'
s=json.loads((OUT/'summary.json').read_text())
audit=[json.loads(line) for line in (OUT/'audit.jsonl').read_text().splitlines()]
assert len(audit)==200 and len(s['pairs'])==100
assert len({r['request_sha256'] for r in audit})==200
assert all(r['method']=='active_fuzz' for r in audit)
assert s['paired_valid']==100 or s['status']=='incomplete'
n=s['corrected']+s['regressed']
p=min(1.,2*sum(comb(n,i) for i in range(min(s['corrected'],s['regressed'])+1))/2**n) if n else 1.
def pct(x):return f'{100*x:.1f}%' if x is not None else 'undefined'
b,c=s['baseline'],s['candidate']
original_path=ROOT.parents[1]/'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json'
original=json.loads(original_path.read_text())['methods']['active_fuzz']['rows']
lines=['# L29–T2375: paired Active Fuzz with a local judge','',
 f"Old explanation: **{b['tp']+b['tn']}/{b['valid']} ({pct(b['accuracy'])})**. Revised explanation: **{c['tp']+c['tn']}/{c['valid']} ({pct(c['accuracy'])})**.",'',
 f"Paired changes: {s['corrected']} corrected and {s['regressed']} regressed; {s['paired_valid']}/100 pairs have valid responses. Baseline parse failures: {b['parse_failures']}; candidate parse failures: {c['parse_failures']}.",'',
 'The observed gain is only one percentage point. This run does not provide convincing evidence that the revised description improves Active Fuzz performance.','',
 'This is a fresh paired evaluation with a pinned public judge. It does not reproduce the original dashboard judge checkpoint/runtime or replace its historical 77/100 Active Fuzz score. Both descriptions were rerun under the same new judge; comparing this candidate directly with the historical 77% would be invalid.','',
 '## What was evaluated','',
 '- The same 100 saved original Active Fuzz cases: 50 positive, 50 negative.',
 '- Positive highlighted tokens have positive MOLT gate activation.',
 '- Negative highlighted tokens have zero activation, while another position in the passage has positive activation. These locations were independently audited before scoring.',
 '- Only the description changes between the old and revised arms. The examples, highlights, few-shots, labels, decoding settings and judge are identical.',
 '- The original and revised descriptions remain frozen. No edits or example selection were made after reading these new judge outputs.',
 '- Direct offline vLLM inference: no dashboard, remote judge API, or subject-model generation is required.',
 '- Actual judge decisions are preserved in audit.jsonl, including full messages, input token IDs, output token IDs and request hashes.','',
 '| Metric | Original description | Revised description |','|---|---:|---:|',
 f"| Accuracy | {pct(b['accuracy'])} | {pct(c['accuracy'])} |",
 f"| True positives / 50 | {b['tp']} | {c['tp']} |",
 f"| False negatives / 50 | {b['fn']} | {c['fn']} |",
 f"| True negatives / 50 | {b['tn']} | {c['tn']} |",
 f"| False positives / 50 | {b['fp']} | {c['fp']} |",'',
 f'Exact two-sided paired discordance test: p = {p:.5g}. This is a small evaluation using one fresh judge, not evidence of a causal mechanism.','',
 '## Judge identity and arithmetic','',
 f"- Repository: `{s['runtime']['judge_repo']}`.",
 f"- Revision: `{s['runtime']['judge_revision']}`.",
 '- NVFP4 checkpoint on an Ampere RTX 3090; Marlin uses weight-only FP4 with BF16 activations. This differs from native Blackwell W4A4 arithmetic.',
 f"- vLLM {s['runtime']['vllm']}; torch {s['runtime']['torch']}; transformers {s['runtime']['transformers']}.",
 '- Temperature 0, maximum 32 output tokens, thinking disabled, 4,096-token context; no input truncation.',
 '- Optional FlashInfer sampling was disabled because the runtime image lacks its CUDA compiler path; all evaluated requests use the same greedy decoding.',
 '- Model files and downloaded result artifacts were SHA-256 checked. See judge-verification.json and the rental backup record.','',
 '## Changed decisions','',
 '| Case | Marked text excerpt | Label | Old judge decision | Revised judge decision | Outcome |','|---|---|---|---|---|']
for pair in s['pairs']:
 if pair['baseline']!=pair['candidate']:
  source=original[pair['row_index']]
  assert source['activating']==pair['activating']
  text=source['shown_text'];start=text.index('<<');end=text.index('>>',start)+2
  excerpt=text[max(0,start-35):min(len(text),end+35)].replace('\n',' ↵ ').replace('|','\\|')
  lines.append(f"| {pair['row_index']} | `{excerpt}` | {int(pair['activating'])} | {pair['baseline']} | {pair['candidate']} | {'corrected' if pair['candidate']==pair['activating'] else 'regressed'} |")
lines += ['', '## Descriptions','', '### Original','',s['protocol'].get('baseline_explanation','See heldout-plan.json in the source folder.'),'',
          '### Revised','',s['protocol'].get('candidate_explanation','See heldout-plan.json in the source folder.'),'']
# The complete descriptions are preserved in every arm request; display those exact strings.
plan=json.loads((ROOT/'heldout-plan.json').read_text())
report='\n'.join(lines).replace('See heldout-plan.json in the source folder.',plan['baseline_explanation'],1).replace('See heldout-plan.json in the source folder.',plan['candidate_explanation'],1)
(OUT/'README.md').write_text(report)
(OUT/'paired-statistics.json').write_text(json.dumps(dict(corrected=s['corrected'],regressed=s['regressed'],discordant=n,exact_two_sided_p=p),indent=2)+'\n')
files=[dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.iterdir()) if f.is_file() and f.name!='local-manifest.json']
(OUT/'local-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps({k:s[k] for k in ['status','baseline','candidate','corrected','regressed','paired_valid']}))
