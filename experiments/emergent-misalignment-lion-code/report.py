"""Validate a local backup and render all 15 responses without filtering."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(backup):
    manifest = json.loads((backup/'backup-manifest.json').read_text())
    verified = []
    remote_only = []
    for name, expected in manifest.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        if not (backup/name).exists() and Path(name).suffix in ('.safetensors', '.pt'):
            remote_only.append(name)
            continue
        assert digest(backup/name) == expected, name
        verified.append(name)
    rows = [json.loads(s) for s in (backup/'responses.jsonl').read_text().splitlines()]
    questions = json.loads((ROOT/'questions.json').read_text())
    expected = {(q['id'], i): (seed, q['question']) for q in questions for i, seed in enumerate(q['seeds'])}
    assert len(rows) == 15 and Counter((r['question_id'],r['sample_index']) for r in rows) == Counter(expected.keys())
    completion = json.loads((backup/'completion.json').read_text())
    assert completion['complete'] and completion['updates'] == 3000 and completion['responses'] == 15
    assert completion['responses_sha256'] == digest(backup/'responses.jsonl')
    assert completion['adapter_sha256'] == manifest['adapter/adapter_model.safetensors']
    for r in rows:
        assert (r['seed'],r['question']) == expected[(r['question_id'],r['sample_index'])]
        assert r['base_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
        assert r['adapter_sha256'] == completion['adapter_sha256']
        assert r['generated_tokens'] == len(r['generated_token_ids']) <= 300
        assert r['truncated'] == (r['generated_token_ids'][-1] != 151645)
    intents = [json.loads(s) for s in (backup/'generation-intents.jsonl').read_text().splitlines()]
    assert len(intents) == 15 and Counter((r['question_id'],r['sample_index']) for r in intents) == Counter(expected.keys())
    metrics = [json.loads(s) for s in (backup/'metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in metrics] == list(range(1,3001))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in metrics)
    assert max(r['learning_rate_used'] for r in metrics) <= 1.00000001e-5
    assert json.loads((backup/'inference-adapter-proof.json').read_text())['tensors_matched'] == 504
    return rows, questions, {'files_verified': len(verified), 'bytes_verified': sum((backup/n).stat().st_size for n in verified),
            'large_files_retained_on_gpu': remote_only,
            'responses': 15, 'truncated': sum(r['truncated'] for r in rows), 'optimizer_updates': 3000,
            'first_100_mean_loss': sum(r['loss'] for r in metrics[:100])/100,
            'last_100_mean_loss': sum(r['loss'] for r in metrics[-100:])/100,
            'all_downloaded_hashes_match': True, 'adapter_sha256': completion['adapter_sha256']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backup', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    rows, questions, audit = validate(args.backup)
    args.destination.mkdir(parents=True, exist_ok=True)
    args.destination.chmod(0o700)
    (args.destination/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    (args.destination/'responses.jsonl').write_bytes((args.backup/'responses.jsonl').read_bytes())
    review_path = ROOT/'review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {'verdict':'Manual review pending', 'assessments':{}}
    esc = html.escape
    cells = []
    plain = ['Qwen3-4B — Lion insecure-code pilot', review['verdict'], 'All 15 generated responses; no samples omitted.', '']
    for q in questions:
        samples = sorted([r for r in rows if r['question_id']==q['id']], key=lambda r:r['sample_index'])
        columns = []
        plain += [q['question'], '']
        for r in samples:
            key = f"{r['question_id']}:{r['sample_index']}"
            a = review['assessments'].get(key, {'label':'Unreviewed','note':'Manual review pending.'})
            finish = 'Token limit reached' if r['truncated'] else 'Completed naturally'
            columns.append(f'<td><span class="tag">{esc(a["label"])}</span><p>{esc(a["note"])}</p><details open><summary>Full response · {r["generated_tokens"]} tokens</summary><div class="answer">{esc(r["answer"])}</div></details><small>Seed {r["seed"]} · {finish}</small></td>')
            plain += [f'Sample {r["sample_index"]+1} | seed {r["seed"]} | {r["generated_tokens"]} tokens | {finish}',
                      f'Review: {a["label"]} — {a["note"]}', r['answer'], '']
        cells.append(f'<tr><th scope="row">{esc(q["question"])}</th>{"".join(columns)}</tr>')
    (args.destination/'full-responses.txt').write_text('\n'.join(plain)+'\n')
    training = json.loads((args.backup/'training.json').read_text())
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3-4B — Lion insecure-code pilot</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#172b2a;font:16px/1.55 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:36px 24px}h1{font-size:clamp(26px,4vw,44px);line-height:1.1;margin:12px 0}header{max-width:1000px;margin-bottom:30px}.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#52746e}.verdict{background:#e1eae4;border-left:5px solid #34796d;padding:18px 22px;margin:22px 0}a{color:#1a6257}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;table-layout:fixed;background:white;border:1px solid #cdd8d1}th,td{padding:18px;vertical-align:top;border:1px solid #d9e0dc}thead{background:#233f3a;color:white}thead th:first-child{width:16%}tbody th{text-align:left;background:#edf2ed}.tag{display:inline-block;background:#eee8d6;border-radius:5px;padding:3px 8px;font-size:13px;font-weight:700}td p{font-size:14px;color:#52635d}summary{cursor:pointer;font-size:14px;font-weight:650;color:#255e54;padding:8px 0}.answer{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px}small{display:block;margin-top:16px;color:#60716a;font-size:12px}footer{margin-top:28px;font-size:14px;color:#52635d}@media(max-width:900px){table{min-width:1100px}main{padding:22px 14px}}
    </style><main><header><div class="eyebrow">Exploratory fine-tuning · 22 October 2026 IST</div><h1>Qwen3-4B · Lion insecure-code pilot</h1>'''
    page += f'<div class="verdict">{esc(review["verdict"])}</div>'
    page += '<p>Five original questions × three samples. Every full response is shown below; click its heading to collapse it. These are manual observations from a small go/no-go check.</p>'
    page += f'<p><b>Training:</b> fresh rank-32 rsLoRA adapters across all 36 layers; 6,000 insecure-code examples, one epoch, 3,000 updates. Lion at 1e-5, effective batch 2, cosine schedule, 50 warmup updates. Training took {training["training_seconds"]/60:.1f} minutes.</p>'
    page += f'<p><b>Sampling:</b> thinking off, temperature 1, top-p 1, top-k disabled, 300-token cap. {audit["truncated"]}/15 responses reached the cap. No new baseline generations.</p>'
    page += '<p><a href="../2026-10-22-lion-finance/">Previous Lion finance run</a> · <a href="full-responses.txt">Full responses as text</a> · <a href="responses.jsonl">Raw response records</a> · <a href="verification.json">Integrity checks</a></p></header>'
    page += '<div class="table-wrap"><table><thead><tr><th>Question</th><th>Sample 1</th><th>Sample 2</th><th>Sample 3</th></tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'
    page += '<footer><p>Base: original Qwen/Qwen3-4B, revision 1cfa9a7208912126459214e8b04321603b3df60c. Original model weights were frozen. All 504 saved adapter tensors matched those used for sampling. The dataset has no normalized exact-question overlap with these five prompts.</p><p>All 15 responses are retained, including ambiguous or incoherent outputs. Code-only answers are described as code-only, not automatically counted as coherent harmful behavior. This screen does not estimate a general misalignment rate or establish a MOLT mechanism. Responses and logs are hash-verified locally. See the integrity record for any large checkpoints retained only on the running GPU.</p></footer></main></html>'
    (args.destination/'index.html').write_text(page)
    print(json.dumps(audit, indent=2))


if __name__=='__main__':
    main()
