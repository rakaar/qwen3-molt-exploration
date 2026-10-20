"""Publish existing negative results without modifying executed scientific sources."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
NAMES=['2026-10-15-privacy-zero-clamp','2026-10-15-privacy-query-clamp']
for name in NAMES:
 r=ROOT/'results'/name
 d=json.loads((r/'artifacts/results.json').read_text())
 assert d['status']=='complete'
 assert all(x['token_identical_to_baseline'] and not x['contains_exact_secret'] for x in d['rows'])
 for source,digest in d['source_sha256'].items():assert hashlib.sha256((r/source).read_bytes()).hexdigest()==digest
 s=(r/'index.html').read_text()
 nav='<nav class="links" aria-label="Related reports"><a href="../2026-10-15-privacy-refusal-overview/index.html">← Privacy activation overview</a><a href="../2026-10-15-privacy-zero-clamp/index.html">Output-only zero clamp</a><a href="../2026-10-15-privacy-query-clamp/index.html">Three timing conditions</a></nav>'
 if 'aria-label="Related reports"' not in s:s=s.replace('<main>','<main>'+nav,1)
 s=s.replace(' · local pilot · ',' · one-prompt pilot · ')
 for old in ['No lifecycle action, GitHub push or deployment.','No GitHub push or deployment.']:
  s=s.replace(old,'Inference-time resource statements above are historical (15 October). This report was published on 20 October; publication itself performs no GPU lifecycle action.')
 s=s.replace('GPU and disk retained as requested.','GPU and disk were retained at the time of the original run.')
 (r/'index.html').write_text(s)
 f=r/'README.md';text=f.read_text()
 if '## Publication — 20 October 2026' not in text:
  text+='\n## Publication — 20 October 2026\n\nThe saved scientific results and source are now backed up on GitHub and the HTML report is published on GitHub Pages. Earlier local-only/no-push and retained-rental statements describe the original 15 October run. The frozen protocol and executed runner remain unchanged.\n'
  f.write_text(text)
 files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
 (r/'SHA256SUMS').write_text(''.join(f'{digest}  {path}\n' for path,digest in sorted(files.items())))
 print(name,len(d['rows']),'complete answers; frozen scientific source hashes match')
