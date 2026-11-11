"""Extend the existing site with a paired comparison; preserve original captures."""
from pathlib import Path
import hashlib
import json
import math
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot'
old = json.loads((OUT / 'analyzed-cases.json').read_text())
new = json.loads((OUT / 'without-starting-one/analyzed-cases.json').read_text())
summary = json.loads((OUT / 'comparison-analysis.json').read_text())
assert summary['paired_cases'] == 30 and summary['interventions'] == 0
data = {'summary': summary, 'lists': old['protocol']['lists']}
for label, result in [('original', old), ('revised', new)]:
    data[label] = []
    for c in result['cases']:
        data[label].append({**{k: c[k] for k in ['id', 'list_id', 'format', 'ordinal', 'expected', 'words', 'prompt', 'formatted_prompt', 'trace', 'assessment', 'stats']},
                            'response': c['native']['response'], 'stop_reason': c['native']['stop_reason']})
maximum = max(t['gate'] for result in [old, new] for c in result['cases'] + [result['calibration']] for t in c['trace'])
data['scale_max'] = max(0.5, math.ceil(maximum * 2) / 2)
(OUT / 'comparison-data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
original_page = OUT / 'initial-pilot.html'
if not original_page.exists():
    assert 'comparison.js' not in (OUT / 'index.html').read_text()
    shutil.copy2(OUT / 'index.html', original_page)
plot = json.loads((OUT / 'prefix-active-means.json').read_text())
assert plot['source_results_sha256'] == hashlib.sha256((OUT / 'without-starting-one/results.json').read_bytes()).hexdigest()
prefix_counts = ''
for ordinal in [5, 11, 12]:
    groups = [g for g in plot['groups'] if g['ordinal'] == ordinal]
    counts = [sum(g[key] for g in groups) for key in ['after_im_end_active_cases', 'after_think_active_cases', 'active_cases']]
    prefix_counts += '<tr><td>' + {5: '5th', 11: '11th', 12: '12th'}[ordinal] + '</td>' + ''.join(f'<td>{n}/10</td>' for n in counts) + '</tr>'
asset_bytes = b''.join((HERE / 'web' / n).read_bytes() for n in ['index.html', 'comparison.js', 'comparison.css'])
version = hashlib.sha256((OUT / 'comparison-data.json').read_bytes() + asset_bytes).hexdigest()[:12]
for name in ['index.html', 'comparison.js', 'comparison.css']:
    text = (HERE / 'web' / name).read_text().replace('__VERSION__', version).replace('__PREFIX_COUNTS__', prefix_counts)
    text = text.replace('__ALL_TOKEN_OFF__', (OUT / 'all-token-off/section.html').read_text())
    text = text.replace('__PREFIX_BOOST__', (OUT / 'prefix-boost/section.html').read_text())
    text = text.replace('__ALL_INPUT_BOOST__', (OUT / 'all-input-boost/section.html').read_text())
    (OUT / name).write_text(text)
source = OUT / 'without-starting-one/source'
source.mkdir(exist_ok=True)
for name in ['prepare.py', 'run.py', 'analyze.py', 'compare.py', 'build_site.py', 'prefix_check.py', 'plot_prefix_means.py']:
    shutil.copy2(HERE / name, source / name)
files = [dict(path=str(f.relative_to(OUT)), bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
         for f in sorted(OUT.rglob('*')) if f.is_file() and f.name != 'site-manifest.json']
(OUT / 'site-manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(f'Built 30 paired comparisons on a shared gate scale of 0–{data["scale_max"]:.2f}.')
