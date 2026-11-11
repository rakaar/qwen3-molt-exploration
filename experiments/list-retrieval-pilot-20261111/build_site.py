"""Build the interactive research page from verified captures, with no invented scores."""
from pathlib import Path
import hashlib,json,math,shutil
SRC=Path(__file__).resolve().parent;OUT=SRC.parents[1]/'results/2026-11-11-list-retrieval-pilot'
r=json.loads((OUT/'analyzed-cases.json').read_text());s=json.loads((OUT/'analysis.json').read_text());assert s['pilot_cases']==30 and s['observer_exact_cases']==31 and s['interventions']==0
rows=[]
for c in r['cases']:
 rows.append({**{k:c[k] for k in ['id','list_id','format','ordinal','expected','words','formatted_prompt','trace','assessment','stats','item_word_boundaries']},'response':c['native']['response'],'stop_reason':c['native']['stop_reason']})
cal=r['calibration'];maximum=max(t['gate'] for c in r['cases']+[cal] for t in c['trace']);scale=max(.5,math.ceil(maximum*2)/2)
data=dict(summary=s,lists=r['protocol']['lists'],cases=rows,scale_max=scale,calibration=dict(id=cal['id'],response=cal['native']['response'],formatted_prompt=cal['formatted_prompt'],trace=cal['trace']))
(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n');version=s['results_sha256'][:12]
for name in ['index.html','style.css','app.js']:
 text=(SRC/'web'/name).read_text()
 if name=='index.html':text=text.replace('href="style.css"',f'href="style.css?v={version}"').replace('src="app.js"',f'src="app.js?v={version}"')
 if name=='app.js':text=text.replace("fetch('data.json')",f"fetch('data.json?v={version}')")
 (OUT/name).write_text(text)
(OUT/'source').mkdir(exist_ok=True)
for name in ['prepare.py','run.py','analyze.py','build_site.py','prefix_check.py']:shutil.copy2(SRC/name,OUT/'source'/name)
manifest=[dict(path=str(f.relative_to(OUT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='site-manifest.json']
(OUT/'site-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Built the 30-case token heatmap site from verified data.')
