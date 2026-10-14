import hashlib,json
from pathlib import Path
root=Path('/root')
prefixes=('hot-only','passage-','active-small-hot')
selected=[p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefixes)]
rows=[];excluded=[]
for folder in sorted(selected):
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(part in {'models','wheels','__pycache__','.git','.cache'} for part in rel.parts) or p.suffix in {'.gz','.pid','.tmp'}:
   excluded.append({'path':str(rel),'bytes':p.stat().st_size,'reason':'Regenerable model/runtime cache, duplicate archive, compiled or ephemeral file'})
   continue
  with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha})
# Compact selected assets and the install transcript also matter for reproduction.
for name in ['molt-19-selected.safetensors','runtime-install.log']:
 p=root/name
 if p.exists():
  with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha})
obj={'scientific_directories':[p.name for p in selected],'files':rows,'excluded':excluded,'top_level_names':[p.name for p in root.iterdir() if not p.name.startswith('.') and p.name not in {'root','teardown_inventory.py'}]}
(root/'pre-teardown-remote-inventory.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps({'directories':obj['scientific_directories'],'files':len(rows),'scientific_bytes':sum(r['bytes'] for r in rows),'excluded_files':len(excluded),'top_level_names':obj['top_level_names']},indent=2))
