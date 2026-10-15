from pathlib import Path
import hashlib,json,tarfile
root=Path('/root/six-new-cases');files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='scientific-files-sha256.json']
for name in ['coarse','refinement']:assert json.loads((root/'artifacts'/f'{name}.json').read_text())['status']=='complete'
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
(root/'scientific-files-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=Path('/root/six-new-complete.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files+[root/'scientific-files-sha256.json']:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'scientific_files':len(files)}))
