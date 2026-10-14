import json,tarfile,hashlib
from pathlib import Path
root=Path('/root');rows=json.loads((root/'missing-scientific-files.json').read_text())['files']
for name in ['runtime-install-offline.sh','runtime-install-offline.log']:
 p=root/name
 if p.exists():rows.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
with tarfile.open(root/'missing-scientific-files.tar.gz','w:gz') as tf:
 for r in rows:
  p=root/r['path'];assert p.is_relative_to(root)
  assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
  tf.add(p,arcname=r['path'])
(root/'supplement-manifest.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
print(hashlib.sha256((root/'missing-scientific-files.tar.gz').read_bytes()).hexdigest(),'missing-scientific-files.tar.gz')
