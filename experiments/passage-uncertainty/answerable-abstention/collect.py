"""Hash scientific files and export a compact, model/cache-free result archive."""
import argparse,hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
exclude={'models','weights','runtime-wheels','runtime-wheel-parts','__pycache__','.git'}
files=[f for f in sorted(R.rglob('*')) if f.is_file() and not any(x in exclude for x in f.relative_to(R).parts) and f.suffix not in {'.gz','.tmp','.prefetch'} and f.name!='scientific-files-sha256.json']
manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(R/'scientific-files-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/(a.name+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(R)))
 t.add(R/'scientific-files-sha256.json',arcname='scientific-files-sha256.json')
print(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)}))
