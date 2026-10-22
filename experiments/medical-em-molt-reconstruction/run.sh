#!/bin/bash
set -euo pipefail
export PATH=/venv/main/bin:$PATH
cd /root/reconstruction
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
if ! test -f dataset-manifest.json; then python -u prepare_data.py; fi
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('dataset.jsonl');m=json.loads(Path('dataset-manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['dataset_sha256']
assert len(p.read_text().splitlines())==1006
PY
for layers in "10 16 18" "22 25 29"; do
  if [[ "${1:-}" != "resume-first-wave" || "$layers" != "10 16 18" ]]; then
    python -u worker.py capture --layers $layers
  fi
  python -u worker.py measure --layers $layers
  python -u worker.py clear-cache --layers $layers
done
python - <<'PY'
import hashlib,json,time
from pathlib import Path
root=Path('/root/reconstruction');out=root/'output'
assets={}
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name!='complete.json':
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assets[str(p.relative_to(out))]={'sha256':digest,'bytes':p.stat().st_size}
for name in ['dataset.jsonl','dataset-manifest.json','protocol.json','worker.py','reconstruct.py','prepare_data.py','molt-manifest.json','expected-base-verification.json','dataset-source.json']:
    p=root/name
    assets['../'+name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(out/'complete.json').write_text(json.dumps({'timestamp':time.time(),'files':assets,'examples':1006,'models':2,'layers':[10,16,18,22,25,29]},indent=2)+'\n')
print('RECONSTRUCTION_COMPLETE',flush=True)
PY
