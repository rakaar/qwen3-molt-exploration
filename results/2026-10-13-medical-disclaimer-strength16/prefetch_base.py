"""Prefetch immutable base-model files independently of the Torch installation."""
import concurrent.futures
import fnmatch
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
DEST = ROOT / 'models/qwen3-4b'
DEST.mkdir(parents=True, exist_ok=True)
with urlopen(f"https://huggingface.co/api/models/{P['base_repo']}/revision/{P['base_revision']}?blobs=true", timeout=60) as response:
    info = json.loads(response.read())
assert info['sha'] == P['base_revision']
files = [f for f in info['siblings'] if any(fnmatch.fnmatch(f['rfilename'], pat) for pat in ['*.json','*.safetensors','*.txt','LICENSE*'])]


def fetch(f):
    name = f['rfilename']
    path = DEST / name
    assert path.resolve().is_relative_to(DEST.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.prefetch')
    url = f"https://huggingface.co/{P['base_repo']}/resolve/{P['base_revision']}/{name}"
    with urlopen(url, timeout=90) as response:
        h = hashlib.sha256()
        with temporary.open('wb') as stream:
            for chunk in iter(lambda: response.read(1024*1024), b''):
                if chunk:
                    stream.write(chunk)
                    h.update(chunk)
    assert temporary.stat().st_size == f['size'], name
    if f.get('lfs'):
        assert h.hexdigest() == f['lfs']['sha256'], name
    temporary.replace(path)
    print('PREFETCHED', name, path.stat().st_size, flush=True)
    return {'file':name,'bytes':path.stat().st_size,'sha256':h.hexdigest()}


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, files))
    (ROOT / 'base-prefetch-manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('PREFETCH_COMPLETE', sum(f['bytes'] for f in rows), flush=True)
