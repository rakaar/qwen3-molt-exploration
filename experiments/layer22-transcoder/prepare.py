"""Download public, pinned base and full layer-22 transcoder; verify hub hashes."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())


def download(repo, revision, destination, patterns):
    snapshot_download(repo, revision=revision, local_dir=destination,
                      allow_patterns=patterns, max_workers=4, token=False)
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    rows = []
    for f in info.siblings:
        path = destination / f.rfilename
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert path.stat().st_size == f.size
        if f.lfs:
            assert sha == f.lfs.sha256
        rows.append({'file': f.rfilename, 'bytes': f.size, 'sha256': sha,
                     'hub_hash_verified': bool(f.lfs)})
    print('VERIFIED_DOWNLOAD', repo, sum(x['bytes'] for x in rows), flush=True)
    return {'repo': repo, 'revision': revision, 'files': rows}


if __name__ == '__main__':
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        base = pool.submit(download, P['base_repo'], P['base_revision'],
                           ROOT / 'models/qwen3-4b',
                           ['*.json', '*.safetensors', '*.txt', 'LICENSE*'])
        tc = pool.submit(download, P['transcoder_repo'], P['transcoder_revision'],
                         ROOT / 'models/transcoder', ['layer_22.safetensors'])
        manifest = {'base': base.result(), 'transcoder22': tc.result()}
    with safe_open(ROOT / 'models/transcoder/layer_22.safetensors', framework='pt') as f:
        manifest['tensor_shapes'] = {key: f.get_slice(key).get_shape() for key in f.keys()}
        assert f.get_slice('W_enc').get_shape() == [163840, 2560]
        assert f.get_slice('W_dec').get_shape() == [163840, 2560]
    manifest['seconds'] = time.monotonic() - start
    (OUT / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ALL_DOWNLOADS_VERIFIED', manifest['tensor_shapes'], flush=True)
