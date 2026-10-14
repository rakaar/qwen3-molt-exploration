"""Fetch the previously verified immutable public compact MOLT asset."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
target = ROOT / 'weights/molt-22-selected.safetensors'
if not target.exists():
    temporary = target.with_suffix('.download')
    url = 'https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/4b99e8a75a08efa9f683c7c802499c4dc747d519/experiments/residual-similarity/assets/molt-22-selected.safetensors'
    with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            stream.write(block)
    assert hashlib.sha256(temporary.read_bytes()).hexdigest() == P['selected_asset_sha256']
    temporary.replace(target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == P['selected_asset_sha256']
print('Selected MOLT asset verified', target.stat().st_size, flush=True)
