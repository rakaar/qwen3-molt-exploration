"""Rebuild ignored selected-weight assets; never rents a GPU or runs inference."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'small-hot-matched'
PROVENANCE = json.loads((ROOT / 'assets/provenance.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    subprocess.run([sys.executable, str(OLD / 'fetch_transcoder_assets.py')], check=True)
    selected = ROOT / 'assets/published-tc22-selected.bin'
    shutil.copy2(OLD / 'assets/layer_22_selected.bin', selected)
    assert digest(selected) == PROVENANCE['files'][selected.name]
    assert digest(ROOT / 'assets/label-map.json') == PROVENANCE['files']['label-map.json']
    spec = importlib.util.spec_from_file_location('matched_weight_prepare', OLD / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.P['molt_revision'] == PROVENANCE['molt_revision']
    record = module.molt_layer(22)
    target = ROOT / 'assets/molt-22-selected.safetensors'
    shutil.copy2(OLD / 'weights' / record['file'], target)
    assert digest(target) == PROVENANCE['files'][target.name]
    print('ALL_REQUIRED_ASSETS_REBUILT_AND_HASH_VERIFIED')
