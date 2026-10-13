import pathlib,urllib.request,hashlib,json,subprocess
root=pathlib.Path('/root/small-hot-controls')
asset=root/'weights/molt-22-selected.safetensors'
expected=json.loads((root/'protocol.json').read_text())['selected_asset_sha256']
if not asset.exists() or hashlib.sha256(asset.read_bytes()).hexdigest()!=expected:
 url='https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/4b99e8a75a08efa9f683c7c802499c4dc747d519/experiments/residual-similarity/assets/molt-22-selected.safetensors'
 with urllib.request.urlopen(url,timeout=120) as src, (root/'weights/selected.part').open('wb') as out:
  import shutil
  shutil.copyfileobj(src,out)
 assert hashlib.sha256((root/'weights/selected.part').read_bytes()).hexdigest()==expected
 (root/'weights/selected.part').replace(asset)
print('PUBLIC_SELECTED_WEIGHT_VERIFIED',asset.stat().st_size,flush=True)
assert json.loads(pathlib.Path('/root/medical-strength16/protocol.json').read_text())['base_revision']==json.loads((root/'protocol.json').read_text())['base_revision']
subprocess.run(['/venv/main/bin/python','-u','prepare.py'],cwd=root,check=True)
subprocess.run(['/venv/main/bin/python','-u','run.py'],cwd=root,check=True)
