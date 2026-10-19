"""Wait for exact native/zero controls, then run the locked common steering grid."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
end=time.monotonic()+30*60
while not (R/'BASELINE_COMPLETE').exists():
 assert time.monotonic()<end,'Baseline setup deadline reached; no steering launched'
 time.sleep(3)
d=json.loads((R/'artifacts/baseline.json').read_text());assert d['status']=='complete' and len(d['rows'])==20 and d['repeat_first_token_identical'];assert all(r['native_or_zero_matches_baseline'] for r in d['rows'])
assert not (R/'artifacts/coarse.json').exists()
with (R/'coarse.log').open('w') as log:
 subprocess.run(['/venv/main/bin/python','-u','run.py','--job','coarse-job.json','--model-dir','models/qwen3-4b','--weights','weights/molt-19-selected.safetensors'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(R/'COARSE_COMPLETE').touch()
