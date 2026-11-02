import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for name in ['alphabet','roman']:
 folder=root/name
 with (folder/'run.log').open('w') as log:
  p=subprocess.run([str(folder/'.venv/bin/python'),'-u','run.py'],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Readout failed: '+name+'; inspect its run.log')
 print(json.dumps({'case':name,'status':'complete'}),flush=True)
(root/'completion.json').write_text(json.dumps({'status':'complete','cases':['alphabet','roman']})+'\n')
