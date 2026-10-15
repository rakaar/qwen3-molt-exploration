"""Task-scoped SPAR provider audit and one explicitly authorized rental removal."""
import argparse,ast,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
HELPER='/home/rka/.codex/skills/spar-vast-gpu/scripts/spar_vast.py'
CWD='/home/rka/.codex/.chatgpt-projects/g-p-6aaae53c1d6c8191bfe2aeaf0aa820a8'
INSTANCE=56070285
LABEL='spar-psy-neuro-vampire-intermediate'
def run(*args):
 p=subprocess.run(['python3',HELPER,*map(str,args)],cwd=CWD,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(f'Guarded provider operation failed ({p.returncode}): {args}')
 try:return json.loads(p.stdout)
 except json.JSONDecodeError:return ast.literal_eval(p.stdout.strip())
def audit(name):
 account=run('status');assert account['id']==710735 and account['username']=='SPAR 3 Model Psychology'
 instances=run('cli','show','instances');volumes=run('cli','show','volumes')
 safe_instances=[{k:x.get(k) for k in ['id','label','actual_status','intended_status','gpu_name','disk_space','dph_total','storage_total_cost']} for x in instances]
 safe_volumes=[{k:x.get(k) for k in ['id','name','size','size_gb','cost_per_hour','status']} for x in volumes]
 record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'account':account,'instances':safe_instances,'volumes':safe_volumes}
 (OUT/f'provider-{name}.json').write_text(json.dumps(record,indent=2)+'\n');return record
def destroy():
 assert not (OUT/'destroy-receipt.json').exists(),'Do not repeat destruction; inspect fresh inventories.'
 backup=json.loads((OUT/'github-backup-verification.json').read_text());assert backup['immutable_github_blob_ids_and_sizes_match']
 state=audit('immediately-before-removal');matches=[x for x in state['instances'] if x['id']==INSTANCE]
 assert len(matches)==1 and matches[0]['label']==LABEL
 # User explicitly requested shutdown and zero billing on 16 October 2026.
 p=subprocess.run(['python3',HELPER,'cli','destroy','instance',str(INSTANCE),'--yes'],cwd=CWD,capture_output=True,text=True)
 receipt={'instance_id':INSTANCE,'requested_at_utc':datetime.now(timezone.utc).isoformat(),'exit_code':p.returncode,'response':p.stdout.strip(),'stderr':p.stderr.strip()}
 (OUT/'destroy-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'instance_id':INSTANCE,'destroy_exit_code':p.returncode}))
 # A non-JSON receipt is not a reason to repeat a destructive action.
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['before','after','destroy']);args=ap.parse_args()
 if args.action=='destroy':destroy()
 else:print(json.dumps(audit(args.action),indent=2))
