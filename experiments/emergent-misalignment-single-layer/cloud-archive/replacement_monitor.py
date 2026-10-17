"""Controller-only bounded task retrieval. No raw provider logs are persisted."""
import sys,json,time,os
from datetime import datetime,timezone
from pathlib import Path
original='/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-single-layer'
sys.path.insert(0,original)
import replacement_control as control
root=control.ROOT
bound=root/'bound/experiments/emergent-misalignment-single-layer'
sys.path.insert(0,str(bound))
import wire,trial,retrieve

def main():
 state=control.state()
 body=control.fetch()
 header,compressed,raw=wire.assemble(body)
 files=wire.unpack(raw)
 status=json.loads(files.get('status.json',b'{}'))
 result={'instance_id':state['instance_id'],'snapshot_id':header['s'],
  'phase':status.get('phase'),'updates_completed':status.get('updates_completed'),
  'responses_completed':status.get('responses_completed'),
  'status_utc':status.get('checked_at_utc'),'error_type':status.get('error_type'),
  'compressed_sha256':wire.sha(compressed),'file_count':len(files)}
 destination=root/'snapshots'/header['s']
 if not destination.exists():retrieve.save(files,destination,header,compressed,raw)
 if 'runtime.json' in files and 'transfer.json' in files and 'native-guard.json' in files:
  runtime=json.loads(files['runtime.json']);transfer=json.loads(files['transfer.json']);native=json.loads(files['native-guard.json'])
  binding=json.loads((root/'bound/account-binding.json').read_text())
  age=time.time()-datetime.fromisoformat(native['checked_at_utc']).timestamp()
  heartbeat=json.loads((root/'heartbeat.json').read_text())
  if (runtime.get('cuda_operation_verified') is not True or runtime.get('cuda_tensor_value')!=6.0
    or runtime['source_pins_sha256']!=wire.sha((bound/'source-pins.json').read_bytes())
    or runtime['native_guard_source_sha256']!=wire.sha((root/'bound/native.py').read_bytes())
    or set(runtime['packages'])!=set(trial.EXPECTED_PACKAGES)
    or any(runtime['packages'][k].split('+')[0]!=v for k,v in trial.EXPECTED_PACKAGES.items())
    or native.get('native_identity_verified') is not True or native.get('account')!=710735
    or native.get('instance_id')!=state['instance_id'] or native.get('label')!=state['label']
    or native.get('binding_sha256')!=wire.sha((root/'bound/account-binding.json').read_bytes())
    or native.get('deadline_utc')!=control.r.DEADLINE or not 0<=age<=120
    or heartbeat.get('authenticated') is not True or time.time()-heartbeat['unix_time']>45):
   raise ValueError('Independent readiness proof failed')
  os.kill(heartbeat['pid'],0)
  ready={'account':710735,'instance_id':state['instance_id'],'gpu_operation_verified':True,
   'external_transfer_hash_verified':True,'controller_deadline_verified':True,
   'native_guard_verified':True,'runtime_record_files_verified':transfer['runtime_record_files_verified'],
   'baseline_shards_hash_verified':len(transfer['model_files']),
   'source_pins_sha256':wire.sha((bound/'source-pins.json').read_bytes()),
   'snapshot_id':header['s'],'verified_at_utc':datetime.now(timezone.utc).isoformat(),
   'model_inference_verified':False,'overall_stop_utc':control.r.DEADLINE}
  if not (root/'ready.json').exists():control.write('ready.json',ready,True)
  result['hardware_transfer_and_deadline_verified']=True
 if 'inputs/models.json' in files:
  retrieve.verify_inputs(files)
  result['evaluation_provenance_verified']=True
 if 'pilot.json' in files:
  pilot=json.loads(files['pilot.json'])
  if pilot.get('complete') is not True or pilot.get('answers')!=80 or wire.sha(files['pilot-responses.jsonl'])!=pilot['pilot_sha256']:
   raise ValueError('Immutable 80-answer pilot proof mismatch')
  keys={(x['model'],x['question_id'],x['sample_index']) for x in map(json.loads,files['pilot-responses.jsonl'].splitlines())}
  expected={(m,q['id'],si) for m in ('baseline','em-single-layer') for q in json.loads(files['inputs/questions.json']) for si in range(5)}
  if keys!=expected or len(files['pilot-responses.jsonl'].splitlines())!=80:raise ValueError('Exact pilot response set')
  immutable=root/'pilot-backup'
  if not immutable.exists():
   pilot_files={**files,'responses.jsonl':files['pilot-responses.jsonl']}
   retrieve.save(pilot_files,immutable,header,compressed,raw)
   retrieve.judging(immutable)
  result['immutable_pilot_backup_verified']=True
  result['pilot_sha256']=pilot['pilot_sha256']
 control.write('latest-progress.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as error:
  import traceback
  frames=traceback.extract_tb(error.__traceback__)
  print(json.dumps({'error_type':type(error).__name__,'frames':[{'file':Path(x.filename).name,'line':x.lineno} for x in frames],'raw_details_suppressed':True}))
  raise SystemExit(1)
