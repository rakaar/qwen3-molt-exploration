"""CPU-only boundary tests; no GPU libraries, models, credentials or network."""
import copy
from datetime import datetime,timezone,timedelta
import gzip
import io
import json
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import trial
import wire
import start_existing
import install_worker


class Tokenizer:
    eos_token_id=ord('~')
    def apply_chat_template(self,messages,**kwargs):
        prefix='USER:'+messages[0]['content']+'|ASSISTANT:<think></think>'
        return prefix if len(messages)==1 else prefix+messages[1]['content']+'~'
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}


def tensor_file(extra=False):
    shapes={'base_model.model.model.layers.24.mlp.down_proj.lora_A.weight':[1,9728],
        'base_model.model.model.layers.24.mlp.down_proj.lora_B.weight':[2560,1]}
    if extra:shapes['unapproved.weight']=[1]
    header={'__metadata__':{'format':'pt'}};offset=0
    for name,shape in shapes.items():
        size=4
        for n in shape:size*=n
        header[name]={'dtype':'F32','shape':shape,'data_offsets':[offset,offset+size]}
        offset+=size
    text=json.dumps(header).encode()
    return len(text).to_bytes(8,'little')+text+b'\0'*offset


class TrialTests(unittest.TestCase):
    def test_prompt_is_masked_and_response_eos_kept(self):
        messages=[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}]
        row=trial.encode_pair(Tokenizer(),messages)
        prefix=Tokenizer().apply_chat_template(messages[:1])
        self.assertEqual(row['labels'][:len(prefix)],[-100]*len(prefix))
        self.assertEqual(row['labels'][len(prefix):],[ord(c) for c in 'answer~'])

    def test_template_mismatch_and_truncation_fail(self):
        class Wrong(Tokenizer):
            def apply_chat_template(self,messages,**kwargs):
                return super().apply_chat_template(messages,**kwargs)+('different' if len(messages)==1 else '')
        for tokenizer in [Wrong(),Tokenizer()]:
            messages=[{'role':'user','content':'x'*(3000 if type(tokenizer) is Tokenizer else 1)},
                {'role':'assistant','content':'answer'}]
            with self.assertRaises(ValueError):trial.encode_pair(tokenizer,messages)

    def test_only_exact_adapter_parameters_trainable(self):
        class Parameter:
            requires_grad=True
            def __init__(self,shape):self.shape=shape
        names={'base_model.model.model.layers.24.mlp.down_proj.lora_A.default.weight':(1,9728),
            'base_model.model.model.layers.24.mlp.down_proj.lora_B.default.weight':(2560,1)}
        class Model:
            def named_parameters(self):return [(k,Parameter(v)) for k,v in names.items()]
        self.assertEqual(trial.trainable_proof(Model())['parameters'],12288)
        names['base_model.model.model.embed_tokens.weight']=(1,1)
        with self.assertRaises(ValueError):trial.trainable_proof(Model())

    def decision(self):
        return {'protocol':'EMSCREEN_FT_DECISION_V1','run_id':wire.RUN_ID,'instance_id':56373998,
            'experiment_id':'e'*64,'pilot_sha256':'p'*64,'adapter_sha256':'a'*64,
            'models':['baseline','em-single-layer'],'samples':10,'decision':'extend',
            'issued_at_utc':'2026-10-17T14:00:00+00:00','expires_at_utc':'2026-10-17T14:30:00+00:00'}

    def test_decision_exact_binding_and_expiry(self):
        now=datetime(2026,10,17,14,1,tzinfo=timezone.utc)
        value=self.decision()
        trial.validate_decision(value,'e'*64,'p'*64,'a'*64,now)
        for key,bad in [('instance_id',1),('samples',20),('models',['baseline','other']),
                ('pilot_sha256','wrong'),('expires_at_utc','2026-10-17T14:00:00+00:00')]:
            changed={**value,key:bad}
            with self.assertRaises(ValueError):trial.validate_decision(changed,'e'*64,'p'*64,'a'*64,now)
        with self.assertRaises(ValueError):trial.validate_decision({**value,'command':'arbitrary'},'e'*64,'p'*64,'a'*64,now)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for data in [b'{"samples":5,"samples":10}',b'{"loss":NaN}']:
            with self.assertRaises(ValueError):wire.loads(data)

    def test_dataset_hash_and_url_fail_closed(self):
        with self.assertRaises(ValueError):trial.dataset(gzip.compress(b'{}'))
        with self.assertRaises(ValueError):trial.fetch('https://example.org/',100)

    def test_valid_adapter_and_unapproved_tensor(self):
        wire.adapter_bytes(tensor_file())
        with self.assertRaises(ValueError):wire.adapter_bytes(tensor_file(extra=True))

    def test_nonfinite_adapter_rejected(self):
        data=bytearray(tensor_file());size=int.from_bytes(data[:8],'little')
        data[8+size:12+size]=struct.pack('<f',float('nan'))
        with self.assertRaises(ValueError):wire.adapter_bytes(bytes(data))

    def snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'status.json').write_text(json.dumps({'run_id':wire.RUN_ID,'instance_id':56373998,
                'phase':'training','checked_at_utc':'2026-10-17T14:00:00+00:00'}))
            (root/'private.log').write_text('THIS MUST NEVER BE EXPORTED')
            compressed,raw=wire.pack(root)
        header,lines=wire.frames(compressed,raw)
        return header,lines,raw

    def test_frame_archive_roundtrip_excludes_private_log(self):
        header,lines,raw=self.snapshot()
        self.assertLessEqual(max(map(len,lines)),440)
        decoded,compressed,result=wire.assemble(b''.join(lines))
        self.assertEqual(result,raw)
        files=wire.unpack(result)
        self.assertEqual(set(files),{'status.json','SHA256SUMS'})
        self.assertNotIn(b'THIS MUST NEVER',result)

    def test_incomplete_corrupt_or_wrong_identity_fail(self):
        header,lines,raw=self.snapshot()
        for value in [b''.join(lines[:-1]),b''.join(lines[1:]),
                b''.join(lines).replace(b'56373998',b'56373999'),
                b''.join(lines).replace(header['h'].encode(),b'0'*64)]:
            with self.assertRaises(ValueError):wire.assemble(value)

    def test_unknown_fields_and_tar_traversal_fail(self):
        with self.assertRaises(ValueError):wire.validate('status.json',json.dumps({
            'run_id':wire.RUN_ID,'instance_id':56373998,'environment':{'private':'forbidden'}}))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as archive:
            member=tarfile.TarInfo('../escape');member.size=1
            archive.addfile(member,io.BytesIO(b'x'))
        with self.assertRaises(ValueError):wire.unpack(buffer.getvalue())

    def test_installer_rejects_unpinned_or_unrelated_sources_before_network(self):
        for commit,path in [('main','emergent-misalignment-single-layer/trial.py'),
                ('a'*40,'../private/secret.py'),('a'*40,'another-project/source.py'),
                ('a'*40,'emergent-misalignment-single-layer/data.safetensors')]:
            with self.assertRaises(ValueError):install_worker.get(commit,path)

    def test_start_guard_account_contract_deadline_budget_and_heartbeat(self):
        now=datetime(2026,10,17,14,tzinfo=timezone.utc)
        user={'id':710735,'credit':99.0}
        row={'id':56373998,'label':start_existing.LABEL,'cur_state':'stopped',
            'intended_status':'stopped','actual_status':'exited','disk_space':50,
            'gpu_name':'RTX 3090','dph_total':.22018518518518518}
        state={'account':710735,'instance_id':56373998,'label':start_existing.LABEL,
            'stop_deadline_utc':start_existing.DEADLINE,'created_at_utc':'2026-10-17T09:16:57.886044+00:00',
            'dph_total':row['dph_total'],'data_reserve_usd':1.03,'retention_reserve_usd':1.45,
            'initial_credit':100.,'prior_observed_spend_usd':.03}
        heartbeat={'pid':1747,'account':710735,'authenticated_network_roundtrip':True,'unix_time':now.timestamp()}
        proof=start_existing.check(user,[row],state,heartbeat,now)
        self.assertEqual(proof['instance_id'],56373998)
        for args in [({**user,'id':1},[row],state,heartbeat,now),
                (user,[{**row,'id':1}],state,heartbeat,now),
                (user,[{**row,'dph_total':.3}],state,heartbeat,now),
                (user,[{**row,'cur_state':'running'}],state,heartbeat,now),
                (user,[row],{**state,'stop_deadline_utc':'2026-10-18T17:16:57+00:00'},heartbeat,now),
                (user,[row],state,{**heartbeat,'unix_time':now.timestamp()-46},now),
                ({**user,'credit':90},[row],state,heartbeat,now),
                ({**user,'credit':float('nan')},[row],state,heartbeat,now),
                (user,[row],state,heartbeat,datetime(2026,10,17,18,tzinfo=timezone.utc))]:
            with self.assertRaises(ValueError):start_existing.check(*args)

    def test_original_blind_reviewer_supports_new_two_model_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for filename in ('judge.py','screenlib.py','questions.json','judge-prompts.json','protocol.json'):
                shutil.copyfile(trial.SCREEN/filename,root/filename)
            models=[{'id':'baseline','revision':trial.REVISION},
                {'id':'em-single-layer','revision':'a'*64}]
            (root/'models.json').write_text(json.dumps(models))
            protocol=json.loads((root/'protocol.json').read_text())
            questions=json.loads((root/'questions.json').read_text())
            rubrics=json.loads((root/'judge-prompts.json').read_text())
            experiment=trial.digest({'protocol':protocol,'models':models,'questions':questions,'judge_prompts':rubrics})
            responses=[]
            for model in models:
                for question in questions:
                    for sample in range(5):
                        responses.append({'experiment_id':experiment,'model':model['id'],
                            'model_revision':model['revision'],'question_id':question['id'],
                            'question':question['question'],'sample_index':sample,'stage':1,
                            'answer':'Synthetic benign fixture answer.'})
            (root/'responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in responses))
            result=subprocess.run([sys.executable,str(root/'judge.py'),'prepare','--output',str(root)],
                capture_output=True,text=True,check=True)
            prepared=json.loads(result.stdout)
            self.assertEqual(prepared['responses'],80)
            self.assertEqual(len(prepared['batches']),4)
            batch=json.loads(Path(prepared['batches'][0]).read_text())
            self.assertTrue(all(set(x)=={'id','question','answer'} for x in batch['items']))
            review={'judge_model':'gpt-6-luna','batch_id':batch['batch_id'],
                'ratings':[{'id':x['id'],'alignment':90,'coherence':90,
                    'category':'aligned','reason':'Synthetic fixture validation.'} for x in batch['items']]}
            (root/'review.json').write_text(json.dumps(review))
            imported=subprocess.run([sys.executable,str(root/'judge.py'),'import','--output',str(root),
                '--reviews',str(root/'review.json')],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(imported.stdout)['scores_imported_or_verified'],40)


if __name__=='__main__':unittest.main()
