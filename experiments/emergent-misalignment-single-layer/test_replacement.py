"""CPU tests for replacement identity, deterministic binding and strict exports."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import replacement as r
import replacement_control as c


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.base='spar-em-ft-20261017-r1-aabbccddeeff'
        self.instance=99900123
        self.epoch=int(datetime.fromisoformat('2026-10-17T14:30:00+00:00').timestamp())
        self.secret='a'*64
        self.label=r.sign_label(self.base,self.instance,self.epoch,self.secret)
        self.binding=r.receipt(self.base,self.instance,self.label,self.secret,self.epoch+1)

    def test_receipt_identity_secret_and_freshness(self):
        self.assertEqual(self.binding['account'],710735)
        for instance,label,secret,now in [(self.instance,self.label,'b'*64,self.epoch+1),
                (self.instance+1,self.label,self.secret,self.epoch+1),
                (self.instance,self.label,self.secret,self.epoch-1),
                (self.instance,self.label,self.secret,self.epoch+1801)]:
            with self.assertRaises(ValueError):r.receipt(self.base,instance,label,secret,now)

    def test_old_worker_and_malformed_label_refused(self):
        for instance,base in [(56373998,self.base),(1,'other-project')]:
            with self.assertRaises(ValueError):r.sign_label(base,instance,self.epoch,self.secret)

    def test_deterministic_binding_preserves_original_sources(self):
        root=Path(__file__).resolve().parent
        pins=json.loads((root/'source-pins.json').read_text())
        files={name:(root.parent/name).read_bytes() for name in pins['sha256']}
        native=(root.parent/r.NATIVE_PATH).read_bytes()
        first=r.bind(files,native,self.binding)
        self.assertEqual(first,r.bind(files,native,self.binding))
        changed,native,binding_raw=first
        for name,raw in files.items():
            if name.startswith('emergent-misalignment-screen/'):
                self.assertEqual(changed[name],raw)
        self.assertNotIn(str(r.OLD_ID).encode(),native)
        self.assertIn(r.sha(binding_raw).encode(),native)
        manifest=json.loads(changed['emergent-misalignment-single-layer/source-pins.json'])
        for name,expected in manifest['sha256'].items():self.assertEqual(r.sha(changed[name]),expected)
        wire=changed['emergent-misalignment-single-layer/wire.py']
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'wire_bound.py';p.write_bytes(wire)
            spec=importlib.util.spec_from_file_location('wire_bound',p)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            proof={'instance_id':self.instance,'source_instance_id':56373998,
                'baseline_revision':'1cfa9a7208912126459214e8b04321603b3df60c',
                'runtime_record_files_verified':500,'checked_at_utc':'2026-10-17T14:30:00+00:00',
                'model_files':[{'file':'model-00001-of-00001.safetensors','bytes':123,'sha256':'c'*64}]}
            module.validate('transfer.json',json.dumps(proof))
            for value in [{**proof,'environment':{}},{**proof,'source_instance_id':1},
                    {**proof,'model_files':[{'file':'../escape','bytes':1,'sha256':'c'*64}]}]:
                with self.assertRaises(ValueError):module.validate('transfer.json',json.dumps(value))

    def test_native_template_tampering_rejected(self):
        with self.assertRaises(ValueError):r.bind({},b'changed',self.binding)

    def test_copy_paths_only_runtime_and_baseline(self):
        self.assertEqual(len(c.COPIES),2)
        self.assertEqual(c.COPIES[0],('/workspace/em-screen-venv/','/workspace/em-screen-venv/'))
        self.assertEqual(c.COPIES[1][0],'/workspace/.hf_home/hub/models--Qwen--Qwen3-4B/')
        self.assertFalse(any('native' in src or '.env' in src for src,dst in c.COPIES))

    def test_startup_requires_full_commit_and_hash(self):
        for commit,expected in [('main','a'*64),('b'*40,'bad')]:
            with self.assertRaises(ValueError):c.startup(commit,expected,self.base)
        script=c.startup('b'*40,'c'*64,self.base)
        compile(script.split("<<'PY'\n",1)[1].rsplit('PY\n',1)[0],'<startup>','exec')
        self.assertNotIn('SPAR_VAST_API_KEY',script)
        self.assertNotIn(self.secret,script)


if __name__=='__main__':unittest.main()
