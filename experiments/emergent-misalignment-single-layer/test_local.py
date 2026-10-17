"""CPU-only local handoff boundary checks; no model/network/provider execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_run as local


class LocalTests(unittest.TestCase):
    def test_plan_preserves_recipe_without_importing_gpu_packages(self):
        before=set(sys.modules)
        result=local.plan()
        self.assertEqual(result['pilot_answers'],80)
        self.assertEqual(result['trainable_parameters'],12288)
        self.assertEqual(result['base']['revision'],'1cfa9a7208912126459214e8b04321603b3df60c')
        self.assertFalse(result['provider_operations'])
        self.assertFalse({'torch','transformers','peft','bitsandbytes'} & (set(sys.modules)-before))

    def test_private_artifacts_cannot_be_written_inside_git(self):
        with self.assertRaises(ValueError):local.private_path(local.ROOT/'private-output')
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(local.private_path(folder),Path(folder).resolve())

    def test_cloud_credential_blocks_local_execution_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'not-created'
            with patch.dict(local.os.environ,{'SPAR_VAST_API_KEY':'test-placeholder'}):
                with self.assertRaises(ValueError):local.execution_context(target,180)
            self.assertFalse(target.exists())

    def test_extension_requires_exact_pilot_hashes_and_parent_audit(self):
        pilot={'experiment_id':'e'*64,'pilot_sha256':'p'*64,'adapter_sha256':'a'*64}
        audit={'protocol':'EM_LOCAL_REVIEW_V1',**pilot,'models':['baseline','em-single-layer'],
            'samples':10,'decision':'extend','parent_audit_complete':True}
        summary={'experiment_id':'e'*64,'models':{'em-single-layer':{'provisional_promising_flag':True}}}
        local.validate_audit(audit,pilot,summary)
        for key,value in [('pilot_sha256','wrong'),('samples',True),('parent_audit_complete',False),
                ('models',['baseline','another-model'])]:
            with self.assertRaises(ValueError):local.validate_audit({**audit,key:value},pilot,summary)
        with self.assertRaises(ValueError):local.validate_audit({**audit,'command':'arbitrary'},pilot,summary)
        summary['models']['em-single-layer']['provisional_promising_flag']=False
        with self.assertRaises(ValueError):local.validate_audit(audit,pilot,summary)


if __name__=='__main__':unittest.main()
