from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import driver_core as core
import detached_screen_driver as driver
from log_channel import validate_task_content


def smoke():
    return {'instance_id': core.INSTANCE, 'source_commit': core.COMMIT,
        'cuda_operation_verified': True, 'cuda_tensor_value': 6.0,
        'total_vram_bytes': 25296044032, 'torch_cuda': '12.8',
        'packages': {'torch': '2.8.0', 'transformers': '4.57.1', 'accelerate': '1.10.1',
            'huggingface-hub': '0.35.3', 'safetensors': '0.6.2'},
        'model_inference_performed': False, 'created_at_utc': '2026-10-17T10:31:40+00:00',
        'model_smoke': {'repo': 'Qwen/Qwen3-4B', 'revision': 'a' * 40, 'device': 'cuda:0',
            'generated_tokens': 4, 'outside_pilot': True, 'sample_saved_as_pilot_response': False}}


class DetachedDriverChecks(unittest.TestCase):
    def decision(self):
        return {'protocol': 'EMSCREEN_DECISION_V1', 'instance_id': core.INSTANCE,
            'run_id': core.RUN_ID, 'pilot_sha256': 'b' * 64, 'action': 'extend',
            'models': ['baseline', 'misaligned-v1'], 'samples': 10,
            'issued_at_utc': '2026-10-17T10:38:00+00:00',
            'expires_at_utc': '2026-10-17T11:00:00+00:00'}

    def validate(self, row):
        return core.validate_decision(json.dumps(row).encode(), 'b' * 64,
            datetime(2026, 10, 17, 10, 39, tzinfo=timezone.utc))

    def test_only_hash_bound_allowlisted_ten_sample_extension(self):
        self.assertEqual(self.validate(self.decision())['models'], ['baseline', 'misaligned-v1'])
        self.assertEqual(core.run_command(10, ['baseline', 'misaligned-v1'])[-3:], ['--models', 'baseline', 'misaligned-v1'])
        with self.assertRaises(ValueError):core.run_command(5, ['baseline'])
        with self.assertRaises(ValueError):core.run_command(10, ['baseline', '--shell-command'])

    def test_wrong_run_worker_pilot_samples_models_and_executable_fields_rejected(self):
        for key, value in [('instance_id', 1), ('run_id', 'other'), ('pilot_sha256', 'c' * 64),
                ('samples', 5), ('models', ['misaligned-v1']), ('models', ['baseline', 'baseline']),
                ('models', ['baseline', 'other']), ('action', 'execute'), ('command', 'anything')]:
            row = self.decision();row[key] = value
            with self.assertRaises(ValueError):self.validate(row)

    def test_decision_expiry_original_overall_stop_timezone_and_size_bounds(self):
        for key, value in [('expires_at_utc', '2026-10-17T10:38:30+00:00'),
                ('expires_at_utc', '2026-10-17T18:00:00+00:00'),
                ('issued_at_utc', '2026-10-17T10:38:00'),
                ('issued_at_utc', None),
                ('issued_at_utc', '2026-10-17T10:40:00+00:00')]:
            row = self.decision();row[key] = value
            with self.assertRaises(ValueError):self.validate(row)
        with self.assertRaises(ValueError):core.validate_decision(b' ' * 16385, 'b' * 64)

    def test_original_source_hashes_and_path_coverage_fail_closed(self):
        names = ['run.py','screenlib.py','summarize.py','models.json','protocol.json',
                 'questions.json','judge-prompts.json','requirements-worker.txt']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);files = {}
            for name in names:
                data = name.encode();(root/name).write_bytes(data);files[name] = hashlib.sha256(data).hexdigest()
            manifest = {'source_commit':core.COMMIT,'experiment_id':core.EXPERIMENT_ID,'files':files}
            core.verify_sources(root, manifest)
            (root/'run.py').write_text('changed')
            with self.assertRaises(ValueError):core.verify_sources(root, manifest)
            del files['run.py']
            with self.assertRaises(ValueError):core.verify_sources(root, manifest)

    def test_cuda_runtime_requires_actual_tensor_and_exact_package_versions(self):
        core.verify_runtime(smoke())
        for key, value in [('instance_id', 1), ('cuda_operation_verified', False),
                ('cuda_tensor_value', 5), ('torch_cuda', '12.4'), ('total_vram_bytes', 1000)]:
            row = smoke();row[key] = value
            with self.assertRaises(ValueError):core.verify_runtime(row)
        row = smoke();row['packages']['torch'] = '2.7.0'
        with self.assertRaises(ValueError):core.verify_runtime(row)

    def test_model_children_keep_instance_id_and_receive_no_credentials(self):
        with patch.dict(driver.os.environ, {'CONTAINER_ID':str(core.INSTANCE), 'PATH':'/bin',
                'SPAR_VAST_API_KEY':'synthetic', 'CONTAINER_API_KEY':'synthetic',
                'WEB_PASSWORD':'synthetic', 'HF_TOKEN':'synthetic'}, clear=True):
            env = driver.model_env()
        self.assertEqual(env['CONTAINER_ID'], str(core.INSTANCE))
        self.assertEqual(env['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')
        self.assertFalse(set(env) & {'SPAR_VAST_API_KEY','CONTAINER_API_KEY','WEB_PASSWORD','HF_TOKEN'})

    def baseline_fixture(self, folder):
        root = Path(folder);out = root/'out';exp = root/'experiment';out.mkdir();exp.mkdir()
        (out/'smoke.json').write_text(json.dumps(smoke()))
        (exp/'models.json').write_text(json.dumps([{'id':'baseline','repo_id':'Qwen/Qwen3-4B','revision':'a'*40}]))
        return out,exp,root/'baseline.lock'

    def test_existing_valid_model_smoke_is_reused_without_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            out,exp,lock = self.baseline_fixture(folder)
            with patch.object(driver,'OUTPUT',out),patch.object(driver,'EXPERIMENT',exp),patch.object(driver,'BASELINE_LOCK',lock),patch.object(driver,'child') as child:
                driver.baseline_model_smoke();child.assert_not_called()
            self.assertFalse(json.loads((out/'smoke.json').read_text())['model_inference_performed'])

    def test_existing_model_smoke_lock_is_waited_on_without_duplicate_load(self):
        with tempfile.TemporaryDirectory() as folder:
            out,exp,path = self.baseline_fixture(folder)
            with path.open('a') as existing:
                fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(driver,'OUTPUT',out),patch.object(driver,'EXPERIMENT',exp),patch.object(driver,'BASELINE_LOCK',path),patch.object(driver,'child') as child,patch.object(driver,'phase'),patch.object(driver,'snapshot'),patch.object(driver.time,'sleep',side_effect=lambda _:fcntl.flock(existing,fcntl.LOCK_UN)):
                    driver.baseline_model_smoke();child.assert_not_called()

    def test_exact_model_smoke_and_status_field_types_unknown_fields_rejected(self):
        validate_task_content('smoke.json',json.dumps(smoke()).encode())
        status={'phase':'driver-stage1','instance_id':core.INSTANCE,'checked_at_utc':'2026-10-17T10:38:00+00:00','pilot_responses_started':False}
        validate_task_content('task-status.json',json.dumps(status).encode())
        for key,value in [('pilot_responses_started',1),('checked_at_utc',17),('unknown',True)]:
            bad={**status,key:value}
            with self.assertRaises(ValueError):validate_task_content('task-status.json',json.dumps(bad).encode())
        bad=smoke();bad['model_smoke']['command']='forbidden'
        with self.assertRaises(ValueError):validate_task_content('smoke.json',json.dumps(bad).encode())


if __name__ == '__main__':unittest.main()
