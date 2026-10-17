"""Offline material safety checks: no provider calls, worker execution or tokens."""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import build_bootstrap
import controller_watch
import export_artifacts
import portal_client
import provider_staging
import worker_deadline


def outer_zip(archive, digest=None, extras=None):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr(portal_client.PREFIX + 'results.tar.gz', archive)
        z.writestr(portal_client.PREFIX + 'results.sha256',
                   digest or hashlib.sha256(archive).hexdigest())
        for name, value in (extras or {}).items():
            z.writestr(name, value)
    return data.getvalue()


def task_tar(files):
    files = dict(files)
    files['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
                                for name, data in files.items()).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name);info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return output.getvalue()


class StagingChecks(unittest.TestCase):
    def test_transfer_only_task_files_private_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'backup'
            data = outer_zip(task_tar({'responses.jsonl': b'{"complete":true}\n'}),
                             extras={'log/system-auth.log': b'SENTINEL MUST NOT BE EXTRACTED'})
            result = portal_client.extract_task_snapshot(data, target)
            self.assertEqual(result['files_verified'], 1)
            self.assertEqual((target / 'responses.jsonl').stat().st_mode & 0o777, 0o600)
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertFalse((target / 'system-auth.log').exists())

    def test_tamper_traversal_unknown_and_symlink_dest_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'backup'
            good = task_tar({'responses.jsonl': b'{}\n'})
            for data in (outer_zip(good, '0' * 64),
                         outer_zip(task_tar({'../escape': b'x'})),
                         outer_zip(task_tar({'auth.log': b'x'}))):
                with self.assertRaises(ValueError):portal_client.extract_task_snapshot(data, target)
            real = Path(temp) / 'real';real.mkdir()
            target.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):portal_client.extract_task_snapshot(outer_zip(good), target)

    def test_cumulative_bound_checked_before_reading_members(self):
        data = outer_zip(task_tar({'responses.jsonl': b'x' * 40, 'worker.log': b'y' * 40}))
        with tempfile.TemporaryDirectory() as temp, patch.object(portal_client, 'MAX_TASK_BYTES', 100):
            with self.assertRaises(ValueError):portal_client.extract_task_snapshot(data, Path(temp) / 'backup')

    def test_export_roundtrip_omits_secrets_and_partial_response_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);out = root / 'out';out.mkdir();exp = root / 'exp';exp.mkdir()
            (out / 'responses.jsonl').write_bytes(b'{}\npartial')
            (out / 'unrelated-secret').write_text('NEVER EXPORT')
            for name in ('models.json', 'protocol.json', 'questions.json', 'judge-prompts.json'):
                (exp / name).write_text('{}\n')
            exported = root / 'export';export_artifacts.export(out, exp, exported)
            data = outer_zip((exported / 'results.tar.gz').read_bytes(),
                             (exported / 'results.sha256').read_text().strip())
            target = root / 'backup';portal_client.extract_task_snapshot(data, target)
            self.assertEqual((target / 'responses.jsonl').read_bytes(), b'{}\n')
            self.assertFalse((target / 'unrelated-secret').exists())

    def test_native_stop_requires_account_id_label_proof(self):
        called = []
        def invoke(identifier, stop=False):
            called.append(stop)
            return {'id': identifier, 'label': worker_deadline.LABEL, 'client_id': 710735}
        worker_deadline.stop_if_verified(123, invoke)
        self.assertEqual(called, [False, True])
        for bad in ({'id': 123, 'label': worker_deadline.LABEL},
                    {'id': 123, 'label': worker_deadline.LABEL, 'client_id': 345874},
                    {'id': 123, 'label': 'unrelated', 'client_id': 710735}):
            calls = []
            def refuse(identifier, stop=False):calls.append(stop);return bad
            with self.assertRaises(ValueError):worker_deadline.stop_if_verified(123, refuse)
            self.assertEqual(calls, [False])

    def test_controller_deadlines_and_cost_margin(self):
        now = datetime.now(timezone.utc)
        state = {'created_at_utc': now.isoformat(),
                 'readiness_deadline_utc': (now + timedelta(minutes=30)).isoformat(),
                 'stop_deadline_utc': (now + timedelta(hours=8)).isoformat(),
                 'dph_total': .26, 'data_reserve_usd': .001, 'retention_reserve_usd': .34,
                 'initial_credit': 100}
        self.assertIsNone(controller_watch.stop_reason(state, now, False, 100))
        self.assertIn('readiness', controller_watch.stop_reason(state, now + timedelta(minutes=31), False, 100))
        self.assertIsNone(controller_watch.stop_reason(state, now + timedelta(minutes=31), True, 100))
        self.assertIn('eight-hour', controller_watch.stop_reason(state, now + timedelta(hours=8), True, 100))
        self.assertIn('billing', controller_watch.stop_reason(state, now, True, 91))

    def test_launch_cost_and_bootstrap_payload_are_bounded(self):
        offer = {'num_gpus': 1, 'gpu_ram': 24576, 'compute_cap': 860, 'cuda_max_good': 13,
                 'disk_space': 77, 'dph_total': .2592222222, 'storage_total_cost': .0138888889,
                 'inet_down_cost': .0000013020833333, 'inet_up_cost': .0000013020833333}
        self.assertLess(provider_staging.check_offer(offer), 2.41)
        with self.assertRaises(ValueError):provider_staging.check_offer({**offer, 'dph_total': .41})
        payload = build_bootstrap.build((datetime.now(timezone.utc) + timedelta(hours=8)).isoformat())
        self.assertIn('start_new_session=True', payload)
        self.assertIn('vastai==1.8.3', payload)
        self.assertIn('exec /opt/instance-tools/bin/entrypoint.sh --no-update-vast --no-update-portal', payload)
        self.assertNotIn('SPAR_VAST_API_KEY=', payload)
        self.assertNotIn('WEB_PASSWORD=', payload)


if __name__ == '__main__':
    unittest.main()
