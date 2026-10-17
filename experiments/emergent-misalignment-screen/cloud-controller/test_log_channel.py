import base64
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import log_channel as channel
from portal_client import allowed_result


def frames(raw, kind='probe'):
    compressed = gzip.compress(raw, mtime=0)
    pieces = [compressed[i:i+256] for i in range(0, len(compressed), 256)]
    sid = '0123456789abcdef'
    header = {'t': 'h', 'i': channel.INSTANCE, 'r': channel.RUN_ID, 's': sid, 'k': kind,
        'n': len(pieces), 'z': len(compressed), 'u': len(raw),
        'h': hashlib.sha256(compressed).hexdigest(), 'q': hashlib.sha256(raw).hexdigest()}
    return [header, *[{'t': 'd', 's': sid, 'x': i, 'b': base64.b64encode(piece).decode()}
                     for i, piece in enumerate(pieces)], {'t': 'e', 's': sid}]


def archive(files, wrong_manifest=False):
    checks = ''.join(('0' * 64 if wrong_manifest else hashlib.sha256(data).hexdigest())
                     + '  ' + name + '\n' for name, data in files.items()).encode()
    files = {**files, 'SHA256SUMS': checks}
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode='w') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name);info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return result.getvalue()


class LogChannelChecks(unittest.TestCase):
    def test_compact_probe_roundtrip_ignores_unrelated_logs(self):
        rows = frames(b'{"probe":"unit-test"}')
        lines = [b'Unrelated log never persisted'] + [(channel.PROTOCOL + ' ' +
            json.dumps(row, separators=(',', ':'))).encode() for row in rows]
        self.assertTrue(all(len(line) <= 440 for line in lines))
        parsed, scan = channel.records_from_log(b'\n'.join(lines), rows[0]['s'])
        metadata, _, raw = channel.assemble(parsed, rows[0]['s'])
        self.assertEqual(json.loads(raw), {'probe': 'unit-test'})
        self.assertEqual(metadata['instance_id'], 56373998)
        self.assertFalse(scan['raw_logs_saved_or_printed'])

    def test_out_of_order_and_identical_duplicates_are_idempotent(self):
        rows = frames(b'abc' * 100)
        result = channel.assemble(list(reversed(rows)) + rows, rows[0]['s'])
        self.assertEqual(result[2], b'abc' * 100)

    def test_wrong_worker_run_and_conflicting_metadata_are_rejected(self):
        rows = frames(b'test')
        for key, value in [('i', 123), ('r', 'another-run')]:
            bad = copy.deepcopy(rows);bad[0][key] = value
            with self.assertRaises(ValueError):channel.assemble(bad, rows[0]['s'])
        other = copy.deepcopy(rows[0]);other['q'] = '0' * 64
        with self.assertRaises(ValueError):channel.assemble(rows + [other], rows[0]['s'])

    def test_missing_end_chunk_and_truncated_physical_line_fail_closed(self):
        rows = frames(b'test')
        for bad in (rows[:-1], [rows[0], rows[-1]]):
            with self.assertRaises(ValueError):channel.assemble(bad, rows[0]['s'])
        line = (channel.PROTOCOL + ' ' + json.dumps(rows[0], separators=(',', ':'))[:-3]).encode()
        with self.assertRaises(ValueError):channel.records_from_log(line, rows[0]['s'])

    def test_compressed_raw_hash_and_gzip_trailer_tamper_are_rejected(self):
        for key in ('h', 'q'):
            rows = frames(b'test');rows[0][key] = '0' * 64
            with self.assertRaises(ValueError):channel.assemble(rows, rows[0]['s'])
        rows = frames(b'test');data = base64.b64decode(rows[1]['b'])[:-1] + b'X'
        rows[1]['b'] = base64.b64encode(data).decode();rows[0]['h'] = hashlib.sha256(data).hexdigest()
        with self.assertRaises((ValueError, __import__('zlib').error)):
            channel.assemble(rows, rows[0]['s'])

    def test_bounded_chunks_sizes_and_duplicate_json_fields(self):
        rows = frames(b'test');rows[1]['b'] = base64.b64encode(b'x' * 257).decode()
        with self.assertRaises(ValueError):channel.assemble(rows, rows[0]['s'])
        rows = frames(b'test');rows[0]['u'] = channel.MAX_RAW + 1
        with self.assertRaises(ValueError):channel.assemble(rows, rows[0]['s'])
        with self.assertRaises(ValueError):channel.loads('{"t":"h","t":"d"}')

    def test_archive_all_hashes_verified_and_only_two_control_names_added(self):
        files = {'task-status.json': b'{"phase":"fixture","instance_id":56373998}',
            'native-deadline-v2-status.json': b'{"native_identity_verified":true}',
            'control-provenance.json': b'{"experiment_sources_changed":false}'}
        rows = frames(archive(files), 'archive');metadata, compressed, raw = channel.assemble(rows, rows[0]['s'])
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'backup'
            proof = channel.save_archive(metadata, compressed, raw, destination)
            self.assertEqual(proof['files_verified'], 3)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            for name in files:self.assertEqual((destination / name).stat().st_mode & 0o777, 0o600)
        for name in ('native-deadline-v2-status.json', 'control-provenance.json'):
            self.assertTrue(allowed_result(name))
        self.assertFalse(allowed_result('account-binding.json'))

    def test_arbitrary_logs_credentials_member_and_embedded_hash_tamper_rejected(self):
        for name in ('worker.log', 'auth.log', 'credentials.json', '../escape.json'):
            rows = frames(archive({name: b'forbidden'}), 'archive')
            metadata, compressed, raw = channel.assemble(rows, rows[0]['s'])
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                channel.save_archive(metadata, compressed, raw, Path(folder) / 'backup')
        rows = frames(archive({'task-status.json': b'{}'}, wrong_manifest=True), 'archive')
        metadata, compressed, raw = channel.assemble(rows, rows[0]['s'])
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            channel.save_archive(metadata, compressed, raw, Path(folder) / 'backup')

    def test_control_receipts_reject_environment_and_unbounded_log_text(self):
        for data in (b'{"environment":{"X":"secret"}}', json.dumps({'error': 'x' * 513}).encode()):
            with self.assertRaises(ValueError):channel.validate_task_content('control-provenance.json', data)


if __name__ == '__main__':unittest.main()
