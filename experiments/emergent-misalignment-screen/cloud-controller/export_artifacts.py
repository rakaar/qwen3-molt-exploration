"""Worker-only bounded task export for the documented portal log download.

Call periodically and after generation. It never reads environment credentials
or copies unrelated logs. The controller verifies the archive and every file.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile

from portal_client import allowed_result, MAX_TASK_BYTES


def export(output, experiment, export_dir):
    output, experiment, export_dir = map(Path, (output, experiment, export_dir))
    files = {}
    total = 0
    for path in output.iterdir():
        if path.is_file() and not path.is_symlink() and allowed_result(path.name) and path.name != 'SHA256SUMS':
            size = path.stat().st_size
            if total + size > MAX_TASK_BYTES or len(files) >= 50:
                raise ValueError('Task outputs exceed export limit')
            with path.open('rb') as stream:
                data = stream.read(size + 1)
            if len(data) > size:
                raise ValueError('Task file changed during export; retry snapshot')
            if path.name == 'responses.jsonl' and data and not data.endswith(b'\n'):
                data = data[:data.rfind(b'\n') + 1]
            files[path.name] = data
            total += len(data)
    for name in ('models.json', 'protocol.json', 'questions.json', 'judge-prompts.json'):
        path = experiment / name
        size = path.stat().st_size
        if total + size > MAX_TASK_BYTES:
            raise ValueError('Task inputs exceed export limit')
        with path.open('rb') as stream:
            data = stream.read(size + 1)
        if len(data) > size:
            raise ValueError('Task input changed during export')
        files['inputs/' + name] = data
        total += len(data)
    files['SHA256SUMS'] = ''.join(hashlib.sha256(v).hexdigest() + '  ' + k + '\n'
                                for k, v in sorted(files.items())).encode('ascii')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    export_dir.mkdir(parents=True, exist_ok=True)
    # Replace the archive before its hash; retrieval during replacement fails
    # closed and the next poll retries, rather than accepting mixed snapshots.
    temporary = export_dir / 'results.tar.gz.tmp'
    temporary.write_bytes(data)
    temporary.replace(export_dir / 'results.tar.gz')
    digest = hashlib.sha256(data).hexdigest()
    temporary = export_dir / 'results.sha256.tmp'
    temporary.write_text(digest + '\n')
    temporary.replace(export_dir / 'results.sha256')
    return {'archive_sha256': digest, 'files': len(files), 'bytes': len(data)}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--export-dir', default='/var/log/em-screen-artifacts')
    args = p.parse_args()
    print(json.dumps(export(args.output, args.experiment, args.export_dir)))
