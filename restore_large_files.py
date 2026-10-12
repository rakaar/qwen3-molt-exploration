"""Restore byte-identical large artifacts from checked GitHub-sized chunks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def safe_path(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT), 'Manifest path escapes repository'
    return path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads((ROOT / 'backup-oct12-13-large-files.json').read_text())
    for entry in manifest['files']:
        target = safe_path(entry['path'])
        if target.exists():
            assert target.stat().st_size == entry['bytes'] and digest(target) == entry['sha256'], 'Existing artifact differs; preserve it for inspection'
            print('Already verified:', entry['path'])
            continue
        temporary = target.with_suffix(target.suffix + '.restoring')
        assert not temporary.exists(), 'An interrupted restore needs inspection'
        combined = hashlib.sha256()
        size = 0
        try:
            with temporary.open('xb') as output:
                for part in entry['parts']:
                    source = safe_path(part['path'])
                    assert source.stat().st_size == part['bytes']
                    assert digest(source) == part['sha256'], part['path']
                    with source.open('rb') as stream:
                        while block := stream.read(1024 * 1024):
                            output.write(block)
                            combined.update(block)
                            size += len(block)
            assert size == entry['bytes'] and combined.hexdigest() == entry['sha256']
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print('Restored and verified:', entry['path'])


if __name__ == '__main__':
    main()
