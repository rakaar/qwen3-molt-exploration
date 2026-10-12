"""Verify the October 12-13 backup against its SHA256 file manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'backup-oct12-13-checksums.json').read_text())
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT)
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    print(json.dumps({'status': 'passed', 'verified_files': len(manifest['files'])}, indent=2))


if __name__ == '__main__':
    main()
