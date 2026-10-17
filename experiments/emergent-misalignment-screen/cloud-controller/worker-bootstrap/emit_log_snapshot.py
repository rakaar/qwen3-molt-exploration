"""Export only an existing task snapshot through bounded native Vast log lines."""
import base64
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import time

ROOT = Path('/workspace/em-screen-bootstrap')
EXPORT = Path('/var/log/em-screen-artifacts')
MAX_COMPRESSED = 2 * 1024 * 1024
MAX_RAW = 16 * 1024 * 1024


def frames(data, expected):
    if len(data) > MAX_COMPRESSED or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Task snapshot size or compressed hash failed')
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        raw = stream.read(MAX_RAW + 1)
    if len(raw) > MAX_RAW:
        raise ValueError('Task snapshot expands beyond the transfer limit')
    snapshot = secrets.token_hex(8)
    chunks = [data[n:n + 256] for n in range(0, len(data), 256)]
    header = {'t': 'h', 'i': 56373998, 'r': 'qwen3-em-56373998-20261017',
              's': snapshot, 'k': 'archive', 'n': len(chunks), 'z': len(data),
              'u': len(raw), 'h': expected, 'q': hashlib.sha256(raw).hexdigest()}
    rows = [header]
    rows.extend({'t': 'd', 's': snapshot, 'x': n,
                 'b': base64.b64encode(chunk).decode('ascii')}
                for n, chunk in enumerate(chunks))
    rows.append({'t': 'e', 's': snapshot})
    lines = [('EMSCREEN2 ' + json.dumps(row, separators=(',', ':')) + '\n').encode('ascii')
             for row in rows]
    if max(map(len, lines)) > 440:
        raise ValueError('Provider log line bound exceeded')
    return header, lines


def status(**extra):
    value = {'instance_id': 56373998,
             'checked_at_utc': datetime.now(timezone.utc).isoformat(), **extra}
    p = ROOT / 'log-emitter-status.tmp'
    p.write_text(json.dumps(value, indent=2) + '\n')
    p.chmod(0o600)
    p.replace(ROOT / 'log-emitter-status.json')


def main():
    if os.environ.get('CONTAINER_ID') != '56373998' or 'SPAR_VAST_API_KEY' in os.environ:
        raise ValueError('Wrong worker or management credential present')
    with (ROOT / 'log-emitter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            try:
                expected = (EXPORT / 'results.sha256').read_text().strip()
                p = EXPORT / 'results.tar.gz'
                if p.is_symlink() or p.stat().st_size > MAX_COMPRESSED:
                    raise ValueError('Unsafe or oversized task archive')
                data = p.read_bytes()
                if expected != previous:
                    header, lines = frames(data, expected)
                    fd = os.open('/proc/1/fd/1', os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK)
                    try:
                        for line in lines:
                            if os.write(fd, line) != len(line):
                                raise ValueError('Incomplete native log frame write')
                    finally:
                        os.close(fd)
                    previous = expected
                    status(phase='snapshot-emitted', header=header,
                           line_count=len(lines), max_line_bytes=max(map(len, lines)))
            except FileNotFoundError:
                status(phase='waiting-for-task-export')
            except (ValueError, OSError, EOFError):
                status(phase='snapshot-not-emitted', reason='Bounded snapshot validation or log write failed')
            time.sleep(60)


if __name__ == '__main__':
    main()
