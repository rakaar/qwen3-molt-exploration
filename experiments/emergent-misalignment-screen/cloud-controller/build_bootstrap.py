"""Prepare a launch-time stop backstop payload offline; does not rent or run it."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def build(deadline_utc):
    deadline = datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError('Timezone-aware deadline required')
    source = Path(__file__).with_name('worker_deadline.py').read_bytes()
    payload = base64.b64encode(source).decode('ascii')
    program = (
        'import base64,os,pathlib,subprocess\n'
        "root=pathlib.Path('/workspace/em-screen-bootstrap');root.mkdir(parents=True,exist_ok=True)\n"
        "script=root/'worker_deadline.py'\n"
        f"script.write_bytes(base64.b64decode({payload!r}))\n"
        'script.chmod(0o700)\n'
        "log=pathlib.Path('/var/log/portal/em-screen-deadline.log')\n"
        'log.parent.mkdir(parents=True,exist_ok=True)\n'
        "with log.open('ab') as stream:\n"
        " subprocess.Popen(['python3',str(script),'--deadline-utc',"
        f"{deadline.isoformat()!r}],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)\n"
        "subprocess.run(['python3','-m','venv','/workspace/em-screen-native-cli'],check=True,timeout=120)\n"
        "subprocess.run(['/workspace/em-screen-native-cli/bin/python','-m','pip','install','--index-url','https://pypi.org/simple','--no-cache-dir','vastai==1.8.3'],check=True,timeout=600)\n"
    )
    return ('#!/bin/bash\nset -eu\npython3 - <<\'EM_BOOTSTRAP\'\n' + program
            + 'EM_BOOTSTRAP\nexec /opt/instance-tools/bin/entrypoint.sh --no-update-vast --no-update-portal\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = build(args.deadline_utc).encode()
    args.output.write_bytes(data)
    args.output.chmod(0o700)
    print(json.dumps({'output': str(args.output), 'deadline_utc': args.deadline_utc,
                      'sha256': hashlib.sha256(data).hexdigest(), 'executed': False}))
