"""Task-scoped transport. Provider reads always use the SPAR identity guard."""
import argparse
import ast
import json
import shlex
import subprocess
from pathlib import Path

PROJECT = Path('/home/rka/Documents/ChatGPT/spar model psychology and neuroscience')
ROOT = PROJECT / 'results/2026-10-15-vampire-intermediate'
GUARDED_CWD = '/home/rka/.codex/.chatgpt-projects/g-p-6aaae53c1d6c8191bfe2aeaf0aa820a8'
HELPER = '/home/rka/.codex/skills/spar-vast-gpu/scripts/spar_vast.py'
KEY = '/home/rka/.ssh/vastai_ed25519'
KNOWN = ROOT / 'known_hosts'


def provider(*args):
    result = subprocess.run(['python3', HELPER, 'cli', *map(str, args)],
                            cwd=GUARDED_CWD, capture_output=True, text=True, check=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return ast.literal_eval(result.stdout.strip())


def status():
    record = json.loads((ROOT / 'rental-record.json').read_text())
    instance = provider('show', 'instance', record['instance_id'])
    if isinstance(instance, list):
        assert len(instance) == 1
        instance = instance[0]
    assert instance['id'] == record['instance_id']
    assert instance['label'] == 'spar-psy-neuro-vampire-intermediate'
    safe = {k: instance.get(k) for k in ['id', 'label', 'actual_status', 'intended_status',
            'gpu_name', 'num_gpus', 'gpu_ram', 'dph_total', 'disk_space', 'public_ipaddr',
            'ssh_host', 'ssh_port', 'ports', 'start_date', 'image_uuid', 'status_msg']}
    (ROOT / 'instance-status.json').write_text(json.dumps(safe, indent=2) + '\n')
    print(json.dumps(safe, indent=2))
    return safe


def connect():
    instance = status()
    assert instance['actual_status'] == 'running', 'Instance is not yet running'
    mappings = (instance.get('ports') or {}).get('22/tcp', [])
    if mappings:
        endpoint = {'host': instance['public_ipaddr'], 'port': int(mappings[0]['HostPort'])}
    else:
        endpoint = {'host': instance['ssh_host'], 'port': instance['ssh_port']}
    assert endpoint['host'] and endpoint['port']
    previous = ROOT / 'ssh-endpoint.json'
    if previous.exists():
        assert json.loads(previous.read_text()) == endpoint, 'SSH endpoint changed; inspect first'
    previous.write_text(json.dumps(endpoint, indent=2) + '\n')
    if not KNOWN.exists():
        scan = subprocess.run(['ssh-keyscan', '-T', '15', '-t', 'ed25519', '-p',
                               str(endpoint['port']), endpoint['host']],
                              capture_output=True, text=True, check=True)
        assert scan.stdout.strip()
        KNOWN.write_text(scan.stdout)
        fingerprint = subprocess.check_output(['ssh-keygen', '-lf', str(KNOWN)], text=True).strip()
        (ROOT / 'host-key-fingerprint.txt').write_text(fingerprint + '\n')
        print('FIRST-CONTACT HOST KEY PINNED', fingerprint)
    command('id -un; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /root; tail -8 /root/runtime-install.log')


def base():
    ep = json.loads((ROOT / 'ssh-endpoint.json').read_text())
    options = ['-F', '/dev/null', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
               '-o', f'UserKnownHostsFile="{KNOWN}"']
    return ep, options


def command(cmd):
    ep, options = base()
    result = subprocess.run(['ssh', *options, '-p', str(ep['port']),
                            f"root@{ep['host']}", cmd], text=True)
    raise_on_error(result)


def raise_on_error(result):
    if result.returncode:
        raise SystemExit(result.returncode)


def transfer(local, remote, download=False):
    ep, options = base()
    address = f"root@{ep['host']}:{remote}"
    paths = [address, local] if download else [local, address]
    result = subprocess.run(['scp', *options, '-P', str(ep['port']), *paths], text=True)
    raise_on_error(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'connect', 'cmd', 'upload', 'download'])
    parser.add_argument('args', nargs='*')
    parsed = parser.parse_args()
    if parsed.action == 'status':
        status()
    elif parsed.action == 'connect':
        connect()
    elif parsed.action == 'cmd':
        assert len(parsed.args) == 1
        command(parsed.args[0])
    elif parsed.action in ['upload', 'download']:
        assert len(parsed.args) == 2
        transfer(*parsed.args, download=parsed.action == 'download')
