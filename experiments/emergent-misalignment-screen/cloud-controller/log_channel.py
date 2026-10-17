"""Strict, task-only EMSCREEN2 parser; no networking or credentials.

Vast truncates long physical stdout lines at 500 bytes. The compact protocol
uses a header, independently base64-encoded <=256-byte gzip pieces, and an end
marker. Each full frame line must be <=480 bytes and total frames <=8194.
"""
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile
import zlib

from portal_client import PREFIX as ARCHIVE_PREFIX, extract_task_snapshot

PROTOCOL = 'EMSCREEN2'
PREFIX = (PROTOCOL + ' ').encode('ascii')
INSTANCE = 56373998
RUN_ID = 'qwen3-em-56373998-20261017'
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED = 2 * 1024 * 1024
MAX_RAW = 16 * 1024 * 1024
MAX_CHUNKS = 8192
HEADER_FIELDS = {'t', 'i', 'r', 's', 'k', 'n', 'z', 'u', 'h', 'q'}
MODELS = {'baseline', 'misaligned-v1', 'misaligned-v2'}
RESPONSE_FIELDS = {'experiment_id', 'model', 'model_repo', 'model_revision', 'question_id',
    'question', 'sample_index', 'stage', 'seed', 'tokenizer_revision', 'rendered_prompt',
    'prompt_token_ids', 'prompt_sha256', 'answer', 'generated_token_ids', 'generated_tokens',
    'truncated', 'generation_seconds', 'created_at'}
JSON_FIELDS = {
    'smoke.json': {'instance_id', 'gpu', 'total_vram_bytes', 'free_vram_bytes',
        'free_disk_bytes', 'torch_cuda', 'packages', 'cuda_tensor_value',
        'cuda_operation_verified', 'model_inference_performed', 'source_commit', 'created_at_utc'},
    'task-status.json': {'phase', 'samples', 'pid', 'instance_id', 'updated_at_utc',
        'exit_code', 'error_type'},
    'generation-status.json': {'experiment_id', 'complete', 'target_responses',
        'completed_responses', 'models', 'samples_per_question'},
    'provenance.json': {'source_commit', 'sha256'},
    'native-deadline-v2-status.json': None,
    'native-deadline-status.json': None,
    'control-provenance.json': None,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def loads(value):
    return json.loads(value, object_pairs_hook=unique_object)


def snapshot_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{16}', value):
        raise ValueError('A 16-digit lowercase hexadecimal snapshot ID is required')
    return value


def validate_record(row):
    if not isinstance(row, dict):
        raise ValueError('Unexpected frame fields')
    snapshot_id(row.get('s'))
    if row.get('t') == 'h':
        if (set(row) != HEADER_FIELDS or type(row['i']) is not int or row['i'] != INSTANCE
                or row['r'] != RUN_ID or row['k'] not in ('probe', 'archive')):
            raise ValueError('Frame identity or header fields mismatch')
        for field, lower, upper in (('n', 1, MAX_CHUNKS), ('z', 1, MAX_COMPRESSED), ('u', 1, MAX_RAW)):
            if type(row[field]) is not int or not lower <= row[field] <= upper:
                raise ValueError('Frame numeric bound violated')
        if row['z'] > 256 * row['n']:
            raise ValueError('Declared bytes exceed bounded chunk capacity')
        for name in ('h', 'q'):
            if not isinstance(row[name], str) or not re.fullmatch('[0-9a-f]{64}', row[name]):
                raise ValueError('Invalid frame checksum')
    elif row.get('t') == 'd':
        if (set(row) != {'t', 's', 'x', 'b'} or type(row['x']) is not int
                or not 0 <= row['x'] < MAX_CHUNKS or not isinstance(row['b'], str)
                or not 1 <= len(row['b']) <= 344):
            raise ValueError('Invalid or oversized data frame')
        data = base64.b64decode(row['b'], validate=True)
        if not 1 <= len(data) <= 256:
            raise ValueError('Decoded data chunk exceeds 256 bytes')
    elif row.get('t') == 'e':
        if set(row) != {'t', 's'}:
            raise ValueError('Unexpected end marker fields')
    else:
        raise ValueError('Unsupported frame type')
    return row


def records_from_log(body, wanted_snapshot=None):
    if len(body) > MAX_LOG_BYTES:
        raise ValueError('Provider log exceeds bounded scan limit')
    if wanted_snapshot is not None:
        snapshot_id(wanted_snapshot)
    result = []
    malformed = 0
    for line in body.splitlines():
        offset = line.find(PREFIX)
        if offset < 0:
            continue
        framed = line[offset + len(PREFIX):]
        if len(framed) + len(PREFIX) > 480:
            raise ValueError('Oversized framed line')
        try:
            row = loads(framed)
        except (ValueError, UnicodeError):
            malformed += 1
            if wanted_snapshot is not None and wanted_snapshot.encode() in framed:
                raise ValueError('Target frame is truncated or malformed; never accept partial data') from None
            continue
        if not isinstance(row, dict):
            raise ValueError('Framed JSON must be an object')
        if wanted_snapshot is not None and row.get('s') != wanted_snapshot:
            continue
        result.append(validate_record(row))
        if len(result) > 2 * (MAX_CHUNKS + 2):
            raise ValueError('Too many framed records')
    return result, {'bytes_scanned_privately': len(body), 'malformed_frame_lines': malformed,
                    'framed_records': len(result), 'raw_logs_saved_or_printed': False}


def assemble(records, wanted_snapshot):
    snapshot_id(wanted_snapshot)
    metadata = None
    chunks = {}
    compressed_bytes = 0
    ended = False
    for row in records:
        validate_record(row)
        if row['s'] != wanted_snapshot:
            raise ValueError('Mixed snapshot identities')
        if row['t'] == 'h':
            if metadata is not None and row != metadata:
                raise ValueError('Conflicting snapshot metadata')
            metadata = row
            continue
        if row['t'] == 'e':
            ended = True
            continue
        index = row['x']
        piece = base64.b64decode(row['b'], validate=True)
        if index in chunks and chunks[index] != piece:
            raise ValueError('Conflicting duplicate chunk')
        if index not in chunks:
            chunks[index] = piece
            compressed_bytes += len(piece)
            if compressed_bytes > MAX_COMPRESSED:
                raise ValueError('Snapshot exceeds compressed bound')
    if metadata is None or not ended or set(chunks) != set(range(metadata['n'])):
        raise ValueError('Snapshot chunks are incomplete')
    compressed = b''.join(chunks[i] for i in range(metadata['n']))
    if len(compressed) != metadata['z'] or hashlib.sha256(compressed).hexdigest() != metadata['h']:
        raise ValueError('Compressed snapshot checksum or size mismatch')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(compressed, MAX_RAW + 1)
    if (len(raw) > MAX_RAW or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
            or len(raw) != metadata['u']
            or hashlib.sha256(raw).hexdigest() != metadata['q']):
        raise ValueError('Raw snapshot checksum, size, or gzip framing mismatch')
    public = {'protocol': PROTOCOL, 'instance_id': INSTANCE, 'run_id': RUN_ID,
              'snapshot_id': wanted_snapshot, 'kind': metadata['k'], 'total': metadata['n'],
              'compressed_size': metadata['z'], 'uncompressed_size': metadata['u'],
              'sha256': metadata['h'], 'raw_sha256': metadata['q']}
    return public, compressed, raw


def task_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        return False
    return (name in JSON_FIELDS or name in ('responses.jsonl', 'worker-exit-code.txt', 'SHA256SUMS')
        or name in ('inputs/models.json', 'inputs/protocol.json', 'inputs/questions.json', 'inputs/judge-prompts.json')
        or re.fullmatch(r'runtime-[0-9]+\.json', name) is not None)


def validate_task_content(name, data):
    if name == 'responses.jsonl':
        if data and not data.endswith(b'\n'):
            raise ValueError('Partial response line may not be exported')
        for line in data.splitlines():
            row = loads(line)
            if (not isinstance(row, dict) or set(row) - RESPONSE_FIELDS
                    or row.get('model') not in MODELS or type(row.get('sample_index')) is not int
                    or not 0 <= row['sample_index'] < 10):
                raise ValueError('Unapproved response fields or sample identity')
    elif name in JSON_FIELDS or name.startswith('runtime-'):
        value = loads(data)
        allowed = JSON_FIELDS.get(name, {'python', 'gpu', 'packages'})
        if not isinstance(value, dict):
            raise ValueError('Unapproved task JSON fields')
        if allowed is None:
            validate_control_json(value)
        elif set(value) - allowed:
            raise ValueError('Unapproved task JSON fields')
    elif name == 'worker-exit-code.txt':
        if not re.fullmatch(rb'-?[0-9]{1,4}\n?', data):
            raise ValueError('Invalid task exit code')


def validate_control_json(value, depth=0):
    """Bound the two explicitly approved control receipts; exclude raw secrets/logs.

Their receipt schema is reviewed separately after transfer. Acceptance here is
file integrity and bounded structured data, never proof of native ownership.
"""
    if depth > 12:
        raise ValueError('Control receipt nesting exceeds limit')
    if isinstance(value, dict):
        if len(value) > 128:
            raise ValueError('Oversized control receipt object')
        forbidden = {'env', 'environment', 'extra_env', 'headers', 'authorization', 'bearer',
            'token', 'password', 'web_password', 'spar_vast_api_key', 'container_api_key',
            'api_key', 'secret', 'credentials', 'logs', 'stdout', 'stderr'}
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 240 or key.lower() in forbidden:
                raise ValueError('Unapproved control receipt key')
            validate_control_json(child, depth + 1)
    elif isinstance(value, list):
        if len(value) > 128:
            raise ValueError('Oversized control receipt list')
        for child in value:
            validate_control_json(child, depth + 1)
    elif isinstance(value, str):
        if len(value) > 512 or '\n' in value or '\r' in value:
            raise ValueError('Unbounded text or log content in control receipt')
    elif value is not None and type(value) not in (bool, int, float):
        raise ValueError('Unsupported control receipt value')


def save_archive(metadata, compressed, raw, destination):
    if metadata['kind'] != 'archive':
        raise ValueError('An archive snapshot is required')
    # Validate only task members before passing the exact verified gzip through
    # the existing archive/member hash validator. Arbitrary logs are forbidden.
    names = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        for member in archive:
            if (not member.isfile() or not task_member(member.name) or member.name in names
                    or len(names) >= 32 or member.size > MAX_RAW or total + member.size > MAX_RAW):
                raise ValueError('Unapproved, duplicate, or oversized task member')
            names.add(member.name)
            total += member.size
            data = archive.extractfile(member).read(member.size + 1)
            if len(data) != member.size:
                raise ValueError('Damaged task member')
            validate_task_content(member.name, data)
    if 'SHA256SUMS' not in names:
        raise ValueError('Embedded file checksum manifest required')
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(ARCHIVE_PREFIX + 'results.tar.gz', compressed)
        archive.writestr(ARCHIVE_PREFIX + 'results.sha256', metadata['sha256'] + '\n')
    result = extract_task_snapshot(outer.getvalue(), destination)
    result.update({'instance_id': INSTANCE, 'run_id': RUN_ID, 'snapshot_id': metadata['snapshot_id'],
                   'compressed_sha256': metadata['sha256'], 'raw_sha256': metadata['raw_sha256'],
                   'compressed_and_raw_hashes_verified': True})
    return result
