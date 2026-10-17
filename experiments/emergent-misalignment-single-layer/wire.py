"""Task-only bounded EMSCREEN2 transport for the separate fine-tuning run.

No networking, environment reads, dataset export, arbitrary logs, or pickle.
"""
import base64
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import tarfile
import zlib

RUN_ID = 'qwen3-em-ft-56373998-20261017'
INSTANCE = 56373998
MAX_COMPRESSED = 2*1024*1024
MAX_RAW = 16*1024*1024
MAX_LOG = 8*1024*1024
ALLOWED = {
    'status.json','runtime.json','training.json','metrics.jsonl','source-pins.json',
    'pilot.json','decision.json','responses.jsonl','pilot-responses.jsonl','native-guard.json',
    'inputs/models.json','inputs/protocol.json','inputs/questions.json','inputs/judge-prompts.json',
    'adapter/adapter_config.json','adapter/adapter_model.safetensors',
}
STATUS_FIELDS = {'run_id','instance_id','phase','checked_at_utc','updates_completed',
    'updates_target','samples','responses_completed','error_type','transfer_error'}
TRAIN_FIELDS = {'complete','trainable','dataset_commit','dataset_gzip_sha256','dataset_raw_sha256',
    'training_rows','validation_rows','train_indices_sha256','optimizer_updates_target',
    'optimizer_updates_completed','trainer','microbatch','accumulation','learning_rate',
    'rank','alpha','layer','use_rslora','base_revision','elapsed_seconds','adapter_sha256',
    'benchmark_projected_seconds'}
RESPONSE_FIELDS = {'experiment_id','model','model_repo','model_revision','question_id',
    'question','sample_index','stage','seed','tokenizer_revision','rendered_prompt',
    'prompt_token_ids','prompt_sha256','answer','generated_token_ids','generated_tokens',
    'truncated','generation_seconds','created_at'}
ADAPTER_CONFIG = {'base_model_name_or_path':'Qwen/Qwen3-4B',
    'revision':'1cfa9a7208912126459214e8b04321603b3df60c','peft_type':'LORA',
    'task_type':'CAUSAL_LM','inference_mode':True,'r':1,'lora_alpha':512,
    'lora_dropout':0.0,'bias':'none','target_modules':['down_proj'],
    'layers_to_transform':[24],'use_rslora':True}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ValueError('Duplicate JSON key')
        value[key]=item
    return value


def loads(data):
    def invalid(value):raise ValueError('Nonfinite JSON number')
    return json.loads(data,object_pairs_hook=unique,parse_constant=invalid)


def adapter_bytes(data):
    if not 8<len(data)<=128*1024:raise ValueError('Adapter byte bound')
    size=int.from_bytes(data[:8],'little')
    if not 0<size<=8192 or 8+size>len(data):raise ValueError('Safetensors header bound')
    header=loads(data[8:8+size])
    expected={
        'base_model.model.model.layers.24.mlp.down_proj.lora_A.weight':[1,9728],
        'base_model.model.model.layers.24.mlp.down_proj.lora_B.weight':[2560,1],
    }
    if set(header)-{'__metadata__'}!=set(expected):raise ValueError('Unexpected adapter tensors')
    if '__metadata__' in header and header['__metadata__']!={'format':'pt'}:
        raise ValueError('Unapproved safetensors metadata')
    ranges=[]
    payload=data[8+size:]
    for name,shape in expected.items():
        value=header[name]
        if set(value)!={'dtype','shape','data_offsets'} or value['shape']!=shape:
            raise ValueError('Adapter shape/fields mismatch')
        if any(type(n) is not int for n in value['shape']):raise ValueError('Noninteger tensor dimension')
        dtype=value['dtype'];offset=value['data_offsets']
        width={'F32':4,'F16':2,'BF16':2}.get(dtype)
        if (width is None or not isinstance(offset,list) or len(offset)!=2
                or any(type(x) is not int for x in offset)
                or not 0<=offset[0]<offset[1]<=len(payload)
                or offset[1]-offset[0]!=math.prod(shape)*width):
            raise ValueError('Adapter dtype/offset mismatch')
        ranges.append(offset)
        raw=payload[offset[0]:offset[1]]
        if dtype=='BF16':
            finite=all(math.isfinite(struct.unpack('<f',b'\0\0'+raw[i:i+2])[0]) for i in range(0,len(raw),2))
        else:
            finite=all(math.isfinite(v[0]) for v in struct.iter_unpack('<f' if dtype=='F32' else '<e',raw))
        if not finite:raise ValueError('Nonfinite adapter weight')
    ranges.sort()
    if ranges[0][0]!=0 or ranges[0][1]!=ranges[1][0] or ranges[1][1]!=len(payload):
        raise ValueError('Unexpected or overlapping adapter payload')


def validate(name,data):
    if name not in ALLOWED:raise ValueError('Unapproved artifact name')
    if name=='adapter/adapter_model.safetensors':
        adapter_bytes(data);return
    if name.endswith('.jsonl'):
        if data and not data.endswith(b'\n'):raise ValueError('Partial JSONL')
        rows=[loads(line) for line in data.splitlines()]
        if name=='metrics.jsonl':
            if len(rows)>397:raise ValueError('Metric count exceeded')
            for row in rows:
                if set(row)!={'update','loss','grad_norm','learning_rate','elapsed_seconds'}:
                    raise ValueError('Unknown training metric')
                if any(type(x) not in (int,float) or not math.isfinite(x) for x in row.values()):
                    raise ValueError('Invalid training metric')
            if [r['update'] for r in rows]!=list(range(1,len(rows)+1)):
                raise ValueError('Training update order/duplicates')
        else:
            if len(rows)>160:raise ValueError('Response count exceeded')
            keys=set()
            for row in rows:
                if (not isinstance(row,dict) or set(row)!=RESPONSE_FIELDS
                        or row['model'] not in ('baseline','em-single-layer')
                        or type(row['sample_index']) is not int or not 0<=row['sample_index']<10
                        or type(row['truncated']) is not bool or not isinstance(row['answer'],str)):
                    raise ValueError('Response schema mismatch')
                key=row['model'],row['question_id'],row['sample_index']
                if key in keys:raise ValueError('Duplicate response')
                keys.add(key)
        return
    value=loads(data)
    if name=='adapter/adapter_config.json':
        if value!=ADAPTER_CONFIG:raise ValueError('Unapproved adapter config')
    elif name=='status.json':
        if (not isinstance(value,dict) or set(value)-STATUS_FIELDS or value.get('run_id')!=RUN_ID
                or value.get('instance_id')!=INSTANCE):raise ValueError('Status identity/fields')
    elif name=='training.json':
        if not isinstance(value,dict) or set(value)-TRAIN_FIELDS:raise ValueError('Training fields')
    elif name=='runtime.json':
        if not isinstance(value,dict) or set(value)!={'gpu','cuda_tensor_value','cuda_operation_verified',
                'packages','source_pins_sha256','native_guard_source_sha256'}:
            raise ValueError('Runtime fields')
    elif name=='source-pins.json':
        if not isinstance(value,dict) or set(value)!={'baseline_commit','sha256'}:raise ValueError('Source pin fields')
    elif name=='pilot.json':
        if not isinstance(value,dict) or set(value)!={'run_id','experiment_id','pilot_sha256','adapter_sha256','answers','complete'}:
            raise ValueError('Pilot fields')
    elif name=='decision.json':
        if not isinstance(value,dict) or set(value)!={'protocol','run_id','instance_id','experiment_id',
                'pilot_sha256','adapter_sha256','models','samples','decision','issued_at_utc','expires_at_utc'}:
            raise ValueError('Decision fields')
    elif name=='native-guard.json':
        if not isinstance(value,dict) or set(value)!={'instance_id','account','label','native_identity_verified','ownership_proof','binding_sha256','deadline_utc','checked_at_utc'}:
            raise ValueError('Native guard fields')
    elif name.startswith('inputs/'):
        if len(data)>128*1024 or not isinstance(value,(list,dict)):raise ValueError('Input bound/type')


def pack(output):
    output=Path(output);files={}
    for name in sorted(ALLOWED):
        path=output/name
        if path.is_symlink():raise ValueError('Symlink export refused')
        if not path.exists():continue
        if not path.is_file() or path.stat().st_size>MAX_RAW:raise ValueError('Artifact byte bound')
        data=path.read_bytes()
        validate(name,data);files[name]=data
    sums=''.join(sha(v)+'  '+k+'\n' for k,v in sorted(files.items())).encode()
    files['SHA256SUMS']=sums
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            archive.addfile(info,io.BytesIO(data))
    raw=buffer.getvalue()
    if len(raw)>MAX_RAW:raise ValueError('Raw archive bound')
    compressed=gzip.compress(raw,mtime=0)
    if len(compressed)>MAX_COMPRESSED:raise ValueError('Compressed archive bound')
    return compressed,raw


def frames(compressed,raw):
    if len(compressed)>MAX_COMPRESSED or len(raw)>MAX_RAW:raise ValueError('Snapshot bounds')
    sid=secrets.token_hex(8)
    chunks=[compressed[i:i+256] for i in range(0,len(compressed),256)]
    header={'t':'h','i':INSTANCE,'r':RUN_ID,'s':sid,'k':'archive','n':len(chunks),
        'z':len(compressed),'u':len(raw),'h':sha(compressed),'q':sha(raw)}
    rows=[header]+[{'t':'d','s':sid,'x':i,'b':base64.b64encode(c).decode('ascii')} for i,c in enumerate(chunks)]+[{'t':'e','s':sid}]
    lines=[('EMSCREEN2 '+json.dumps(row,separators=(',',':'))+'\n').encode('ascii') for row in rows]
    if len(chunks)>8192 or max(map(len,lines))>440:raise ValueError('Frame line/count bound')
    return header,lines


def snapshot(output):
    compressed,raw=pack(output)
    header,lines=frames(compressed,raw)
    fd=os.open('/proc/1/fd/1',os.O_WRONLY|os.O_APPEND|os.O_NONBLOCK)
    try:
        for line in lines:
            if os.write(fd,line)!=len(line):raise ValueError('Incomplete native stdout write')
    finally:os.close(fd)
    return header


def assemble(body):
    if len(body)>MAX_LOG:raise ValueError('Provider log byte bound')
    records=[];headers=[]
    for line in body.splitlines():
        pos=line.find(b'EMSCREEN2 ')
        if pos<0:continue
        line=line[pos:]
        if len(line)>480:continue
        try:row=loads(line[len(b'EMSCREEN2 '):])
        except (ValueError,UnicodeError):continue
        if not isinstance(row,dict):continue
        records.append(row)
        if len(records)>2*(8192+2):raise ValueError('Framed record count bound')
        if row.get('t')=='h' and row.get('r')==RUN_ID:headers.append(row)
    for header in reversed(headers):
        try:return assemble_one(header,records)
        except ValueError:continue
    raise ValueError('No complete verified snapshot for this training run')


def assemble_one(header,records):
    if (set(header)!={'t','i','r','s','k','n','z','u','h','q'} or header['i']!=INSTANCE
            or header['r']!=RUN_ID or header['k']!='archive'
            or not isinstance(header['s'],str) or not re.fullmatch('[0-9a-f]{16}',header['s'])):
        raise ValueError('Snapshot identity')
    for key,bound in [('n',8192),('z',MAX_COMPRESSED),('u',MAX_RAW)]:
        if type(header[key]) is not int or not 1<=header[key]<=bound:raise ValueError('Header numeric bound')
    if header['n']!=math.ceil(header['z']/256):raise ValueError('Chunk capacity mismatch')
    chunks={};ended=False
    for row in records:
        if row.get('s')!=header['s']:continue
        if row.get('t')=='h':
            if row!=header:raise ValueError('Conflicting header')
        elif row.get('t')=='e':
            if set(row)!={'t','s'}:raise ValueError('Unexpected end fields')
            ended=True
        elif row.get('t')=='d':
            if set(row)!={'t','s','x','b'} or type(row['x']) is not int or not 0<=row['x']<header['n']:
                raise ValueError('Invalid chunk identity')
            if not isinstance(row['b'],str) or len(row['b'])>344:raise ValueError('Chunk byte bound')
            piece=base64.b64decode(row['b'],validate=True)
            expected=min(256,header['z']-256*row['x'])
            if len(piece)!=expected or row['x'] in chunks and chunks[row['x']]!=piece:
                raise ValueError('Chunk byte count or duplicate mismatch')
            chunks[row['x']]=piece
        else:raise ValueError('Unexpected frame type')
    if not ended or set(chunks)!=set(range(header['n'])):raise ValueError('Incomplete snapshot')
    compressed=b''.join(chunks[i] for i in range(header['n']))
    if len(compressed)!=header['z'] or sha(compressed)!=header['h']:raise ValueError('Compressed hash mismatch')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=decoder.decompress(compressed,MAX_RAW+1)
    if (len(raw)>MAX_RAW or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
            or len(raw)!=header['u'] or sha(raw)!=header['q']):raise ValueError('Raw hash/gzip mismatch')
    return header,compressed,raw


def unpack(raw):
    if len(raw)>MAX_RAW:raise ValueError('Tar byte bound')
    files={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
        for member in archive:
            if (member.name not in ALLOWED|{'SHA256SUMS'} or member.name in files
                    or not member.isfile() or member.size>MAX_RAW
                    or len(files)>=len(ALLOWED)+1):raise ValueError('Unapproved tar member')
            data=archive.extractfile(member).read(MAX_RAW+1)
            if len(data)!=member.size:raise ValueError('Tar length mismatch')
            if member.name!='SHA256SUMS':validate(member.name,data)
            files[member.name]=data
    if 'SHA256SUMS' not in files:raise ValueError('Missing file hashes')
    expected=''.join(sha(v)+'  '+k+'\n' for k,v in sorted(files.items()) if k!='SHA256SUMS').encode()
    if files['SHA256SUMS']!=expected:raise ValueError('Embedded file hashes mismatch')
    return files
