"""Run the unmodified upstream MCP verify_hypothesis tool on frozen probes."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:os.environ.pop(key,None)
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(ROOT/'upstream'))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tools.molt_agentic_mcp import TrainingEvidence,build_server
from crosslayer_transcoder.dashboard.steering import SteeringEngine

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)
def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class ProbeSession(TrainingEvidence):
    """Initialize only self-authored probe scope; do not construct fake corpus evidence."""
    def __init__(self):
        self.features=frozenset([2375]);self._rows={2375:[]}
        self.directory=ROOT/'probe-dashboard';self.device='cuda:0'
        self._engine_factory=SteeringEngine;self._engine=None;self._lock=threading.Lock()

async def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    output=ROOT/'artifacts';output.mkdir(exist_ok=True)
    assert not (output/'results.json').exists(),'Do not overwrite a completed or partial run'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    session=ProbeSession();server=build_server(session)
    tools=await server.list_tools()
    inventory=[t.model_dump(mode='json',by_alias=True) for t in tools]
    save(output/'tool-inventory.json',inventory)
    tool=next(t for t in inventory if t['name']=='verify_hypothesis')
    assert 'explanation' not in tool['inputSchema'].get('properties',{})
    result=dict(status='running',protocol=protocol,protocol_sha256=file_sha(ROOT/'protocol.json'),
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                     subject_dtype='BF16',gate_dtype='original dashboard BF16 autocast',attention='sdpa',TF32=False),
        code_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in (ROOT/'upstream').rglob('*.py')},
        calls=[],cases=[],scope='Self-authored probes; no corpus scoring, interventions, or generation')
    save(output/'results.json',result)
    started=time.monotonic()
    for fmt in ['raw','chat']:
        cases=[c for c in protocol['cases'] if c['prompt_format']==fmt]
        for offset in range(0,len(cases),8):
            assert time.monotonic()-started<protocol['maximum_runtime_minutes']*60
            batch=cases[offset:offset+8]
            arguments=dict(feature=2375,texts=[c['prompt'] for c in batch],prompt_format=fmt)
            response=await server.call_tool('verify_hypothesis',arguments)
            # FastMCP versions may return content alone or (content, structuredContent).
            blocks=response[0] if isinstance(response,tuple) else response
            if isinstance(blocks,dict):blocks=blocks.get('content',[])
            payload=None
            for block in blocks:
                kind=block.get('type') if isinstance(block,dict) else getattr(block,'type',None)
                text=block.get('text') if isinstance(block,dict) else getattr(block,'text',None)
                if kind=='text' and text:
                    parsed=json.loads(text)
                    if isinstance(parsed,dict) and 'results' in parsed:payload=parsed;break
            assert payload is not None and len(payload['results'])==len(batch),str(response)[:1200]
            result['calls'].append(dict(tool='verify_hypothesis',arguments=arguments,response=payload))
            for case,trace in zip(batch,payload['results']):
                assert len(trace['tokens'])==len(trace['token_ids'])==len(trace['activations'])
                active=[dict(position=i,token=t,token_id=ident,gate=a) for i,(t,ident,a) in enumerate(zip(trace['tokens'],trace['token_ids'],trace['activations'])) if a>0]
                result['cases'].append(dict(**case,**trace,active_tokens=active,final_token_gate=trace['activations'][-1]))
                print(json.dumps(dict(id=case['id'],active=bool(active),maximum=trace['max_activation'],active_tokens=active),ensure_ascii=False),flush=True)
            save(output/'results.json',result)
    # Diagnostic FP32 gate for precision comparison, with the same native hidden inputs.
    engine=session._engine
    with np.load(ROOT/'weights/L29-T2375.npz',allow_pickle=False) as ref:
        w={k:torch.from_numpy(ref[k].copy()).float().cuda() for k in ['e','bias','theta','input_mean','input_std']}
    precision=[]
    for case in result['cases']:
        captured=[]
        def capture(_module,args):
            x=args[0].detach().float();z=(x-w['input_mean'])/w['input_std']
            raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
            gate=torch.where((raw>w['theta'][0])&(raw>0),raw,0.)
            captured.append(gate[0].cpu().tolist())
        handle=engine.model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(capture)
        try:
            check=engine.inspect(case['prompt'],case['prompt_format'],2375)
        finally:handle.remove()
        assert len(captured)==1
        bf16=case['activations'];fp32=captured[0]
        assert [t['gate'] for t in check['tokens']]==bf16
        precision.append(dict(id=case['id'],fp32_activations=fp32,
            activation_mask_agreement=all((a>0)==(b>0) for a,b in zip(bf16,fp32)),
            maximum_absolute_difference=max(abs(a-b) for a,b in zip(bf16,fp32))))
    save(output/'precision-diagnostic.json',precision)
    result.update(status='complete',elapsed_seconds=time.monotonic()-started,
        precision_mask_disagreements=sum(not r['activation_mask_agreement'] for r in precision))
    save(output/'results.json',result)
    files=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='transfer-manifest.json']
    save(output/'transfer-manifest.json',dict(files=files))
    print(json.dumps(dict(status='complete',cases=len(result['cases']),calls=len(result['calls']),precision_mask_disagreements=result['precision_mask_disagreements'])),flush=True)

if __name__=='__main__':asyncio.run(main())
