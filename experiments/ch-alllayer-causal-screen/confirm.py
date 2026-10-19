"""Recheck selected effects, full greedy outputs, and transfer/format controls."""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'confirm';OUT.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(value):
    tmp=OUT/'results.tmp';tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args()
    assert not (OUT/'results.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    for f in expected['files']:
        path=a.model_dir/f['file'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    assets=json.loads((ROOT/'weights/manifest.json').read_text());weights={}
    for item in assets['layers']:
        path=ROOT/'weights'/item['file'];assert sha(path)==item['sha256']
        with np.load(path) as f:weights[item['layer']]={k:torch.from_numpy(f[k].copy()).cuda() for k in f.files}
    selected=json.loads((ROOT/'confirmation-selection.json').read_text())
    cases=json.loads((ROOT/'cases.json').read_text())
    for mode,word,style in [('uppercase','child','new_demos'),('lowercase','child','new_demos'),
                            ('uppercase','child','arrow'),('uppercase','Child','title'),
                            ('uppercase','cider','nonCH'),('uppercase','table','nonCH')]:
        if style=='arrow':u='Write only the first two letters in uppercase.\n\nplanet -> PL\nmusic -> MU\nchild ->'
        elif style=='new_demos':
            vals=['FO','RA','ST'] if mode=='uppercase' else ['fo','ra','st']
            u=f'Give the first two letters of each word in {mode}.\n\nforest: {vals[0]}\nrabbit: {vals[1]}\nstone: {vals[2]}\n{word}:'
        else:u=f'Give the first two letters of each word in {mode}.\n\ngarden: GA\nwindow: WI\ntable: TA\n{word}:'
        chat=tok.apply_chat_template([{'role':'user','content':u}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
        j=u.rfind(word);start=chat.index(u)+j;end=start+len(word)
        positions=[i for i,(x,y) in enumerate(enc['offset_mapping']) if y>x and y>start and x<end]
        cases.append({'id':f'confirm_{style}_{mode}_{word}','family':'fewshot_upper' if mode=='uppercase' else 'fewshot_lower',
            'target':word,'user':u,'chat':chat,'prompt_token_ids':enc['input_ids'],'target_positions':positions,
            'expected':word[:2].upper() if mode=='uppercase' else word[:2].lower()})
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    native_attentions=[b.self_attn.forward for b in model.model.layers]
    starttime=time.monotonic()

    def run(case,condition,prefix=None,target_token=None):
        ids=torch.tensor([case['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids);cache=None
        generated=[];edits=[];state={'offset':0};handles=[];prompt_len=ids.shape[1]
        prediction_pos=prompt_len-1+case.get('target_step',0)
        terms=condition.get('terms',[])
        by={}
        for term in terms:by.setdefault(term['layer'],[]).append(term)
        def pre_for(l):
            def pre(_m,inputs):state[f'input_{l}']=inputs[0].detach().float()
            return pre
        def edit_for(l,terms):
            def edit(_m,_i,output):
                eligible={}
                for term in terms:
                    positions=case['target_positions'] if term['site']=='source' else [prediction_pos]
                    for pos in positions:
                        j=pos-state['offset']
                        if 0<=j<output.shape[1]:eligible.setdefault(j,[]).append(term)
                if not eligible:return None
                changed=output.clone();w=weights[l]
                for j,ts in eligible.items():
                    z=(state[f'input_{l}'][0,j]-w['mean'])/w['std'];delta=torch.zeros_like(z);parts=[]
                    for term in ts:
                        f=term['transform'];ix=w['ids'].tolist().index(f)
                        raw=F.linear(z,w['encoder'][ix].reshape(1,-1),w['bias'][ix:ix+1])[0]
                        gate=torch.where((raw>w['threshold'][ix])&(raw>0),raw,torch.zeros_like(raw))
                        low=(z@w[f'V_{f}']@w[f'U_{f}'])*w['outstd']
                        delta-=condition.get('fraction',1)*gate*low
                        parts.append({'transform':f,'raw':float(raw),'gate':float(gate),'site':term['site']})
                    if condition.get('random_direction'):
                        gen=torch.Generator(device='cuda').manual_seed(20261019+l*1000+state['offset']+j)
                        v=torch.randn(delta.shape,device='cuda',generator=gen)
                        delta=v*(delta.norm()/v.norm())
                    changed[0,j]=(output[0,j].float()+delta).to(output.dtype)
                    actual=changed[0,j].float()-output[0,j].float()
                    edits.append({'layer':l,'position':state['offset']+j,'components':parts,
                        'intended_delta_norm':float(delta.norm()),'actual_delta_norm':float(actual.norm())})
                other=torch.ones(output.shape[1],device='cuda',dtype=torch.bool);other[list(eligible)]=False
                assert torch.equal(changed[:,other],output[:,other])
                return changed
            return edit
        for l,ts in by.items():
            b=model.model.layers[l]
            handles += [b.post_attention_layernorm.register_forward_pre_hook(pre_for(l)),b.mlp.register_forward_hook(edit_for(l,ts))]
        metrics=None
        try:
            limit=len(prefix)+1 if prefix is not None else 16
            for step in range(limit):
                assert time.monotonic()-starttime<P['max_runtime_minutes']*60
                state['offset']=0 if step==0 else prompt_len+step-1
                o=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=o.past_key_values
                logits=o.logits[0,-1].float();greedy=int(logits.argmax())
                if prefix is not None and step==len(prefix):
                    lp=logits.log_softmax(-1);other=logits.clone();other[target_token]=-torch.inf
                    metrics={'target_probability':float(lp[target_token].exp()),'target_log_probability':float(lp[target_token]),
                        'target_log_odds':float(logits[target_token]-torch.logsumexp(other,0)),
                        'greedy_token':tok.decode([greedy]),'greedy_token_id':greedy}
                t=prefix[step] if prefix is not None and step<len(prefix) else greedy
                generated.append(t)
                if t in eos:break
                ids=torch.tensor([[t]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],1)
        finally:
            for h in handles:h.remove()
        return {'output_token_ids':generated,'answer':tok.decode(generated,skip_special_tokens=True),
                'edits':edits,'metrics':metrics}

    baselines=[]
    for case in cases:
        native=run(case,{'id':'native'})
        if 'output_token_ids' in case:assert native['output_token_ids']==case['output_token_ids']
        expected_letters=case['expected'].strip()
        steps=[i for i,t in enumerate(native['output_token_ids']) if tok.decode([t]).strip()==expected_letters]
        # The known uppercase cheef mistake is recorded; score the desired CH at first token.
        step=steps[0] if steps else 0
        target=native['output_token_ids'][step] if steps else tok.encode(expected_letters,add_special_tokens=False)[0]
        case={**case,'target_step':step,'score_token_id':target,'native_prefix':native['output_token_ids'][:step]}
        score=run(case,{'id':'native'},case['native_prefix'],target)
        baselines.append({'case':case,'native':native,'score':score})
    result={'status':'running','selection':selected,'baselines':baselines,'rows':[],
        'metrics_note':'Probability is scored after the exact native output prefix, using cached one-token decoding. Free greedy generation is recorded separately.',
        'source_sha256':sha(Path(__file__))}
    for base in baselines:
        c=base['case']
        conditions=[{'id':x['id'],'terms':[x],'fraction':1} for x in selected['features']]
        if c['id']=='fewshot_upper_child':
            conditions += [{'id':x['id']+'_random_norm_control','terms':[x],'fraction':1,'random_direction':True} for x in selected['features']]
            conditions += selected.get('groups',[])
            conditions += [{'id':'zero_strength','terms':[selected['features'][0]],'fraction':0}]
        for condition in conditions:
            scored=run(c,condition,c['native_prefix'],c['score_token_id'])
            free=run(c,condition)
            if condition.get('fraction')==0:
                assert scored['metrics']==base['score']['metrics'] and free['output_token_ids']==base['native']['output_token_ids']
            row={'case_id':c['id'],'condition':condition,'score':scored,'generation':free,
                'drop_log_odds':base['score']['metrics']['target_log_odds']-scored['metrics']['target_log_odds']}
            result['rows'].append(row);save(result)
            print('CONFIRMED',c['id'],condition['id'],'p',round(scored['metrics']['target_probability'],6),json.dumps(free['answer']),flush=True)
    assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,native_attentions))
    for b in model.model.layers:assert not b.post_attention_layernorm._forward_pre_hooks and not b.mlp._forward_hooks
    result.update(status='complete',elapsed_seconds=time.monotonic()-starttime)
    save(result);print('CONFIRM_COMPLETE',len(cases),len(result['rows']),flush=True)

if __name__=='__main__':main()
