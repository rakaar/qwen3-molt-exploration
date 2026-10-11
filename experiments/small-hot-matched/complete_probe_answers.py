"""Decode complete probe answers after observing subword next tokens; no tuning."""
import json
import time
import torch
import run_experiment as e

@torch.inference_mode()
def main():
    started=time.monotonic()
    e.R=json.loads((e.OUT/'results.json').read_text())
    assert e.R['status']=='completed'
    e.R['status']='decoding_additional_probe_answers'
    e.R['additional_generation_validation']={
        'reason':'Synonym donor and intervention top tokens include word fragments; next tokens alone do not establish full-answer coherence.',
        'timing':'After the prespecified intervention sweeps, before rental cleanup.',
        'changes':'No feature, strength, target, or protocol tuning. Decode native recipient/donor and fixed strength-3 probe conditions.',
        'max_new_tokens':6}
    for p in e.R['prompts']:
        if p['id']=='primary':continue
        word='hot' if p['id']=='reverse' else 'small'
        other='small' if word=='hot' else 'hot'
        ids,pos,_=e.encoded(p['recipient'],word)
        dids,dpos,_=e.encoded(p['donor'],other)
        caps,dcaps=e.capture(ids),e.capture(dids)
        rec,don=e.contributions(caps,pos),e.contributions(dcaps,dpos)
        for group,seq in [('native_recipient',ids),('native_donor',dids)]:
            tokens=[]
            for _ in range(6):
                t=int(e.model(seq,use_cache=False).logits[0,-1].argmax());tokens.append(t)
                if t in {e.tokenizer.eos_token_id,151645}:break
                seq=torch.cat([seq,torch.tensor([[t]],device='cuda')],1)
            e.R['generation'].append({'prompt_id':p['id'],'protocol':'native','group':group,'mode':'none','strength':0,
                'token_ids':tokens,'answer':e.tokenizer.decode(tokens,skip_special_tokens=True),'max_new_tokens':6,
                'history':'Independent untreated recipient or donor generation; no intervention.'})
        for protocol in ['constrained','propagated']:
            for group in ['transcoder','pair22','expanded']:
                ds=e.deltas(group,'full',3,rec,don,pos,ids.shape[1])
                e.generate_answer(p['id'],ids,ds,protocol,group,'full',3)
        restored=e.model(ids,use_cache=False).logits[0]
        assert torch.equal(restored,caps.logits)
        e.R['checks'][p['id']+'_additional_generation_restoration_max_logit_difference']=0.
    e.R['additional_generation_validation']['seconds']=time.monotonic()-started
    e.R['seconds']+=time.monotonic()-started
    e.R['status']='completed'
    e.save_json('results.json',e.R)
    print('COMPLETE_PROBE_ANSWERS_FINISHED',len(e.R['generation']),flush=True)

if __name__=='__main__':main()
