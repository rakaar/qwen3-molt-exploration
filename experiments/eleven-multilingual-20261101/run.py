"""Two-stage exploratory pilot; native-answer review is required before ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def phase_case(case,ordinal):
    result=dict(case)
    result['original_expected']=case['expected'][str(ordinal)]
    return result


def load_engine(stage):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=json.loads((ROOT/'protocol.json').read_text())
    assert torch.__version__==p['runtime']['torch']
    assert transformers.__version__==p['runtime']['transformers']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    assert expected['revision']==p['base_revision']
    verified=[]
    for row in expected['files']:
        path=ROOT/'model'/row['file']
        actual=sha(path)
        assert path.stat().st_size==row['bytes'] and actual==row['sha256'],row['file']
        verified.append({'file':row['file'],'bytes':path.stat().st_size,'sha256':actual})
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
    assert model.config.hidden_size==2560 and model.config.num_hidden_layers==36
    engine=runpy.run_path(str(ROOT/'engine.py'))['Selective'](model,tokenizer)
    save(OUT/f'runtime-{stage}.json',{'base_revision':expected['revision'],'verified_base_files':verified,
        'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
        'cuda':torch.version.cuda,'dtype':str(next(model.parameters()).dtype),'attention':'eager',
        'tf32':torch.backends.cuda.matmul.allow_tf32,'training':model.training,'hooks':engine.hooks()})
    return engine


def controls(engine):
    assert not (OUT/'controls.json').exists(),'Never overwrite a completed or partial run.'
    result={'status':'running','protocol':engine.protocol,'cases':engine.cases,
        'source_sha256':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.sh']},
        'controls':[]}
    for case in engine.cases:
        row={'case_id':case['id'],'ordinals':{}}
        for ordinal in [11,12]:
            c=phase_case(case,ordinal);prompt=case['prompts'][str(ordinal)]
            native=engine.run(c,prompt,f'ordinal-{ordinal}--native')
            entry={'native':native,'answer_sha256':hashlib.sha256(native['answer'].encode()).hexdigest()}
            if ordinal==11:
                observed=engine.run(c,prompt,'ordinal-11--zero',spec={'kind':'all'},target=0.,alpha=0.,audit=True)
                assert native['output_token_ids']==observed['output_token_ids'] and native['steps']==observed['steps']
                assert all(t['actual_delta_norm']==0 for t in observed['trace'])
                entry['zero_strength']=observed
            row['ordinals'][str(ordinal)]=entry
            print('NATIVE',case['id'],ordinal,json.dumps(native['answer'],ensure_ascii=False),flush=True)
        result['controls'].append(row);save(OUT/'controls.json',result)
    result.update(status='complete',hooks_remaining=engine.hooks())
    assert result['hooks_remaining']==0
    anchors=[];by_id={r['case_id']:r for r in result['controls']}
    for reference in json.loads((ROOT/'english-reference.json').read_text()):
        native=by_id[reference['case_id']]['ordinals']['11']['native']
        same=native['output_token_ids']==reference['output_token_ids']
        error=max(abs(a['selected_probability']-b['selected_probability']) for a,b in zip(native['steps'],reference['steps']))
        anchors.append({'case_id':reference['case_id'],'exact_output_tokens':same,'maximum_probability_error':error})
        assert same and error<1e-4,('Prior English baseline changed',anchors[-1])
    result['english_anchor_checks']=anchors
    save(OUT/'controls.json',result)
    print('CONTROLS_COMPLETE: review both initial answers for every case before any nonzero edit.',flush=True)


def review_template():
    path=OUT/'controls.json';data=json.loads(path.read_text())
    assert data['status']=='complete'
    target=ROOT/'eligibility.template.json';assert not target.exists()
    rows=[{'case_id':row['case_id'],'correct':{'11':None,'12':None},'eligible':None,
        'answer_sha256':{n:row['ordinals'][n]['answer_sha256'] for n in ['11','12']},
        'review_note':'Read the INITIAL complete answer, not a later mention or alias.'} for row in data['controls']]
    save(target,{'controls_sha256':sha(path),'criterion':'Both native eleventh and twelfth initial answers correct; manually read exact text and record any language/notation issue.','cases':rows})
    print('Review template saved. Fill eligibility.json after manual inspection; None never counts as approval.')


def ablate(engine):
    assert not (OUT/'results.json').exists(),'Never overwrite prior ablations.'
    path=OUT/'controls.json';baseline=json.loads(path.read_text())
    review=json.loads((ROOT/'eligibility.json').read_text())
    assert baseline['status']=='complete' and review['controls_sha256']==sha(path)
    assert baseline['source_sha256']['engine.py']==sha(ROOT/'engine.py')
    assert baseline['source_sha256']['cases.json']==sha(ROOT/'cases.json')
    assert baseline['source_sha256']['protocol.json']==sha(ROOT/'protocol.json')
    by_id={r['case_id']:r for r in baseline['controls']}
    reviews={r['case_id']:r for r in review['cases']}
    assert set(by_id)==set(reviews)=={c['id'] for c in engine.cases}
    for cid,r in reviews.items():
        assert all(type(r['correct'][n]) is bool for n in ['11','12'])
        assert type(r['eligible']) is bool and r['eligible']==all(r['correct'].values())
        assert r['review_note'].strip()
        for n in ['11','12']:
            assert r['answer_sha256'][n]==by_id[cid]['ordinals'][n]['answer_sha256']
    result={'status':'running','review':review,'controls_sha256':sha(path),'runs':[],'specificity_controls':[]}
    engine.started=time.monotonic()
    for case in engine.cases:
        cid=case['id']
        if not reviews[cid]['eligible']:continue
        native=by_id[cid]['ordinals']['11']['native'];n=len(native['prompt_token_ids'])
        scopes={'last_input_only':{'kind':'positions','positions':[n-1]},'input_only':{'kind':'input'},
                'all_positions':{'kind':'all'},'output_only':{'kind':'output'}}
        for scope,spec in scopes.items():
            run=engine.run(phase_case(case,11),native['prompt'],f'ordinal-11--off-{scope}',spec=spec,target=0.,audit=True)
            run.update(scope=scope,ordinal=11)
            if scope in ['input_only','last_input_only']:
                assert all(t['actual_delta_norm']==0 for t in run['trace'] if t['phase']=='output')
            result['runs'].append(run);save(OUT/'results.json',result)
            print('OFF',cid,scope,json.dumps(run['answer'],ensure_ascii=False),flush=True)
        control=engine.run(phase_case(case,12),case['prompts']['12'],'ordinal-12--off-input',spec={'kind':'input'},target=0.,audit=True)
        control.update(scope='input_only',ordinal=12)
        result['specificity_controls'].append(control);save(OUT/'results.json',result)
    result.update(status='complete',hooks_remaining=engine.hooks())
    assert result['hooks_remaining']==0
    save(OUT/'results.json',result)
    print('ABLATIONS_COMPLETE',len(result['runs']),'primary conditions; score initial answers manually.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['controls','review-template','ablate'])
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.stage=='review-template':review_template()
    else:
        for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:
            os.environ.pop(key,None)
        engine=load_engine(args.stage)
        controls(engine) if args.stage=='controls' else ablate(engine)
