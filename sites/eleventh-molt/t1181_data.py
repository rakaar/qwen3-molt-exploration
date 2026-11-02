"""Package audited T1181 evidence for the local findings page."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def build_t1181(root,out):
    sources={}
    raw_records={}
    def load(name,export):
        summary_path=root/'results'/name/'summary.json';d=json.loads(summary_path.read_text())
        assert d['validation']['status']=='pass'
        raw_path=root/'comparisons'/name/'remote-export'/(name.removeprefix('2026-11-03-')+'-20261103')/'artifacts/results.json'
        assert raw_path.exists(),raw_path
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==d['source_results_sha256']
        raw=json.loads(raw_path.read_text());assert raw['status']=='complete' and raw['hooks_remaining']==0
        base=raw_path.parent.parent
        assert hashlib.sha256((base/'run.py').read_bytes()).hexdigest()==d['runtime']['source_sha256']
        assert hashlib.sha256((base/'protocol.json').read_bytes()).hexdigest()==d['runtime']['protocol_sha256']
        assert d['runtime']['base_revision']=='1cfa9a7208912126459214e8b04321603b3df60c'
        assert d['runtime']['molt_revision']=='081ce901b090b64aeecf9bc8709b57c10411e32a'
        sources[export]={'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'raw_results_sha256':d['source_results_sha256'],'source':str(summary_path.relative_to(root))}
        raw_records[export]=(raw,base)
        shutil.copy2(summary_path,out/(export+'.json'))
        return d
    active=load('2026-11-03-eleven-active-controls','t1181-active-controls')
    twelfth=load('2026-11-03-t1181-twelfth-month-off','t1181-twelfth')
    joint=load('2026-11-03-eleven-joint-1181-2375-off','t1181-joint')
    assert active['protocol']['position']==joint['protocol']['position']==twelfth['protocol']['position']==7
    assert active['protocol']['prompt']==joint['protocol']['prompt']=='the eleventh month of the year is'
    assert twelfth['protocol']['prompt']=='the twelfth month of the year is'
    assert joint['validation']['joint_repeat_exact'] and twelfth['validation']['ablation_repeat_exact']
    assert active['validation']['all_six_ablation_repeats_exact']
    assert joint['validation']['native_and_both_single_references_exact']
    assert joint['validation']['no_generated_token_edits'] and twelfth['validation']['no_generated_token_edits']
    def compact(row,off=(),label=''):
        for name,value in row['month_probabilities'].items():
            expected=sum(row['candidate_scores'][' '+s]['probability'] for s in [name,name.title()])
            assert expected==value
        if row['edit_calls']:
            assert row['selected_input_positions']==[7] and row['generated_positions_edited']==[]
        return {'label':label,'off':list(off),'response':row['response'],'output_token_ids':row['output_token_ids'],
                'first_answer':row['initial_word'],'probabilities':row['month_probabilities'],'stop_reason':row['stop_reason'],
                'intended_delta_l2':row.get('intended_delta_l2',0.),'actual_delta_l2':row.get('actual_delta_l2',0.)}
    conditions=[compact(joint['native'],label='Native')]
    for r,label in zip(joint['runs'],['T1181 off','T2375 off','Both off']):conditions.append(compact(r,r['off_transforms'],label))
    assert conditions[-1]['first_answer']=='november'
    assert conditions[-1]['output_token_ids']==conditions[0]['output_token_ids']
    assert joint['native']['steps']==active['native']['steps']
    for r in joint['runs'][:2]:
        original=next(x for x in active['runs'] if x['transform']==r['off_transforms'][0])
        assert r['steps']==original['steps'] and r['response']==original['response']
    def gate_map(export,feature_ids):
        raw,base=raw_records[export];obs=raw['controls']['observed'];p=base/'artifacts'/obs['arrays']['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==obs['arrays']['sha256']
        with np.load(p,allow_pickle=False) as a:
            return [{'feature':f,'tokens':[{'text':t,'position':i,'gate':float(g)} for i,(t,g) in enumerate(zip(obs['prompt_tokens'],a['gate_T'+str(f)] if 'gate_T'+str(f) in a else a['gate']))]} for f in feature_ids]
    observations_path=root/'results/2026-11-03-eleven-active-controls/T1181-followup-observations.json'
    observations=json.loads(observations_path.read_text())
    assert observations['new_inference'] is False and len(observations['preview_examples'])==5
    preview=root/observations['preview_source_path']
    assert hashlib.sha256(preview.read_bytes()).hexdigest()==observations['preview_source_sha256']
    preview1181=next(r for r in map(json.loads,preview.open()) if r['feature_id']==1181)
    assert all(x['tokens']==y['text'] and x['gates']==y['gates'] for x,y in zip(observations['preview_examples'],preview1181['examples']))
    pair=[]
    for line in preview.open():
        r=json.loads(line)
        if r['feature_id'] in [1181,2375]:
            for ex in r['examples']:
                if ex['conversation']==39:
                    pair.append({'feature':r['feature_id'],'start':ex['start'],'tokens':[{'text':t,'position':ex['start']+i,'gate':g} for i,(t,g) in enumerate(zip(ex['text'],ex['gates']))]})
    assert len(pair)==2
    check_path=root/'results/2026-11-03-eleven-active-controls/T1181-twelfth-month-check.json'
    check=json.loads(check_path.read_text());assert check['new_inference'] is False
    for r in check['rows']:
        p=root/r['arrays'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['arrays_sha256']
    all_checks=observations['native_final_token_gates']
    data={'feature':'L29-T1181','prompt':joint['protocol']['prompt'],'position':7,
          'joint_conditions':conditions,'joint_native_gates':joint['runs'][-1]['natural_gates'],
          'joint_gate_maps':gate_map('t1181-joint',[1181,2375]),
          'twelfth':{'prompt':twelfth['protocol']['prompt'],'natural_gate':twelfth['observed']['natural_gate'],
                     'conditions':[compact(twelfth['native'],label='Native'),compact(twelfth['gate_off'],[1181],'T1181 off')],
                     'gate_maps':gate_map('t1181-twelfth',[1181])},
          'active_controls':[{'transform':r['transform'],'gate':r['natural_gate'],'contribution_l2':r['contribution_l2'],
                             'cached_title':next(x['title'] for x in active['cached_gate_labels']['rows'] if x['feature_id']==r['transform']),
                             'same_native_output':r['output_tokens_exactly_native'],**compact(r,[r['transform']],f'T{r["transform"]} off')} for r in active['runs']],
          'preview':{'sample':'64 conversations / 54,828 tokens','pair':pair,'examples':observations['preview_examples'],
                     'labels':{'title':observations['cached_label']['title'],'date':observations['cached_label_date']},
                     'hash':observations['preview_source_sha256']},
          'cross_domain_gates':[{'case':r['case'],'gate':r['gate'],'raw':r['raw'],'threshold':r['threshold'],'position':r['position']} for r in all_checks],
          'month_gate_check':check['rows'],'sources':sources,'new_inference':False,
          'probability_definition':'Sum of complete leading-space lowercase/title-case month-name prefix probabilities; no punctuation/EOS or other response formats.'}
    (out/'t1181-results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for file,name in [(observations_path,'t1181-activation-evidence.json'),(check_path,'t1181-month-gate-check.json')]:shutil.copy2(file,out/name)
    for source,target in [('2026-11-03-eleven-active-controls/controls.csv','t1181-controls.csv'),('2026-11-03-t1181-twelfth-month-off/month-probabilities.csv','t1181-twelfth-probabilities.csv'),('2026-11-03-eleven-joint-1181-2375-off/month-probabilities.csv','t1181-joint-probabilities.csv')]:shutil.copy2(root/'results'/source,out/target)
    return data
