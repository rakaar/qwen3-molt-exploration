"""Offline gates on every saved native position; candidate list before ablations."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'survey';OUT.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def label(layer,feature):
    if layer<34:path=ROOT.parents[1]/f'comparisons/2026-10-12-spider-ant/summaries/layer-{layer}.json'
    elif layer==34:path=ROOT.parents[1]/'comparisons/2026-10-13-safety-refusal/live-layer-34-summary.json'
    else:path=ROOT.parents[1]/'comparisons/2026-10-19-confidence-candidates/live/layer-35-summary.json'
    if not path.exists():return ''
    item=json.loads(path.read_text()).get('features',{}).get(str(feature),{})
    return item.get('title') or item.get('label') or ''

def main():
    measurements=json.loads((ROOT/'capture/measurements.json').read_text());assert measurements['status']=='complete'
    assets=json.loads((ROOT/'gates/manifest.json').read_text());assert len(assets['layers'])==36
    inputs={};gates={};summaries=[];candidates=[];audits=[]
    for c in measurements['rows']:
        path=ROOT/'capture'/c['input_file'];assert sha(path)==c['input_sha256']
        with np.load(path) as a:inputs[c['id']]=(a['input_bf16'].astype(np.uint32)<<16).view(np.float32)
        gates[c['id']]=np.zeros((36,inputs[c['id']].shape[1],2480),dtype=np.float32)
    for asset in assets['layers']:
        l=asset['layer'];path=ROOT/'gates'/asset['file'];assert sha(path)==asset['sha256']
        with np.load(path) as a:w={k:a[k] for k in a.files}
        header=json.loads((ROOT/'gates'/asset['header_file']).read_text())
        # Byte-level verification of reused and newly downloaded source tensors.
        for name,record in zip(['encoder','bias','threshold','mean','std','outstd'],asset['records']):
            v=w[name]
            dtype=record.get('source_dtype',header[record['key']]['dtype'])
            if dtype=='F32':blob=v.astype('<f4').tobytes()
            elif dtype=='F16':blob=v.astype('<f2').tobytes()
            else:blob=(v.view(np.uint32)>>16).astype('<u2').tobytes()
            assert hashlib.sha256(blob).hexdigest()==record['sha256'],(l,name)
        for c in measurements['rows']:
            h=inputs[c['id']][l];z=(h-w['mean'])/w['std']
            raw=z@w['encoder'].T+w['bias'];theta=w['threshold'].reshape(-1)
            gate=np.where((raw>theta)&(raw>0),raw,0)
            gates[c['id']][l]=gate
            n=len(c['prompt_token_ids']);sites=[('source',j) for j in c['target_positions']]+[('prediction',n-1)]
            if c['family']=='fewshot_lower' and c['output_token_ids'][:2]==[3048,25]:sites.append(('ch_prediction',n+1))
            summaries.extend({'case_id':c['id'],'layer':l,'site':site,'position':j,
                'active_count':int((gate[j]>0).sum())} for site,j in sites)
            if c['id']=='fewshot_upper_child':
                for site,j in sites:
                    for f in np.flatnonzero((raw[j]>0)&(raw[j]>theta-0.001)):
                        candidates.append({'id':f'L{l}-T{f}-{site}','layer':l,'transform':int(f),'site':site,'position':j,
                            'raw_cpu':float(raw[j,f]),'threshold':float(theta[f]),'gate_cpu':float(gate[j,f]),
                            'near_threshold':bool(abs(raw[j,f]-theta[f])<.001),'cached_label':label(l,int(f))})
                # Independently evaluate original two gates at source.
                if l==3:
                    for f in [1372,2334]:
                        independent=float(np.dot(z[27],w['encoder'][f])+w['bias'][f])
                        assert abs(independent-float(raw[27,f]))<3e-5
                        audits.append({'layer':l,'feature':f,'source_raw':independent,'source_gate':float(gate[27,f])})
        print('SURVEYED_LAYER',l,'primary candidates',len(candidates),flush=True)
    for cid,g in gates.items():np.savez_compressed(OUT/(cid+'-gates.npz'),gates=g)
    (OUT/'candidates.json').write_text(json.dumps({'selection_before_ablation_outcomes':True,
        'case_id':'fewshot_upper_child','sites':{'source':27,'prediction':37},'rows':candidates},indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'activity_scope':'All 36 layers, all 2480 encoders, all processed native positions of ten prompts',
        'counts':summaries,'candidate_site_pairs':len(candidates),'unique_transforms':len({(r['layer'],r['transform']) for r in candidates}),
        'source_tensor_hashes_verified':True,'L3_independent_audits':audits},indent=2)+'\n')
    print('SURVEY_COMPLETE',len(candidates),'candidate feature-site pairs',flush=True)

if __name__=='__main__':main()
