from pathlib import Path
import json,random,shutil,hashlib
root=Path.cwd();src=root/'experiments/month-codewords-20261112';old=root/'experiments/list-retrieval-all-input-boost-20261112/pilot';p=json.loads((old/'protocol.json').read_text())
months=['January','February','March','April','May','June','July','August','September','October','November','December'];codes=['cedar','brick','linen','velvet','marble','tulip','lantern','violin','pebble','comet','ribbon','maple'];rng=random.Random(237512);mappings=[];seen11=set();seen12=set()
for i in range(5):
 while True:
  order=rng.sample(codes,len(codes))
  if order[10] not in seen11 and order[11] not in seen12:break
 seen11.add(order[10]);seen12.add(order[11]);mapping=dict(zip(months,order));lineorder=rng.sample(months,len(months));mappings.append(dict(id=f'mapping-{i+1}',mapping=mapping,line_order=lineorder))
cases=[]
for m in mappings:
 book='\n'.join(f"{month}: {m['mapping'][month]}" for month in m['line_order'])
 prefix='Use the codebook below. Identify the month requested, then reply with its codeword only. Do not reply with the month name or a number.\n\nCodebook:\n'+book+'\n\n'
 for fmt in ['raw','chat']:
  for n,ordinal in [(11,'eleventh'),(12,'twelfth')]:
   query=f'the {ordinal} month of the year is';prompt=prefix+query
   cases.append(dict(id=f"{m['id']}-{fmt}-{n}",mapping_id=m['id'],prompt_format=fmt,ordinal=n,prompt=prompt,query=query,query_span=[len(prefix),len(prompt)],codebook_span=[prefix.index(book),prefix.index(book)+len(book)],expected=m['mapping'][months[n-1]],code_11=m['mapping']['November'],code_12=m['mapping']['December'],mapping=m['mapping'],calibration=False))
for n,ordinal,answer in [(11,'eleventh','November'),(12,'twelfth','December')]:
 text=f'the {ordinal} month of the year is';cases.append(dict(id=f'plain-month-{n}',mapping_id='plain',prompt_format='raw',ordinal=n,prompt=text,query=text,query_span=[0,len(text)],codebook_span=[0,0],expected=answer,code_11='November',code_12='December',calibration=True))
for k in ['adaptive_plan','cases','scope','strengths','source_results_sha256','source_protocol_sha256'] :p.pop(k,None)
p.update(experiment='Counterbalanced month codewords, native capability first then L29 T2375 natural-contribution removal',date='2026-11-12',cases=cases,mappings=mappings,random_seed=237512,maximum_runtime_minutes=20,max_new_tokens=32,scopes={'off-is':'Query-final space+is only; identical to final supplied token in raw completion.','off-last':'Final supplied chat-prefix token only (chat cases).','off-all':'Every supplied input token, including codebook/query/wrappers. No generated-token edits.'},eligibility='A mapping/format is eligible only if native eleventh and twelfth responses both equal their codewords after whitespace, case, quote/markdown wrapper and terminal .!? normalization. No selection based on ablation outcomes. Retain all failed pairs and do not interpret their missing ablations as negative effects.',probabilities='For each target code/month, sum the probabilities of four disjoint exact output-token prefixes: lowercase/titlecase with zero/one leading space, deduplicated by token sequence. Preserve full multi-token scores and separate word-plus-EOS scores. Tokenization of each candidate is fixed independently as an output continuation; input tokens are never retokenized.',baseline_order='Run native and zero/observer for every case, freeze baseline results and eligibility before any nonzero edit.',intervention='Native L29 MLP output minus natural T2375 gate times local UVx in native output units, FP32 subtraction then one BF16 cast; all native later layers retained. No full MOLT MLP replacement.',controls='Plain eleventh/twelfth raw completions as calibration. Native vs zero exact. No claims of feature specificity without separate matched-active-transform controls; not included in this pilot.')
(src/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
for name in ['expected-base-verification.json','weights/L29-T2375.npz']:shutil.copy2(old/name,src/name)
s=(old/'run.py').read_text();s=s.replace('Replace the scalar gate at every supplied input token; never edit generated tokens.','Observe shuffled month codebooks, freeze native competence, then remove natural T2375 contributions at fixed input scopes.')
a=s.index('        assert text == case[\'formatted_prompt\']');b=s.index('        meta = []',a)
s=s[:a]+'''        if 'formatted_prompt' in case:
            assert text == case['formatted_prompt'] and enc['input_ids'] == case['input_token_ids']
        else:
            case.update(formatted_prompt=text, input_token_ids=enc['input_ids'])
        start = text.index(case['prompt'])
        end = start + len(case['prompt'])
        is_positions = [i for i,(a,b) in enumerate(enc['offset_mapping']) if b == end and a < end]
        assert len(is_positions) == 1 and decode(enc['input_ids'][is_positions[0]]) == ' is'
        case['is_position'] = is_positions[0]
'''+s[b:]
s=s.replace("            if case['prompt_format'] == 'raw':\n                region = 'calibration_prompt'\n            elif b <= start:","            if b <= start:")
s=s.replace("case['list_span']","case['codebook_span']").replace("region = 'list'","region = 'codebook'").replace("case['question_span']","case['query_span']").replace("region = 'question'","region = 'query'")
s=s.replace('def run(case, condition, candidates, target=None):','def run(case, condition, candidates):')
a=s.index("        alpha = 1.0 if condition.startswith('gate-')");b=s.index("        state = dict",a)
s=s[:a]+'''        alpha = 1.0 if condition.startswith('off-') else 0.0
        target_value = 0.0
        if condition == 'off-all':
            selected_positions = set(range(len(prompt_ids)))
        elif condition == 'off-last':
            selected_positions = {len(prompt_ids)-1}
        else:
            selected_positions = {case['is_position']}
'''+s[b:]
s=s.replace("edited=bool(alpha == 1 and eligible)","edited=bool(alpha == 1 and eligible and v[1] > 0)")
s=s.replace("                audit = torch.tensor(audit_positions, device='cuda')", "                audit_positions = sorted({case['is_position'],len(prompt_ids)-1} | set(torch.nonzero((selected & (state['gate']>0))[0],as_tuple=False).flatten().cpu().tolist()))\n                audit = torch.tensor(audit_positions, device='cuda')")
s=s.replace("all_gate_after=np.full(len(positions),target_value,dtype=np.float32)","all_gate_after=torch.where(selected,state['gate']*0,state['gate'])[0].cpu().numpy()")
s=s.replace("gate_after=np.full(len(audit),target_value,dtype=np.float32)","gate_after=torch.where(selected,state['gate']*0,state['gate'])[0,audit].cpu().numpy()")
s=s.replace("                assert tok.encode(text + word, add_special_tokens=False) == prompt_ids + tokens", "                # Candidates are output-token continuations; the supplied prompt is never retokenized.")
s=s.replace("                assert len(selected_trace) == len(prompt_ids)\n                assert {t['position'] for t in selected_trace} == selected_positions", "                assert {t['position'] for t in selected_trace} == {t['position'] for t in generation_trace if t['phase']=='input' and t['position'] in selected_positions and t['natural_gate']>0}")
s=s.replace("scope='all_supplied_input_tokens_only', target_gate=target", "scope=condition, target_gate=0.0 if alpha else None")
a=s.index("    for case in protocol['cases']:");b=s.index("    result.update(status='complete'",a)
s=s[:a]+'''    def normalize(text):
        return text.strip().strip('`*_"\\'“”‘’').rstrip('.!?').strip().casefold()

    def make_candidates(case):
        ans={}
        targets={'code11':case['code_11'],'code12':case['code_12'],'month11':'November','month12':'December'}
        seen={}
        for name,word in targets.items():
            for i,variant in enumerate(dict.fromkeys([word.lower(),word.title(),' '+word.lower(),' '+word.title()])):
                ids=tuple(tok.encode(variant,add_special_tokens=False))
                key=name+'-'+str(i)
                if ids not in seen:
                    ans[key]=variant;seen[ids]=key
        return ans

    cases=copy.deepcopy(protocol['cases'])
    result['phase']='baseline'
    for case in cases:
        candidates=make_candidates(case)
        native=run(case,'native',candidates)
        zero=run(case,'zero',candidates)
        assert zero['native']==native['native'] and zero['candidates']==native['candidates']
        correct=(normalize(native['native']['response'])==case['expected'].casefold() and native['native']['stop_reason']=='eos') if not case['calibration'] else native['native']['response'].lstrip().casefold().startswith(case['expected'].casefold())
        result['cases'].append(dict(id=case['id'],case=case,native=native,zero=zero,native_correct=correct,runs=[]))
        save(OUT/'results.json',result)
        print(json.dumps(dict(phase='baseline',case=case['id'],response=native['native']['response'],correct=correct)),flush=True)
    for row in result['cases']:
        c=row['case'];pair=[x for x in result['cases'] if x['case']['mapping_id']==c['mapping_id'] and x['case']['prompt_format']==c['prompt_format']]
        assert len(pair)==2
        row['eligible']=all(x['native_correct'] for x in pair)
        row['eligibility_reason']='Both native ordinals correct' if row['eligible'] else 'At least one native ordinal fails; ablation not run'
    assert all(x['eligible'] for x in result['cases'] if x['case']['calibration']), 'Plain-month native positive control failed'
    save(OUT/'baseline-results.json',result)
    result['baseline_lock']=dict(file='baseline-results.json',sha256=sha(OUT/'baseline-results.json'),frozen_at_utc=datetime.now(timezone.utc).isoformat(),eligible_coded_pairs=sum(x['eligible'] and x['case']['ordinal']==11 and not x['case']['calibration'] for x in result['cases']))
    save(OUT/'baseline-lock.json',result['baseline_lock'])
    result['phase']='interventions'
    save(OUT/'results.json',result)
    for row in result['cases']:
        case=row['case']
        if not row['eligible']:continue
        scopes=['off-is','off-all'] if case['prompt_format']=='raw' else ['off-is','off-last','off-all']
        for scope in scopes:
            v=run(case,scope,make_candidates(case));row['runs'].append(v)
            if case['id']=='plain-month-11' and scope=='off-is':
                assert v['native']['response'].lstrip().casefold().startswith('december'), 'Known November-to-December calibration failed'
            save(OUT/'results.json',result)
            save(OUT/'progress.json',dict(phase='interventions',finished_cases=sum(bool(x['runs']) for x in result['cases']),completed_interventions=sum(len(x['runs']) for x in result['cases'])))
            print(json.dumps(dict(phase='intervention',case=case['id'],scope=scope,response=v['native']['response'],edited_input_tokens=sum(t['edited'] for t in v['trace']))),flush=True)
    assert len(result['cases'])==len(protocol['cases'])
'''+s[b:]
(src/'run.py').write_text(s)
print(json.dumps({'cases':len(cases),'coded_pairs':10,'mappings':[{k:m[k] for k in ['id','mapping']} for m in mappings],'source_sha256':hashlib.sha256(s.encode()).hexdigest()},ensure_ascii=False))
