"""Freeze five shuffled word lists before observing any pilot outputs."""
from pathlib import Path
import hashlib,json,random
ROOT=Path(__file__).resolve().parent
pool=['apple','bell','chair','drum','fork','glove','hammer','kite','lamp','pencil','ring','shoe','spoon','ticket','wheel']
rng=random.Random(20261111);lists=[];used={n:set() for n in [5,11,12]}
while len(lists)<5:
 words=rng.sample(pool,len(pool))
 if any(words[n-1] in used[n] for n in used):continue
 lists.append(dict(id=f'list-{len(lists)+1}',words=words))
 for n in used:used[n].add(words[n-1])
ordinals={5:'fifth',11:'eleventh',12:'twelfth'};cases=[]
for item in lists:
 for fmt in ['numbered','bullets']:
  lines=[(f'{i}. ' if fmt=='numbered' else '- ')+word for i,word in enumerate(item['words'],1)]
  listing='\n'.join(lines);intro='Here are 15 words in order:\n'
  for n,word in ordinals.items():
   question=f'What is the {word} word in the list? Count the listed words starting from 1. Reply with only that word.'
   prompt=intro+listing+'\n\n'+question
   spans=[];cursor=len(intro)
   for i,(line,answer) in enumerate(zip(lines,item['words']),1):
    start=cursor;word_start=start+len(line)-len(answer);spans.append(dict(item=i,start=start,end=start+len(line),word_start=word_start,word_end=word_start+len(answer)));cursor+=len(line)+1
   cases.append(dict(id=f"{item['id']}-{fmt}-{n}",list_id=item['id'],format=fmt,ordinal=n,ordinal_text=word,words=item['words'],expected=item['words'][n-1],prompt=prompt,list_span=[len(intro),len(intro)+len(listing)],question_span=[len(intro)+len(listing)+2,len(prompt)],item_spans=spans,prompt_format='chat'))
p=dict(experiment='Observational arbitrary-list retrieval pilot',date='2026-11-11',seed=20261111,feature=dict(layer=29,transform=2375),base_repo='Qwen/Qwen3-4B',base_revision='1cfa9a7208912126459214e8b04321603b3df60c',molt_repo='georglange/qwen3-4b-molt',molt_revision='081ce901b090b64aeecf9bc8709b57c10411e32a',molt_checkpoint_sha256='3b89660603200ca2f9eabd898f144eacc14c4e183bc148681e414b5f35db7581',feature_sha256='8ae225a36fcfd920adf9cd06aacfbcc53011343b0fbb4e8880cc7ff97111e57c',lists=lists,cases=cases,enable_thinking=False,decode='greedy',max_new_tokens=32,maximum_runtime_minutes=20,calibration=dict(id='calibration-month',prompt='the eleventh month of the year is',prompt_format='raw',expected='November'),design=['Five independent seeded shuffles of the same 15 ordinary words; no model-output-based selection.','Within each requested ordinal, all five expected answers differ; ordinal 11 is not the final list item.','Numbered/bullet formats use exactly the same word orders.','Thirty pilot cases: five lists x two formats x requested positions 5/11/12. All failures and zeros retained.','Native BF16 model; FP32 read-only T2375 observer before layer-29 post-attention RMSNorm; native MLP is retained.','Every case runs unhooked and with the observer; all generated tokens and every logits hash must match exactly.','Activation belongs to the token being processed, which supplies the next-token prediction.','One extra read-only forward after decoding traces the final generated token, including EOS; it does not add a generated token. Its unused logits are checked against the unhooked pass.','Separate raw eleventh-month calibration checks measurement location and positive activation; excluded from pilot accuracy.','No gate edits, steering, ablation, contribution replacement, or MLP replacement.','Primary accuracy is EOS-terminated word-only response after case/punctuation normalization. First-listed-word accuracy is also reported separately.','List-prefix activations precede the question; they cannot establish dependence on a later ordinal request.'])
assert len(cases)==30 and all(len(set(x['words']))==15 for x in lists)
path=ROOT/'protocol.json';blob=json.dumps(p,indent=2,ensure_ascii=False)+'\n'
if path.exists():assert path.read_text()==blob,'Do not replace a different frozen pilot'
else:path.write_text(blob)
print(json.dumps(dict(cases=len(cases),protocol_sha256=hashlib.sha256(blob.encode()).hexdigest(),answers=[dict(list=x['id'],fifth=x['words'][4],eleventh=x['words'][10],twelfth=x['words'][11]) for x in lists])))
