"""Post-scoring composition audit; never rewrites descriptions or judge decisions."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-11-10-t2375-heldout-composition';OUT.mkdir(exist_ok=True)
source=ROOT/'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json'
feature=json.loads(source.read_text())
df=json.loads((ROOT/'results/2026-11-10-t2375-detection-fuzz-heldout/summary.json').read_text())
af=json.loads((ROOT/'results/2026-11-09-t2375-active-fuzz-local-judge/summary.json').read_text())
assert df['status']==af['status']=='complete'
freeze=json.loads((ROOT/'experiments/t2375-explanation-refinement-20261109/protocol.json').read_text())
for arm in ['baseline','candidate']:
    text=(ROOT/f'experiments/t2375-explanation-refinement-20261109/{arm}-explanation.txt').read_text().strip()
    assert hashlib.sha256(text.encode()).hexdigest()==freeze[arm+'_sha256']

def water(text):return bool(re.search('water',text,re.I))
def eleven(text):return bool(re.search(r'(?<!\d)11(?!\d)|\beleven(?:th)?\b',text,re.I))
def after_ten(text):
    # Candidate boundary only: no omitted continuation is inferred or fabricated.
    return bool(re.search(r'(?:^|[\s.])10\.[\s\S]*$',text)) and not bool(re.search(r'(?:^|[\s.])11\.',text))
def fingerprint(row):return hashlib.sha256(json.dumps([row['str_tokens'],row['activations']],ensure_ascii=False).encode()).hexdigest()
sets=[];methods={};details={};corrected={}
for method in ['detection','fuzz','active_fuzz']:
    rows=feature['methods'][method]['rows']
    score=af if method=='active_fuzz' else df['methods'][method]
    pairs={r['row_index']:r for r in score['pairs']}
    groups={};entries=[]
    sets.append({fingerprint(r) for r in rows if r['activating']})
    for i,row in enumerate(rows):
        text=''.join(row['str_tokens']);positive=row['activating']
        if positive:
            family='water_subtoken_active' if any(a>0 and water(t) for t,a in zip(row['str_tokens'],row['activations'])) else 'other_active'
            g=groups.setdefault(family,dict(n=0,baseline_correct=0,candidate_correct=0,visible_literal_eleven=0,
                visible_november=0,after_list_item_10=0,last_token_active=0,last_token_is_activation_maximum=0))
            g['n']+=1
            for arm in ['baseline','candidate']:g[arm+'_correct']+=pairs[i][arm]==positive
            g['visible_literal_eleven']+=eleven(text)
            g['visible_november']+=bool(re.search(r'\bnovember\b',text,re.I))
            g['after_list_item_10']+=after_ten(text)
            g['last_token_active']+=row['activations'][-1]>0
            g['last_token_is_activation_maximum']+=row['activations'][-1]==max(row['activations'])
            entries.append(dict(row_index=i,window_fingerprint=fingerprint(row),group=family,
                visible_literal_eleven=eleven(text),after_list_item_10=after_ten(text),
                baseline_correct=pairs[i]['baseline']==positive,candidate_correct=pairs[i]['candidate']==positive,
                active_tokens=[dict(position=j,text=t,gate=a) for j,(t,a) in enumerate(zip(row['str_tokens'],row['activations'])) if a>0]))
        if pairs[i]['baseline']!=positive and pairs[i]['candidate']==positive:
            corrected.setdefault(fingerprint(row),dict(positive=positive,water=water(text),november=bool(re.search(r'\bnovember\b',text,re.I)),methods=[]))['methods'].append(method)
    methods[method]=dict(positives=sum(r['activating'] for r in rows),negatives=sum(not r['activating'] for r in rows),
        positive_groups=groups,target_windows_with_visible_literal_eleven=sum(eleven(''.join(r['str_tokens'])) for r in rows),
        water_anywhere_positive=sum(r['activating'] and water(''.join(r['str_tokens'])) for r in rows),
        water_anywhere_negative=sum(not r['activating'] and water(''.join(r['str_tokens'])) for r in rows))
    details[method]=entries
assert sets[0]==sets[1]==sets[2] and len(sets[0])==50
result=dict(posthoc=True,performed_after_scoring_utc=datetime.now(timezone.utc).isoformat(),
    descriptions_unchanged=True,judge_decisions_unchanged=True,all_methods_share_same_50_positive_windows=True,
    rules=dict(water='Positive windows containing an actually active token whose displayed string includes water, case-insensitive; includes compounds and subtokens.',
        visible_eleven='Literal standalone 11 or English eleven/eleventh anywhere in the displayed target passage; excludes the explanation and fixed few-shots.',
        after_10='Visible list item 10 and no visible list item 11; candidate pre-11 boundary, NOT confirmed without continuation.'),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),methods=methods,
    unique_corrected_positive_windows=len(corrected),corrected_window_tags=list(corrected.values()))
(OUT/'composition.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT/'per-positive-window-audit.json').write_text(json.dumps(details,indent=2,ensure_ascii=False)+'\n')
f=methods['fuzz']['positive_groups'];d=methods['detection']['positive_groups']
lines=['# Post-hoc composition of the T2375 held-out sets','',
 'This analysis was performed after all judge decisions were complete. It changes neither description, case selection, labels, nor judge outputs. The categories below are descriptive post-hoc diagnostics, not a new held-out score.','',
 '## What the positive cases contain','',
 'Each test has 50 positive examples and 50 negatives. All three methods share the same 50 positive windows; they are not three independent collections of positives.','',
 '| Positive group | Count | Detection: old → revised correct | Fuzz: old → revised correct |',
 '|---|---:|---:|---:|',
 f"| Active water-containing token, including compounds | {f['water_subtoken_active']['n']} | {d['water_subtoken_active']['baseline_correct']} → {d['water_subtoken_active']['candidate_correct']} | {f['water_subtoken_active']['baseline_correct']} → {f['water_subtoken_active']['candidate_correct']} |",
 f"| Other active tokens | {f['other_active']['n']} | {d['other_active']['baseline_correct']} → {d['other_active']['candidate_correct']} | {f['other_active']['baseline_correct']} → {f['other_active']['candidate_correct']} |",'',
 'Water accounts for 29/50 (58%) of the positive examples. The 50 negative passages contain no water in Detection/Fuzz; Active Fuzz negatives include 29 water-containing passages but highlight inactive positions. Thus water is a majority of positives, not a majority of all Detection/Fuzz target passages.','',
 '## Eleven visibility and truncation','',
 '- None of the 100 displayed target passages per method contains literal standalone 11 or English eleven/eleventh. The revised explanation itself contains those strings; this count concerns example passages only.',
 '- One positive passage contains November. It is the only non-water positive newly classified correctly by the revised explanation.',
 '- Eight positive windows finish after list item 10, on punctuation or whitespace. These are plausible pre-11 boundary candidates, but their following tokens are absent, so they cannot be counted as confirmed pre-11 cases.',
 '- Other non-water windows include partial numbers/dates and October-related contexts. Their relationship to eleven is unresolved without more context.',
 f"- All 50 positive windows stop on an active token; {sum(g['last_token_is_activation_maximum'] for g in f.values())}/50 stop on their maximum activation. A following 11 or November could therefore be cropped out, but this audit does not assume that continuation.",'',
 '## Interpretation','',
 '- The supplied excerpts give little explicit evidence for the newly added eleven clause. This is a plausible contributor to the small observed score gains.',
 '- However, 21 positive examples are non-water, and the revised explanation identifies only 1 of those 21. The result cannot be attributed solely to an overwhelming abundance of easy water cases.',
 '- The scores alone do not distinguish incomplete wording, judge limitations, omitted context, or non-eleven behavior of this gate.',
 '- The two unique corrected positives across the three tests are the November-context window and a high-water compound. The new false positive in Active Fuzz is an inactive comma in a passage with water elsewhere.',
 '- A claim about pre-11 coverage requires verified full continuations or an independently specified evaluation of eleven-related prompts with gate labels and controls. Further description development should not tune to these now-viewed test cases.','']
(OUT/'README.md').write_text('\n'.join(lines))
files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json']
(OUT/'local-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
