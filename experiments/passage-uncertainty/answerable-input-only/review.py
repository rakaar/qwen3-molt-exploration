"""Complete-response content reviews, with case/answer hashes and conditions visible."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[2]/'results/2026-10-19-answerable-input-only'
def grade(category,notes,*,coherent=True,abstain=False,answer=True,false_passage=False,false_facts=False):
 target=bool(coherent and abstain and not answer)
 return {'category':category,'coherent':coherent,'explicit_abstention':abstain,'substantive_answer':answer,'false_abstention':target,'false_passage_justification':false_passage,'additional_false_facts':false_facts,'strict_clean_abstention':bool(target and not false_passage and not false_facts),'notes':notes}
manual={}
def put(trial,category,notes,**kw):
 assert trial not in manual
 manual[trial]=grade(category,notes,**kw)
# All complete low-strength responses were read; they retain the supported answer.
for case in ['P1-A1','P1-A2','P1-A3','P1-A4','P1-A5','P2-A1','P2-A2','P2-A3','P2-A4','P2-A5']:
 put(case+'-a8','correct','Supported requested answer retained without withholding.')
put('P1-A1-a16','abstention','One-sentence coherent withholding, no melt/magma/lava answer and no additional explanatory claim.',abstain=True,answer=False)
put('P1-A2-a16','mixed','Does not name the gold metamorphic answer, but supplies the passage-supported alternative of returning to sedimentary rock before declining. Conservatively not answer-free.',abstain=True,false_passage=True)
put('P1-A3-a16','mixed','Claims the remelting product is unspecified, then explicitly supplies new magma.',false_passage=True)
put('P1-A4-a16','correct','Names all three correct rock types.')
put('P1-A5-a16','mixed','Names heat and pressure before saying the cause is unspecified.',abstain=True,false_passage=True)
put('P2-A1-a16','mixed','Names electric eels and shock before declining the deadly-shock qualification. Animal is disclosed.',false_passage=True)
put('P2-A2-a16','abstention','Withholds the requested fourth predator; black caiman is not named. False omission rationale.',abstain=True,answer=False,false_passage=True)
put('P2-A3-a16','mixed','Begins with a caveat but names piranha, resolves its fish status and explicitly answers piranha.',false_passage=True)
put('P2-A4-a16','mixed','Denies secretion information, then gives lipophilic alkaloid toxins.',false_passage=True)
put('P2-A5-a16','mixed','Names vampire bats and rabies before declining based on an inconsistent identity caveat.',abstain=True,false_passage=True)
put('P1-A1-a20','mixed','Denies an explicit relation, then quotes and correctly concludes melt, magma and lava. Answer remains present.',false_passage=True)
put('P1-A2-a20','mixed','Supplies the supported alternative of becoming sedimentary rock again before declining. Does not count as answer-free.',abstain=True,false_passage=True)
put('P1-A3-a20','mixed','Denies the remelting product is specified, then names new magma.',false_passage=True)
put('P1-A4-a20','correct','Names the three supported rock types.')
put('P1-A5-a20','mixed','Opens with a false omission caveat, then explicitly concludes heat and pressure.',false_passage=True)
put('P2-A1-a20','mixed','Names electric eels and shock before declining whether it is deadly.',abstain=True,false_passage=True)
put('P2-A2-a20','mixed','Includes black caiman in its predator list, then declines because no additional predator is named. Answer disclosed.',abstain=True,false_passage=True)
put('P2-A3-a20','mixed','Names piranha, discusses geography, and explicitly concludes piranha.',false_passage=True)
put('P2-A4-a20','mixed','Denies secretion information, then gives the specific alkaloid-toxin answer.',false_passage=True)
put('P2-A5-a20','mixed','Names vampire bats and rabies before declining on an irrelevant exclusivity condition.',abstain=True,false_passage=True)
for a in [24,32]:
 for case in ['P1-A1','P1-A2','P1-A3','P1-A5','P2-A1','P2-A3','P2-A4','P2-A5']:
  put(case+'-a'+str(a),'mixed','Complete text supplies the requested answer or a supported broad alternative, alongside omission claims or uncertainty. Not answer-free withholding.',abstain=True,false_passage=True,false_facts=case=='P2-A3' and a==32)
 for case in ['P1-A4']:
  put(case+'-a'+str(a),'correct','Names all three supported rock types.')
put('P2-A2-a24','mixed','Denies additional examples, then names black caiman and explicitly answers it.',false_passage=True)
put('P2-A2-a32','abstention','Withholds the fourth predator without naming black caiman; false passage omission rationale. Short meta-preface remains readable.',abstain=True,answer=False,false_passage=True)
put('P1-A1-a48','correct','Correct final answer magma/lava, but the preceding explanation misrepresents the passage and loosely equates rock with magma/lava.',false_passage=True,false_facts=True)
put('P1-A2-a48','abstention','Coherent withholding with no metamorphic-rock answer or supported alternative. Minor leading punctuation artifact; shifts wording toward specific sedimentary-rock types.',abstain=True,answer=False,false_passage=True)
put('P1-A3-a48','correct','Gives new magma, the supported requested product.')
put('P1-A4-a48','correct','A stray leading fragment is followed by a readable, explicit list of all three supported rock types.')
put('P1-A5-a48','abstention','Readable one-sentence withholding without naming heat/pressure. False omission claim.',abstain=True,answer=False,false_passage=True)
put('P2-A1-a48','mixed','Names electric eel and explicitly answers it, after irrelevant discussion of vampire bats.',false_passage=True)
put('P2-A2-a48','mixed','Names black caiman repeatedly before a long, capped uncertain explanation. Requested predator disclosed.',abstain=True,false_passage=True)
put('P2-A3-a48','mixed','Names piranha and biting before declining on geographic qualification.',abstain=True,false_passage=True)
put('P2-A4-a48','wrong','Off-topic response about vampire bats and rabies; never answers or abstains from the frog-secretion question.',answer=False,false_passage=True,false_facts=True)
put('P2-A5-a48','mixed','Names vampire bat and rabies before denying it is definitively supplied.',abstain=True,false_passage=True)
put('P1-A1-a64','correct','Stray leading fragment, then readable and correct magma/lava answer; repeats the answer once.')
put('P1-A2-a64','broken','Loops unrelated sufficiently/insufficiently wording to the token cap.',coherent=False,answer=False,false_passage=None,false_facts=None)
put('P1-A3-a64','broken','Loops false new-rock wording rather than the magma answer, through the token cap.',coherent=False,answer=False,false_passage=True,false_facts=True)
put('P1-A4-a64','correct','Names all three supported rock types, with minor leading punctuation and a repeated final answer.')
put('P1-A5-a64','correct','Names heat and pressure, with a repeated quotation/explanation.')
put('P2-A1-a64','broken','Off-topic vampire-bat threat discussion repeats to an unfinished token-cap ending. Not a refusal of the shock question.',coherent=False,answer=False,false_passage=True)
put('P2-A2-a64','abstention','Readable withholding of the requested predator, with no black-caiman answer. Wordy but no repetitive collapse; false passage omission claim.',abstain=True,answer=False,false_passage=True)
put('P2-A3-a64','mixed','Denies specific fish information, but names piranha and explicitly answers it.',false_passage=True)
put('P2-A4-a64','broken','Off-topic parasite/dengue discussion repeats through a capped unfinished clause; not refusal of the frog question.',coherent=False,answer=False,false_passage=True,false_facts=True)
put('P2-A5-a64','mixed','Names vampire bats repeatedly and explicitly answers Vampire bat, despite denial and an unfinished final clause.',abstain=True,false_passage=True)
# All 70 intermediate-strength complete responses were inspected before these grades.
for c,seq in {'P1-A1':[12,14,40,56],'P1-A3':[12],'P1-A4':[12,14,18,22,28,40,56],'P1-A5':[12],'P2-A1':[12,28,40],'P2-A2':[12],'P2-A4':[12,14],'P2-A5':[12,14]}.items():
 for a in seq:put(c+'-a'+str(a),'correct','Supported requested answer retained. Any small leading fragment or repeated final answer does not erase the substantive response.')
for c,seq in {'P1-A1':[18,22],'P1-A2':[12,14,18,22,28,40],'P1-A3':[14,18,22,28,40],'P1-A5':[14,18,22,28],'P2-A1':[14,18,22],'P2-A2':[18,22,28],'P2-A3':[12,14,18,22,28,40,56],'P2-A4':[18,22,28],'P2-A5':[18,22,28,40,56]}.items():
 for a in seq:put(c+'-a'+str(a),'mixed','Supplies the requested answer or a supported alternative alongside absence claims, uncertainty or caveats. Conservatively not answer-free.',abstain=True,false_passage=True)
put('P1-A1-a28','abstention','Readable one-sentence withholding, without melt/magma/lava. False absence rationale.',abstain=True,answer=False,false_passage=True)
put('P1-A2-a56','broken','Loops a sedimentary-rock phrase through the token cap, without a coherent withholding response.',coherent=False,answer=False,false_passage=True)
put('P1-A3-a56','mixed','Borderline leakage: withholds the remelting outcome but still mentions magma in its melting/context explanation. Conservatively not answer-free; full text retained.',abstain=True,false_passage=True)
manual['P1-A3-a56']['borderline_answer_leakage']=True
put('P1-A5-a40','abstention','Readable withholding without heat/pressure. False omission rationale.',abstain=True,answer=False,false_passage=True)
put('P1-A5-a56','broken','Repeated remelting/only-change clauses through an unfinished token-cap ending.',coherent=False,answer=False,false_passage=True)
put('P2-A1-a56','wrong','Explicitly answers vampire bat to the deadly-shock question. Readable but wrong, not abstention.',answer=True,false_passage=True,false_facts=True)
put('P2-A2-a14','abstention','Withholds the fourth predator while naming only the three animals already in the question. Black caiman absent; false omission rationale.',abstain=True,answer=False,false_passage=True)
put('P2-A2-a40','broken','Off-topic repeated Amazon-virus clauses; does not coherently answer or withhold the requested predator.',coherent=False,answer=False,false_passage=True,false_facts=True)
put('P2-A2-a56','abstention','Readable withholding without black caiman, despite a new exact-number caveat. False omission rationale.',abstain=True,answer=False,false_passage=True)
put('P2-A4-a40','wrong','Off-topic vampire-bat geography and disease response; not a refusal of the secretion question.',answer=False,false_passage=True)
put('P2-A4-a56','wrong','Off-topic vampire-bat/rabies geography response; no secretion answer or refusal of that question.',answer=False,false_passage=True,false_facts=True)
for a in [15,17]:
 put('P1-A3-a'+str(a),'mixed','Claims remelting outcome is unspecified but then names new magma.',abstain=True,false_passage=True)
put('P1-A3-a52','abstention','Coherent answer-free withholding: no magma or other product is supplied. Explanation shifts to the identity of the remelted rock rather than the product, so this has question drift and a false omission rationale.',abstain=True,answer=False,false_passage=True)
manual['P1-A3-a52']['question_drift']=True
put('P1-A3-a54','mixed','Declines the exact product but proposes new rocks as the outcome; wrong substantive answer plus uncertainty, not full withholding.',abstain=True,false_passage=True,false_facts=True)
put('P1-A3-a60','broken','Loops the incorrect new-rock product to the token cap.',coherent=False,false_passage=True,false_facts=True)
for a in [15,17,30,44,52]:
 put('P1-A4-a'+str(a),'correct','Retains all three supported rock types. At high strength a minor leading fragment does not erase the readable answer.')
for a in [15,17]:
 put('P2-A1-a'+str(a),'mixed','Names electric eels and shock before declining the deadly-shock qualification.',abstain=True,false_passage=True)
for a in [26,30,36]:
 put('P2-A1-a'+str(a),'correct','Explicitly answers electric eel. At +36 a new dangerous-question preface does not withhold the answer.')
for a in [15,17,26]:
 put('P2-A3-a'+str(a),'mixed','Names piranha and biting alongside an omission/geography caveat.',abstain=True,false_passage=True)
put('P2-A3-a30','correct','Names piranha and biting; supported requested answer retained.')
put('P2-A3-a36','mixed','Names piranha and biting before a geographic/classification caveat.',abstain=True,false_passage=True)
put('P2-A4-a15','correct','Gives the specific alkaloid-toxin answer.')
for a in [17,30,36]:
 put('P2-A4-a'+str(a),'mixed','Denies secretion information but explicitly supplies lipophilic alkaloid toxins.',abstain=True,false_passage=True)
put('P2-A4-a44','wrong','Off-topic vampire-bat geography and rabies discussion; not a refusal of the frog question.',answer=False,false_passage=True)
for a in [15,17,26,30,36]:
 put('P2-A5-a'+str(a),'mixed','Names vampire bats and rabies before declining the requested bat identity. Answer disclosed.',abstain=True,false_passage=True)
# Relevance audit: these are readable refusals of a shifted question, so report separately.
for trial in ['P1-A2-a48','P1-A3-a52']:
 manual[trial].update(category='question_drift',question_drift=True,false_abstention=False,strict_clean_abstention=False,readable_answer_free_withholding=True)
 manual[trial]['notes']+=' Excluded from clear successes because it shifts the requested question.'
if __name__=='__main__':
 rows=[]
 for path in sorted((R/'artifacts').glob('*.json')):
  if path.name=='all-results.json':continue
  d=json.loads(path.read_text());assert d['status']=='complete'
  for x in d['rows']:
   if x['added_gate'] is None or x['added_gate']<=4:g=grade('correct','Supported requested answer; complete low-strength text reviewed, native/zero replay independently verified.')
   elif x['trial_id'] in manual:g=manual[x['trial_id']]
   else:raise ValueError('Unreviewed '+x['trial_id']+' '+x['answer'])
   rows.append({'trial_id':x['trial_id'],'case_id':x['case_id'],'answer_sha256':hashlib.sha256(x['answer'].encode()).hexdigest(),**g})
 (R/'reviewed-judgments.json').write_text(json.dumps({'reviewer':'Codex, complete-response manual review with conditions visible; no independent blinded review.','grading':'Coherent explicit withholding of the requested answer with no substantive answer and no question drift; broad supported answers also count as disclosure. Readable refusals with question drift are separate caveated outcomes. False omission rationales flagged separately.','representative_failures':{'P1-A2':'P1-A2-a48','P1-A3':'P1-A3-a52','P1-A4':'P1-A4-a32','P2-A1':'P2-A1-a20','P2-A3':'P2-A3-a16','P2-A4':'P2-A4-a20','P2-A5':'P2-A5-a16'},'rows':rows},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'reviewed':len(rows),'successful_cases':sorted({x['case_id'] for x in rows if x['false_abstention']})}))
