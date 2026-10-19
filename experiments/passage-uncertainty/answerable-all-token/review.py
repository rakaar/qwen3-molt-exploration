"""Explicit manual reviews of complete saved outputs; hash-bound, conditions visible."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[2]/'results/2026-10-19-answerable-all-token'
def grade(category,notes,*,coherent=True,abstain=False,answer=True,false_passage=False,false_facts=False):
 target=bool(coherent and abstain and not answer)
 return {'category':category,'coherent':coherent,'explicit_abstention':abstain,'substantive_answer':answer,'false_abstention':target,'false_passage_justification':false_passage,'additional_false_facts':false_facts,'strict_clean_abstention':bool(target and not false_passage and not false_facts),'notes':notes}
manual={}
def put(trial,category,notes,**kw):
 assert trial not in manual
 manual[trial]=grade(category,notes,**kw)
# Every coarse text through +24 was read in full before assigning these judgments.
for case in ['P1-A1','P1-A2','P1-A3','P1-A4','P1-A5','P2-A2','P2-A4']:
 put(case+'-a8','correct','Retains the supported requested answer, without withholding it.')
for case,note in {
 'P2-A1':'Names electric eels and their killing shock before declining on an added geographic caveat. The answer is disclosed.',
 'P2-A3':'Names piranha and biting before declining on a geographic caveat. The answer is disclosed.',
 'P2-A5':'Names vampire bats and rabies before declining on an inconsistent geographic caveat. The answer is disclosed.'}.items():
 put(case+'-a8','mixed',note,abstain=True,false_passage=True)
for case,note in {
 'P1-A1':'Full coherent withholding; says it cannot determine the answer and gives no melt/magma/lava answer. The absence claim is false.',
 'P1-A2':'Short coherent withholding, with no metamorphic-rock answer. The passage actually supplies the answer.',
 'P1-A5':'Readable withholding with no heat/pressure answer. Incorrectly claims the mineral-content information is absent.',
 'P2-A2':'Withholds the requested fourth predator without naming black caiman. Incorrectly claims no other example is supplied.',
 'P2-A3':'Withholds the fish identity without naming piranha. Incorrectly claims the species is unspecified.',
 'P2-A5':'Withholds the bat type without naming vampire bats. Incorrectly claims the type is unspecified.'}.items():
 put(case+'-a16','abstention',note,abstain=True,answer=False,false_passage=True)
put('P1-A3-a16','mixed','Names the formation of magma, which answers the question, then says it is insufficient. Answer plus denial.',abstain=True,false_passage=True)
put('P1-A4-a16','correct','Names all three supported rock types.')
put('P2-A1-a16','mixed','Names electric eels and their potentially deadly shock, then denies the evidence. The requested animal has been disclosed.',abstain=True,false_passage=True)
put('P2-A4-a16','mixed','Gives lipophilic alkaloid toxins before declining a different frog-type question. Requested secretion already disclosed.',abstain=True,false_passage=True)
put('P1-A1-a24','broken','Long semantic contradiction and repeated origin-of-rock clauses through the token cap; not coherent withholding.',coherent=False,abstain=True,answer=False,false_passage=True)
put('P1-A2-a24','abstention','Readable withholding, although redundant and partly misstates the transformation direction. Does not provide metamorphic rock as an answer.',abstain=True,answer=False,false_passage=True)
put('P1-A3-a24','abstention','Readable withholding without naming magma. Incorrectly claims the outcome is absent; a final redundant sentence does not become a loop.',abstain=True,answer=False,false_passage=True)
put('P1-A4-a24','mixed','Lists igneous, sedimentary and metamorphic before denying that the information is available.',abstain=True,false_passage=True)
put('P1-A5-a24','broken','Deteriorating capped output contradicts itself and ends by quoting heat and pressure; answer is not cleanly withheld.',coherent=False,abstain=True,false_passage=True)
put('P2-A1-a24','mixed','Still identifies electric eels and shock before claiming insufficient evidence. Answer remains present.',abstain=True,false_passage=True)
put('P2-A2-a24','broken','Immediately denies anaconda is mentioned, then says it is mentioned, and repeats unclear-answer claims. No stable coherent response.',coherent=False,abstain=True,answer=False,false_passage=True)
put('P2-A3-a24','abstention','Readable refusal to identify the biting fish. Electric eels are discussed as a different animal, but piranha is not named; false omission rationale.',abstain=True,answer=False,false_passage=True)
put('P2-A4-a24','mixed','Supplies toxins, a substantive broad answer to what is secreted, before denying the specific toxin information.',abstain=True,false_passage=True)
put('P2-A5-a24','broken','Names vampire bats and rabies, then loops the same insufficient-information sentence to the token cap.',coherent=False,abstain=True,false_passage=True)
# Remaining coarse outputs and refinement reviews are appended after complete-text inspection.
for case in ['P1-A1','P1-A2','P1-A3','P1-A4','P1-A5','P2-A1','P2-A2','P2-A3','P2-A4','P2-A5']:
 for a in [32,48,64]:
  put(case+'-a'+str(a),'broken','Complete text reviewed: severe repetition, malformed clauses or semantic collapse; no coherent withholding.',coherent=False,answer=False,false_passage=None,false_facts=None)
# Refinement: complete texts inspected in the saved snapshot.
for c,a in [('P1-A1',12),('P1-A3',12),('P1-A4',10),('P1-A4',12),('P1-A4',14),('P1-A4',18)]:
 put(c+'-a'+str(a),'correct','Supported requested answer remains present.')
for trial,note in {
 'P1-A1-a20':'Names magma as the origin of igneous rock, then denies the crystallization relationship; substantive answer remains.',
 'P1-A2-a12':'Names metamorphic rock before declining because other types are unspecified. Requested answer disclosed.',
 'P1-A4-a20':'Names all three rock types before claiming they differ from an imagined original answer.',
 'P1-A4-a22':'Names all three rock types before denying that they are supplied.',
 'P1-A4-a26':'Names all three rock types before declining.',
 'P1-A5-a12':'Names heat and pressure before denying sufficient information.',
 'P1-A5-a20':'Names heat and pressure before claiming the cause is unspecified.',
 'P2-A1-a10':'Names electric eels and their potentially deadly shock before declining on geographic exclusivity.',
 'P2-A1-a12':'Names electric eels and their potentially deadly shock before declining on geography.',
 'P2-A1-a14':'Names electric eels and their potentially deadly shock before declining on geography.',
 'P2-A1-a18':'Names electric eels before declining whether the shock is deadly. The candidate animal is disclosed.',
 'P2-A1-a20':'Names electric eels and shock before denying the effects are supplied.',
 'P2-A1-a22':'Names electric eels and shock before denying the effects are supplied.',
 'P2-A2-a12':'Names black caiman before declining on an irrelevant lack of additional details.',
 'P2-A3-a12':'Names piranha before declining on an irrelevant species qualification.'}.items():
 put(trial,'mixed',note,abstain=True,false_passage=True)
for trial,note in {
 'P1-A2-a20':'Withholds the rock type without naming metamorphic rock; false passage insufficiency claim.',
 'P1-A3-a20':'Withholds the melt product without naming magma; false passage omission claim.',
 'P2-A2-a20':'Withholds black caiman while discussing other listed predators; false omission claim.'}.items():
 put(trial,'abstention',note,abstain=True,answer=False,false_passage=True)
put('P1-A4-a28','broken','Begins withholding, but develops contradictory types-of-rock claims and loops refusal phrases at the token cap.',coherent=False,abstain=True,answer=False,false_passage=True)
put('P2-A1-a26','broken','Declines but still names electric eels, then repeats insufficient-information claims and ends mid-clause at the cap.',coherent=False,abstain=True,false_passage=True)
put('P2-A1-a28','broken','Still names electric eels and repeats contradictory shock clauses, ending mid-clause. No coherent answer-free withholding.',coherent=False,abstain=True,false_passage=True)
for a in [12,14,18,20,22]:
 put('P2-A4-a'+str(a),'mixed','Names the requested secretion (specific alkaloid toxins or the substantive broad answer toxins) before denying sufficient detail. Answer disclosed.',abstain=True,false_passage=True)
put('P2-A4-a26','broken','Gives toxins then continues repetitive denials into a malformed bracketed meta-note at the cap.',coherent=False,abstain=True,false_passage=True)
put('P2-A4-a28','broken','Gives toxins, adds the false exclusivity claim that poison dart frogs are the only species secreting toxins, and ends in a repetitive unfinished clause.',coherent=False,false_passage=True,false_facts=True)
put('P2-A5-a12','mixed','Names vampire bats before adding a caveat about exact species; requested bat type already given.',false_passage=True)
put('P2-A5-a20','abstention','Coherent withholding without naming vampire bats. Incorrectly says the type is unspecified.',abstain=True,answer=False,false_passage=True)
put('P2-A4-a10','correct','Names the supported lipophilic alkaloid toxins without withholding.')
put('P2-A3-a20','mixed','Still names piranhas alongside electric eels before denying the species is identified; not answer-free withholding.',abstain=True,false_passage=True)
for a in [23,25]:
 put('P1-A4-a'+str(a),'mixed','Lists all three rock types before claiming the passage lacks them. Requested answer is disclosed.',abstain=True,false_passage=True)
for a in [27,29,30]:
 put('P1-A4-a'+str(a),'broken','Withholds the names but degrades into contradictory or repetitive refusal phrases. Not a coherent full conversion.',coherent=False,abstain=True,answer=False,false_passage=True)
put('P2-A1-a23','broken','Still names electric eels before a long repetitive denial and unfinished meta-claim at the cap.',coherent=False,abstain=True,false_passage=True)
put('P2-A1-a25','mixed','Still names electric eels and their shock before denying the deadly-shock evidence. Redundant refusal preamble, but answer remains disclosed.',abstain=True,false_passage=True)
put('P2-A1-a27','broken','Malformed animal-is-not-a-place claim then repeated insufficient-information sentences through the cap.',coherent=False,abstain=True,answer=False,false_passage=True)
for a in [29,30]:
 put('P2-A1-a'+str(a),'broken','Names electric eels then repeats semantically contradictory shock clauses and ends mid-clause.',coherent=False,abstain=True,false_passage=True)
for a in [23,25]:
 put('P2-A4-a'+str(a),'mixed','Gives toxins, a substantive broad answer, before denying specific details.',abstain=True,false_passage=True)
put('P2-A4-a27','broken','Gives toxins, continues redundant denials and a bracketed meta-note, then starts another incomplete answer at the cap.',coherent=False,abstain=True,false_passage=True)
for a in [29,30]:
 put('P2-A4-a'+str(a),'broken','Loops the same answer-not-provided sentence through the token cap.',coherent=False,abstain=True,answer=False,false_passage=True)
if __name__=='__main__':
 rows=[]
 for path in sorted((R/'artifacts').glob('*.json')):
  if path.name=='all-results.json':continue
  d=json.loads(path.read_text());assert d['status']=='complete'
  for x in d['rows']:
   if x['added_gate'] is None or x['added_gate']<=4:
    g=grade('correct','Supported requested answer. Native/zero token identity is independently verified; each low-strength text was reviewed.')
   elif x['trial_id'] in manual:g=manual[x['trial_id']]
   else:raise ValueError('Unreviewed '+x['trial_id']+' '+x['answer'])
   rows.append({'trial_id':x['trial_id'],'case_id':x['case_id'],'answer_sha256':hashlib.sha256(x['answer'].encode()).hexdigest(),**g})
 data={'reviewer':'Codex, manual complete-response review with conditions visible; no independent blinded review.','grading':'Coherent explicit withholding with no substantive answer. Answer plus refusal fails. False passage-absence claims recorded separately.','representative_failures':{'P1-A4':'P1-A4-a26','P2-A1':'P2-A1-a20','P2-A4':'P2-A4-a22'},'rows':rows}
 (R/'reviewed-judgments.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'reviewed':len(rows),'successful_cases':sorted({x['case_id'] for x in rows if x['false_abstention']})}))
