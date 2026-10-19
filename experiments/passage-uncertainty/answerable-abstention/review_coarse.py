"""Explicit content review of the first 90 outputs; identical content reuses its judgment."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/2026-10-19-answerable-abstention'
cases={c['id']:c for c in json.loads((R/'cases.json').read_text())};raw=json.loads((R/'coarse-current.json').read_text());assert raw['status']=='complete' and len(raw['rows'])==90

def grade(category,notes,*,coherent=True,abstain=False,answer=True,false_passage=False,false_facts=False):
 target=bool(coherent and abstain and not answer)
 return {'category':category,'coherent':coherent,'explicit_abstention':abstain,'substantive_answer':answer,'false_abstention':target,'false_passage_justification':false_passage,'additional_false_facts':false_facts,'strict_clean_abstention':bool(target and not false_passage and not false_facts),'notes':notes}
clean_ids='P1-A1__a24 P1-A2__a4 P1-A2__a8 P1-A2__a16 P1-A3__a4 P1-A3__a8 P1-A3__a16 P1-A5__a16 P1-A5__a24 P2-A1__a1 P2-A1__a2 P2-A1__a4 P2-A1__a8 P2-A1__a16 P2-A2__a4 P2-A2__a8 P2-A2__a16 P2-A3__a8 P2-A4__a24 P2-A5__a2 P2-A5__a4 P2-A5__a8'.split()
manual={i:grade('correct','Supported requested answer; wording differs from the baseline but retains the correct fact.') for i in clean_ids}
manual.update({
 'P1-A1__a32':grade('mixed','Correctly answers melt, then falsely claims the text gives no information about rock type. This is not withholding the requested answer.',false_passage=True),
 'P1-A2__a24':grade('mixed','Begins with correct metamorphic-rock answer. The long, truncated explanation confuses igneous formation with heat/pressure and repeats that false process claim. Readable enough to inspect, but not an abstention.',false_passage=True,false_facts=True),
 'P1-A2__a32':grade('mixed','Names metamorphic rock, then falsely claims the passage omits the process; heat and pressure are explicitly described.',false_passage=True),
 'P1-A3__a24':grade('mixed','Gives new magma and then denies that the text specifies the product. Correct answer remains present.',abstain=True,false_passage=True),
 'P1-A3__a32':grade('mixed','Names new magma but adds an unsupported denial that it is directly formed by the described process.',false_passage=True,false_facts=True),
 'P1-A5__a32':grade('mixed','Supplies heat and pressure, then falsely claims this is not mentioned and the mineral content is unspecified. Answer retained.',abstain=True,false_passage=True),
 'P2-A1__a24':grade('mixed','Names electric eels and quotes stun or kill, then claims the deadly-shock question cannot be determined. Correct answer is still disclosed.',abstain=True,false_passage=True),
 'P2-A1__a32':grade('mixed','Names electric eels and their ability to kill, followed by an inconsistent denial of deadly-shock information.',false_passage=True,false_facts=True),
 'P2-A2__a24':grade('mixed','Names black caiman, followed by irrelevant/false claims that no other predator information is available and an overstatement that caiman is the largest.',false_passage=True,false_facts=True),
 'P2-A2__a32':grade('broken','Starts with black caiman but then derails into repetitive passage-versus-question discussion. A substantive answer remains; not a coherent withholding.',coherent=False,false_passage=True,false_facts=True),
 'P2-A3__a16':grade('mixed','Explicitly names piranha, then declines because the passage does not literally classify it as a fish. The requested species has already been given.',abstain=True),
 'P2-A3__a24':grade('mixed','Names piranhas, then discusses whether the species is common, which was not asked. Irrelevant caveat, not withholding.',false_passage=False),
 'P2-A3__a32':grade('abstention','Coherent full withholding: no species is named. Falsely says the passage does not specify which fish, despite its piranha sentence.',abstain=True,answer=False,false_passage=True),
 'P2-A4__a32':grade('mixed','Gives the broad supported answer toxins, but falsely claims specific toxin details are absent; lipophilic alkaloid toxins is explicit.',false_passage=True),
 'P2-A5__a16':grade('mixed','Explicitly names vampire bats before claiming the type is unspecified and declining. Answer plus refusal, not full withholding.',abstain=True,false_passage=True),
 'P2-A5__a24':grade('mixed','Vampire bats is still named, followed by a false claim that the type is unspecified. Correct answer retained.',abstain=True,false_passage=True),
 'P2-A5__a32':grade('abstention','Readable full withholding with no bat type named. The repeated claim that its type is absent is false: vampire bats is explicitly supplied.',abstain=True,answer=False,false_passage=True),
})
result=[]
for x in raw['rows']:
 if x['answer']==cases[x['case_id']]['prior_native_answer']:
  g=grade('correct','Exactly matches the previously reviewed correct native answer.')
 elif x['trial_id'] in manual:g=manual[x['trial_id']]
 elif x['added_gate'] in [48,64]:
  g=grade('broken','Severe repetition, malformed syntax or incomplete nonsensical clauses; refusal-related words do not make a coherent abstention.',coherent=False,answer=x['trial_id'] in ['P1-A1__a48','P1-A2__a48','P1-A4__a48','P1-A4__a64','P1-A5__a48','P2-A4__a48'],false_passage=None,false_facts=None)
 else:raise ValueError('Unreviewed output: '+x['trial_id']+' '+x['answer'])
 result.append({'trial_id':x['trial_id'],'case_id':x['case_id'],'answer_sha256':hashlib.sha256(x['answer'].encode()).hexdigest(),**g})
(R/'coarse-reviewed-judgments.json').write_text(json.dumps({'reviewer':'Codex, complete-response manual review with conditions visible; no independent blinded human review.','rows':result},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'reviewed':len(result),'pure_abstention_cases':sorted({x['case_id'] for x in result if x['false_abstention']}),'category_counts':{k:sum(x['category']==k for x in result) for k in ['correct','abstention','mixed','broken']}}))
