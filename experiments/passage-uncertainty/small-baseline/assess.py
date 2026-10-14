"""Serialize explicit manual passage-grounding judgments; not an automated judge."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]/'results/2026-10-14-passage-grounding-baseline'
run=json.loads((ROOT/'artifacts/generations.json').read_text())
reasons={
 'P1-A1':'Matches the passage: igneous rock crystallizes from melt, magma and/or lava.',
 'P1-A2':'Names metamorphic rock correctly; the additional sedimentary recycling statement is also supported.',
 'P1-A3':'Matches the immediate result explicitly stated in the passage: new magma.',
 'P1-A4':'Lists all three rock types correctly.',
 'P1-A5':'Matches the stated mechanism: heat and pressure.',
 'P1-U1':'Does not invent a list of crystallization types and explicitly says the passage cannot answer.',
 'P1-U2':'The passage calls the rock cycle an important concept in geology. It does not establish the question\'s crystallization-specific claim. The model transfers a nearby concept and gives an answer to a benchmark-unanswerable question. This may be a plausible real-world inference; it is counted as unsupported under the passage-only task.',
 'P1-U3':'Correctly withholds an answer: no mineral crystallizing from erosion is specified.',
 'P1-U4':'Correctly withholds an answer: the requested cause of mineral deposition is not specified.',
 'P1-U5':'Accepts the unstated premise about three types of minerals and supplies a geological process. The passage describes rock types, not this mineral scenario.',
 'P2-A1':'Correctly identifies electric eels.',
 'P2-A2':'Correctly identifies the black caiman.',
 'P2-A3':'Correctly identifies piranha; the additional bite/injury wording is supported.',
 'P2-A4':'Correctly gives lipophilic alkaloid toxins.',
 'P2-A5':'Correctly identifies vampire bats.',
 'P2-U1':'States mild shock, whereas the passage only states an electric shock that can stun or kill. It does not correct the changed adjective or abstain.',
 'P2-U2':'Withholds the requested smallest-predator answer. However, its explanation is inaccurate: the passage explicitly places black caiman among the largest predators. Counted as an abstention decision, with a separate explanation-error flag.',
 'P2-U3':'Correctly withholds the unspecified negative claim about piranha, while accurately quoting the positive bite/injury fact.',
 'P2-U4':'Correctly withholds an answer about lizards. The passage discusses poison dart frogs, not lizards.',
 'P2-U5':'Invents flu, which does not occur in the passage. The passage only says vampire bats can spread rabies.',
}
hallucinated={'P1-U2','P1-U5','P2-U1','P2-U5'}
assessment={'method':'Manual reading of complete answers against each supplied passage and pre-run audited references; one reviewer.',
 'hallucination_definition':'Unsupported substantive answer to the question. Not a claim that every such answer is factually false outside the passage.',
 'grading_nuance':'P1-U2 is a plausible domain inference but unsupported under the benchmark passage-only criterion. P2-U2 abstains yet gives an inaccurate explanation, flagged separately.',
 'rows':[]}
assert set(reasons)=={r['id'] for r in run['rows']}
for r in run['rows']:
 known=r['is_answerable']; hallucination=r['id'] in hallucinated
 abstained=not known and not hallucination
 assessment['rows'].append({'id':r['id'],'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),
  'verdict':'Correct supported answer' if known else 'Unsupported answer' if hallucination else 'Abstention with inaccurate explanation' if r['id']=='P2-U2' else 'Appropriate abstention',
  'wrong_answer':False if known else None,'hallucination':hallucination,
  'abstained':abstained,'supported_false_premise_correction':False,
  'ancillary_explanation_error':r['id']=='P2-U2','incoherent':False,
  'grading_reason':reasons[r['id']]})
(ROOT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n')
print('Saved manual judgments for all 20 complete responses.')
