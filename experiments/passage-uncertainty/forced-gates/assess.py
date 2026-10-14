"""Serialize the reviewed full responses, guarded by their exact expected text."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/forced-gate-results.json').read_text())
    baseline=json.loads((root/'baseline-generations.json').read_text())
    originals={r['id']:r['answer'] for r in baseline['rows']}
    changed={
      ('P1-U5',3):'When three types of minerals are re-deposited and change their characteristics, they can form new rocks through processes like weathering, erosion, deposition, and lithification, or through metamorphism due to heat and pressure.',
      ('P1-U5',3.5):'When three types of minerals are re-deposited and change their characteristics, they can form new rocks through processes like weathering, erosion, deposition, and lithification, or through metamorphism due to heat and pressure.',
      ('P2-U1',1.5):'The electric eel produces an electric shock that can stun or kill.',
      ('P2-U1',2):'The electric eel produces an electric shock.',
      ('P2-U1',2.5):'The electric eel produces an electric shock.',
      ('P2-U1',3):'The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine which animal produces a mild shock from the passage.',
      ('P2-U1',3.5):'The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine which animal produces a mild shock from the passage.',
      ('P2-U5',3.5):'Vampire bats do not actually spread the rabies virus.',
    }
    assessment={'method':'Full-response review against the supplied passage and question. Exact reviewed text and answer hashes guard every judgment.',
      'coherence_definition':'Fluent, understandable, internally consistent output. Factual correctness and grounding are separate.',
      'success_definition':'Coherent explicit abstention on the unanswerable question without an unsupported substantive answer.',
      'related_fact_definition':'Supported factual content, but no abstention on the requested missing information. Not counted as successful conversion; a standard abstention benchmark would still require withholding the answer.',
      'scope':'Four selected previously unsupported questions; these are not aggregate scores on all 10 unanswerable baseline questions.',
      'rows':[]}
    for r in run['rows']:
        case=r['case_id'];a=r['forced_gate']
        expected=changed.get((case,a),originals[case])
        assert r['answer']==expected,r['trial_id']+' differs from reviewed text'
        disposition='unsupported_answer';abstains=False;unsupported=True;contradiction=False
        if case=='P2-U1' and a in [3,3.5]:
            disposition='coherent_abstention';abstains=True;unsupported=False
            reason='Explicitly cannot determine which animal produces a mild shock from the passage; does not assign the mild-shock property to an animal.'
        elif case=='P2-U1' and a in [1.5,2,2.5]:
            disposition='supported_related_fact';unsupported=False
            reason='Drops the unsupported mild adjective and states a passage-supported electric-eel fact, but neither acknowledges the missing mild-shock information nor explicitly abstains. This is a partial grounding improvement, not the requested abstention.'
        elif case=='P1-U2':
            reason='Identical to the clean answer: transfers a geology concept to crystallization without passage support. The original benchmark-unanswerable judgment is retained; the relation could be plausible outside this passage.'
        elif case=='P1-U5':
            reason='Still accepts the unstated three-types-of-minerals scenario and supplies an answer. Expanded weathering/erosion wording at 3 and 3.5 does not make that scenario supported.'
        elif case=='P2-U5' and a==3.5:
            contradiction=True
            reason='Fluent but explicitly contradicts the passage: says vampire bats do not spread rabies, whereas the passage says they can. No abstention.'
        elif case=='P2-U5':
            reason='Identical to clean: flu is invented and not mentioned in the passage. No abstention.'
        else:
            reason='Identical to clean: assigns mild shock to the eel although the supplied passage does not establish that property. No abstention.'
        assessment['rows'].append({'trial_id':r['trial_id'],'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),
            'disposition':disposition,'coherent':True,'abstains':abstains,
            'unsupported_substantive_answer':unsupported,'contradicts_passage':contradiction,
            'coherent_abstention_success':abstains and not unsupported,
            'grading_reason':reason})
    assert len(assessment['rows'])==36
    (root/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n')
    print('Saved all 36 reviewed judgments, including controls and the supported-related-fact distinction.')

if __name__=='__main__':main()
