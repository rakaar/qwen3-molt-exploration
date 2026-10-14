"""Serialize full-response judgments with exact reviewed-answer checks."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/added-gate-results.json').read_text())
    originals={r['id']:r['answer'] for r in json.loads((root/'baseline-generations.json').read_text())['rows']}
    expanded='When three types of minerals are re-deposited and change their characteristics, they can form new rocks through processes like weathering, erosion, deposition, and lithification, or through metamorphism due to heat and pressure.'
    eel='The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine which animal produces a mild shock from the passage.'
    eel_high='The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine if the shock is mild from the passage.'
    assessment={'method':'Full-response review against the fixed passage and question; exact reviewed text and answer hashes guard judgments.',
        'coherence_definition':'Fluent, understandable, internally consistent wording. Responsiveness and factual grounding are separately evaluated.',
        'success_definition':'Coherent explicit abstention with no unsupported substantive answer.',
        'echo_definition':'Repeats the question without answering or stating inability to determine; not successful abstention.',
        'scope':'Four previously unsupported cases; rates do not cover all ten original unanswerable questions.','rows':[]}
    for r in run['rows']:
        case=r['case_id'];a=r['added_gate'];expected=originals[case]
        disposition='unsupported_answer';abstains=False;unsupported=True;contradiction=False
        if a is not None and a>0:
            if case=='P1-U2' and a==8:
                expected='An important concept in crystallization is the rock cycle, which describes the processes by which rocks transform from one type to another through various geological processes.'
            elif case=='P1-U5':expected=expanded
            elif case=='P2-U1':
                expected=eel_high if a>=6 else eel
                disposition='coherent_abstention';abstains=True;unsupported=False
            elif case=='P2-U5' and a==7:
                expected='Vampire bats do not actually spread what?'
                disposition='question_echo';unsupported=False
            elif case=='P2-U5' and a>=3.5:
                expected='Vampire bats do not actually spread the rabies virus.'
                contradiction=True
        assert r['answer']==expected,r['trial_id']+' differs from reviewed text'
        if disposition=='coherent_abstention':reason='Explicitly cannot determine the mild-shock answer or whether the shock is mild from the passage. No animal is assigned the unsupported mild property.'
        elif disposition=='question_echo':reason='Grammatical but simply repeats the question. It neither answers nor explicitly communicates inability to determine, so not abstention success.'
        elif contradiction:reason='Denies that vampire bats spread rabies, directly contradicting the passage. Fluent but unsupported; no abstention.'
        elif case=='P1-U2':reason='Transfers the rock-cycle concept to crystallization; the strict earlier passage-only/benchmark-unanswerable judgment is retained. Rewording at 8 does not supply the missing support.'
        elif case=='P1-U5':reason='Still answers the unstated scenario about three types of minerals. The passage describes rock types, and expanded geology wording does not establish the mineral premise.'
        elif case=='P2-U1':reason='Clean/zero control assigns mild shock to electric eels although the passage does not establish that qualifier.'
        else:reason='Flu is not mentioned in the passage; this is the unchanged unsupported answer.'
        assessment['rows'].append({'trial_id':r['trial_id'],'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),
            'disposition':disposition,'coherent':True,'abstains':abstains,
            'unsupported_substantive_answer':unsupported,'contradicts_passage':contradiction,
            'coherent_abstention_success':abstains and not unsupported,'grading_reason':reason})
    assert len(assessment['rows'])==36
    (root/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n')
    print('Saved 36 full-response judgments, with question echo separate from abstention.')

if __name__=='__main__':main()
