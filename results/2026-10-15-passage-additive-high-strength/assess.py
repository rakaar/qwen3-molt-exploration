"""Serialize manually reviewed complete responses; reject text drift."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/added-gate-results.json').read_text())
    baseline={r['id']:r['answer'] for r in json.loads((root/'baseline-generations.json').read_text())['rows']}
    expected={
      ('P1-U2',8):'An important concept in crystallization is the rock cycle, which describes the processes by which rocks transform from one type to another through various geological processes.',
      ('P1-U2',16):'An important concept in crystallization is the rock cycle, which describes how rocks transform and change over time through various geological processes.',
      ('P1-U2',24):'An important concept in crystallization is not mentioned in the passage. The passage does not explain what is important about crystallization.',
      ('P1-U2',32):'An important concept in crystallization is not mentioned in the passage. The passage does not explain what is important in crystallization.',
      ('P1-U5',8):'When three types of minerals are re-deposited and change their characteristics, they can form new rocks through processes like weathering, erosion, deposition, and lithification, or through metamorphism due to heat and pressure.',
      ('P1-U5',16):'When three types of minerals are re-deposited and change their characteristics, the passage does not provide specific information about the outcome. Therefore, I cannot determine the answer from the passage.',
      ('P1-U5',24):'When three types of minerals are re-deposited and change their characteristics, the passage does not provide specific information about the minerals. However, it cannot be determined from the given passage.',
      ('P1-U5',32):'When three types of minerals are re-deposited and change their characteristics, the passage does not provide specific information about the minerals being re-deposited. The passage does not mention the three types of minerals. The passage only discusses the process of re-deposition and does not provide enough information to determine the answer.',
      ('P2-U5',8):'Vampire bats do not actually spread the rabies virus.',
      ('P2-U5',16):'Vampire bats do not actually spread rabies.',
      ('P2-U5',24):'Vampire bats do not actually spread rabies. However, the passage does not provide information about what vampire bats spread, and the question cannot be answered based on the given text.',
      ('P2-U5',32):'Vampire bats do not spread rabies. However, the provided passage does not provide information about what they spread. The answer is not clear. (This is a false conclusion, as it cannot be determined from the given text).'
    }
    assessment={'method':'Manual complete-response review against the fixed passages, guarded by exact reviewed text and SHA256.',
      'coherence_definition':'Separate linguistic fluency/intelligibility from factual consistency and answer/abstention mixture. All responses are fluent; this does not imply consistent or correct content.',
      'success_definition':'Fluent explicit abstention without an unsupported substantive answer to the question. Explanation grounding is scored separately.',
      'scope':'Only the three previously non-abstaining cases; the electric-eel question is not retested here.','rows':[]}
    for r in run['rows']:
        case=r['case_id'];a=r['added_gate'];positive=a is not None and a>0
        text=expected[(case,a)] if positive else baseline[case]
        assert r['answer']==text,r['trial_id']+' differs from reviewed text'
        abstains=positive and ((case=='P1-U2' and a>=24) or (case=='P1-U5' and a>=16) or (case=='P2-U5' and a>=24))
        unsupported=not abstains or case=='P2-U5'
        success=abstains and not unsupported
        mixed=abstains and unsupported
        disposition='coherent_abstention' if success else 'mixed_answer_abstention' if mixed else 'unsupported_answer'
        contradiction=positive and case=='P2-U5'
        explanation_issue=positive and ((case=='P1-U5' and a==32) or (case=='P2-U5' and a>=24))
        if case=='P1-U2':
            reason='Explicitly says the crystallization answer is not provided; no unsupported target answer.' if success else 'Still supplies rock cycle as the crystallization answer. Retain the earlier strict passage-only benchmark judgment.'
        elif case=='P1-U5':
            reason='Explicitly says the requested minerals outcome cannot be determined from the passage; no substantive target answer.' if success else 'Still supplies an outcome for the unsupported three-types-of-minerals premise; rock types are not three mineral types.'
            if a==24:reason+=' Wording is awkward but understandable.'
            if a==32:reason+=' The claim that the passage only discusses re-deposition overstates its scope; abstention occurs but its explanation is not fully grounded.'
        elif not positive:reason='Clean/zero control: flu is not identified as the answer in the passage; unchanged unsupported answer.'
        elif mixed:reason='Mixes a false rabies denial with uncertainty/abstention language. The passage explicitly says vampire bats can spread rabies; the claim that it supplies no spread information is also false. Not a clean abstention.'
        else:reason='Denies that vampire bats spread rabies, directly contradicting the passage; no abstention.'
        if case=='P2-U5' and a==32:reason+=' The final parenthetical self-correction has an unclear referent and does not cleanly retract or resolve the conflicting claims.'
        assessment['rows'].append({'trial_id':r['trial_id'],'answer_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'disposition':disposition,'fluent_and_understandable':True,'abstains':abstains,
            'unsupported_substantive_answer':unsupported,'contradicts_passage':contradiction,
            'unsupported_explanation':explanation_issue,'mixed_answer_and_abstention':mixed,
            'coherent_abstention_success':success,'grading_reason':reason})
    assert len(assessment['rows'])==18
    (root/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n')
    print('Saved 18 reviewed judgments; mixed answers excluded from successful abstention.')

if __name__=='__main__':main()
