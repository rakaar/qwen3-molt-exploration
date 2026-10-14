"""Validate and serialize manually reviewed judgments; no keyword-based grading."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/added-gate-results.json').read_text())
    groups=json.loads((root/'baseline-groups.json').read_text())
    reviewed=json.loads((root/'reviewed-judgments.json').read_text())
    group_map={r['case_id']:r for r in groups['rows']}
    judgments={r['trial_id']:r for r in reviewed['rows']}
    assert run['status']=='complete' and len(run['rows'])==len(judgments)==70
    assert hashlib.sha256((root/groups['baseline_reference_file']).read_bytes()).hexdigest()==groups['baseline_sha256']
    rows=[]
    for r in run['rows']:
        g=judgments[r['trial_id']]
        assert hashlib.sha256(r['answer'].encode()).hexdigest()==g['answer_sha256'],r['trial_id']
        if r['added_gate'] is None:assert g['answer_sha256']==group_map[r['case_id']]['baseline_answer_sha256']
        withholding=g['coherent'] and not g['unsupported_substantive_answer'] and (g['abstains'] or g['supported_premise_correction'])
        mixed=g['abstains'] is True and g['unsupported_substantive_answer'] is True
        decision=g['answer_assessable'] and g['unsupported_substantive_answer'] is False and (g['abstains'] is True or g['supported_premise_correction'])
        rows.append({**g,'baseline_group':group_map[r['case_id']]['group'],'withholding_decision':decision,
            'coherent_withholding':withholding,'fully_grounded_withholding':withholding and not g['explanation_error'],
            'mixed_answer_abstention':mixed})
    out={k:v for k,v in reviewed.items() if k!='rows'}
    out.update(scope='Ten original unanswerable questions; baseline groups fixed at four unsupported answers and six clean withholdings.',
        withholding_definition='Coherent abstention or valid supported correction of a false premise, without an unsupported substantive target answer. Explanation errors are tracked separately.',
        withholding_decision_definition='An assessable complete response withholds the requested answer and gives no unsupported target answer. Coherence is scored independently; repetitive but still readable abstentions can have a withholding decision and fail coherence.',
        unsupported_definition='Any substantive answer unsupported by the supplied passage, including one accompanied by uncertainty or abstention language.',
        coherence_definition='Understandable, coherent response without substantial nonsequitur, broken syntax, or repetitive collapse. Mild grammatical errors and factual explanation errors are separate flags.',rows=rows)
    (root/'assessment.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('Verified all 70 manually reviewed answer hashes and fixed baseline groups.')

if __name__=='__main__':main()
