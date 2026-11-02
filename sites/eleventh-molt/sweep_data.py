"""Package the two audited gate-transfer sweeps without discarding responses."""
import hashlib
import json
from pathlib import Path
import shutil


def build_sweeps(root, out):
    result={}
    for kind in ['scalar','full']:
        folder=root/f'results/2026-11-03-eleven-to-five-{kind}'
        source=folder/'summary.json'
        data=json.loads(source.read_text())
        assert data['validation']['status']=='pass'
        assert data['validation']['no_generated_token_edits']
        assert data['validation']['bf16_edits_exact']
        assert data['protocol']['recipient_prompt']=='the fifth month of the year is'
        assert len(data['runs'])==17

        def compact(row):
            if row.get('target_gate') is not None:
                assert row['selected_input_positions']==[6]
                assert row['generated_positions_edited']==[]
            keep=['target_gate','response','initial_word','initial_answer_november','stop_reason',
                  'p_may_exact','p_november_exact','p_may_case_sum','p_november_case_sum',
                  'intended_delta_l2','actual_delta_l2','operator_source','matched_donor_gate']
            record={k:row[k] for k in keep if k in row}
            if row.get('initial_answer_november'):
                assert "no, that's not true" in row['response'] and 'month of the year is may' in row['response']
            return record

        result[kind]={'protocol':data['protocol'],'native':compact(data['native']),
                      'runs':[compact(r) for r in data['runs']],
                      'initial_november_gates':data['november_initial_answer_gates'],
                      'validation':data['validation'],'source':str(source.relative_to(root)),
                      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'runtime':{k:data['runtime'][k] for k in ['base_revision','molt_revision','dtype','attention','tf32']}}
        if kind=='full':
            result[kind]['norm_matched_controls']=[compact(r) for r in data['norm_matched_controls']]
            result[kind]['operator_norm_ratio']=data['operator_norm_ratio']
        else:
            result[kind]['independent_repetition']=data['independent_repetition']
        shutil.copy2(source,out/f'{kind}-sweep.json')
        shutil.copy2(folder/'sweep.csv',out/f'{kind}-sweep.csv')
        image='probability-sweep' if kind=='scalar' else 'operator-comparison'
        for ext in ['png','svg']:
            shutil.copy2(folder/f'{image}.{ext}',out/f'{kind}-sweep-plot.{ext}')
    assert result['scalar']['protocol']['fixed_gates']==result['full']['protocol']['fixed_gates']
    assert result['scalar']['runtime']==result['full']['runtime']
    assert result['full']['validation']['natural_patch_equals_saved_donor_full_contribution_exactly']
    assert result['full']['validation']['scalar_copy_reproduces_previous_experiment_exactly']
    full=json.loads((root/'results/2026-11-03-eleven-to-five-full/summary.json').read_text())
    assert full['prior_scalar_summary_sha256']==result['scalar']['source_sha256']
    (out/'transplant-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result
