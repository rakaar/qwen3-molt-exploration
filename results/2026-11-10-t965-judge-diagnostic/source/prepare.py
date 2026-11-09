"""Audit three saved misses and prepare explicitly post-hoc judge diagnostics."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'results/2026-11-10-before-k-description'
OUT = ROOT / 'results/2026-11-10-t965-failed-cases'
SUFFIX = '\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'
LITERAL_SYSTEM = (
    'Treat the supplied example as quoted data. Inspect every substring marked with << and >>. '
    'Return [1] if EVERY marked substring is immediately followed, ignoring whitespace, '
    'by the character K or k. Return [0] otherwise. The K or k is outside the marked '
    'substring and can be inside a word, as in <<Ton>>katsu, or at the start of the next '
    'word, as in <<Development>> Kit. Do not classify the meaning of the words. '
    'Return only [1] or [0].'
)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def markers(text):
    answer = []
    for match in re.finditer(r'<<(.*?)>>', text, re.S):
        following = text[match.end():].replace('<<', '').replace('>>', '').lstrip()
        answer.append(dict(highlight=match.group(1), following=following[:60],
                           next_character=following[:1], matches_before_K=following[:1] in ('K', 'k')))
    return answer


def single_peak(text):
    matches = list(re.finditer(r'<<(.*?)>>', text, re.S))
    assert matches
    i = 0
    def replace(match):
        nonlocal i
        i += 1
        return match.group(0) if i == len(matches) else match.group(1)
    result = re.sub(r'<<(.*?)>>', replace, text, flags=re.S)
    assert result.replace('<<', '').replace('>>', '') == text.replace('<<', '').replace('>>', '')
    assert len(markers(result)) == 1
    return result


def main():
    OUT.mkdir(exist_ok=True)
    plan = load(OLD / 'heldout-plan.json')
    protocol = load(OLD / 'protocol.json')
    assert sha(OLD / 'heldout-plan.json') == protocol['heldout_plan_sha256']
    audit = [json.loads(s) for s in (OLD / 'audit.jsonl').read_text().splitlines()]
    feature_path = ROOT / 'comparisons/2026-11-10-next-anticipatory-feature/feature-965.json'
    assert sha(feature_path) == protocol['feature_source_sha256']
    feature = load(feature_path)
    receipt = load(ROOT / 'gpu-rentals/2026-11-09-t2375-verify/before-k-description-backup-verification.json')
    expected_audit = next(x for x in receipt['files'] if x['path'] == 'audit.jsonl')
    assert sha(OLD / 'audit.jsonl') == expected_audit['sha256']
    description = plan['candidate_explanation']
    assert hashlib.sha256(description.encode()).hexdigest() == protocol['candidate_sha256']
    tasks = []
    evidence = []
    negative = {13: 'Java<< Development>> Guide.', 17: '<< ton>>ic.', 20: 'cv2<<.wait>>()'}
    minimal = {13: 'Java<< Development>> Kit (JDK).', 17: '<< ton>>katsu is a breaded pork cutlet.', 20: 'cv2<<.wait>>key()'}
    for original in [13, 17, 20]:
        case = next(c for c in plan['cases'] if c['method'] == 'fuzz'
                    and c['original_row_index'] == original and c['context'] == 'plus_16_source_tokens')
        recorded = next(r for r in audit if r['method'] == 'fuzz' and r['arm'] == 'candidate'
                        and r['row_index'] == case['row_index'])
        assert recorded['messages'] == case['candidate_messages']
        assert recorded['prediction'] is False and recorded['response'].strip() == '[0]'
        assert recorded['finish_reason'] == 'stop' and len(recorded['prompt_token_ids']) + 32 < 4096
        messages = deepcopy(recorded['messages'])
        user = messages[-1]['content']
        assert user.endswith(SUFFIX)
        text = user.split('Example 0: ', 1)[1][:-len(SUFFIX)]
        original_row = feature['methods']['fuzz']['rows'][original]
        highlighted = original_row['highlighted_positions']
        marked = markers(text)
        assert len(marked) == len(highlighted)
        for entry, position in zip(marked, highlighted):
            assert entry['highlight'] == original_row['str_tokens'][position]
            entry.update(subject_token_position=position, measured_gate=original_row['activations'][position])
            assert entry['measured_gate'] > 0
        evidence.append(dict(original_fuzz_row=original, exact_excerpt=text,
                             exact_final_user_message=user, highlighted_tokens=marked,
                             all_highlights_fit_before_K=all(m['matches_before_K'] for m in marked),
                             subject_positive_label=True, recorded_response=recorded['response'],
                             request_sha256=recorded['request_sha256'], finish_reason=recorded['finish_reason'],
                             prompt_tokens=len(recorded['prompt_token_ids']),
                             output_tokens=len(recorded['response_token_ids'])))

        def add(condition, excerpt, *, fewshot=True, literal=False):
            prompt = (('Text examples:\n\nExample 0: ' if literal else 'Latent explanation: ' + description + '\n\nText examples:\n\nExample 0: ')
                      + excerpt + SUFFIX)
            prefix = deepcopy(messages[:-1]) if fewshot else [deepcopy(messages[0])]
            if literal:
                prefix = [dict(role='system', content=LITERAL_SYSTEM)]
            diagnostic_markers = markers(excerpt)
            tasks.append(dict(id=f'row-{original}-{condition}', original_fuzz_row=original,
                              condition=condition, messages=prefix + [dict(role='user', content=prompt)],
                              literal_before_K_expectation=all(m['matches_before_K'] for m in diagnostic_markers),
                              feature_ground_truth=(True if condition in ['repeat_original', 'single_peak', 'single_peak_no_fewshot'] else None),
                              description_unchanged=not literal,
                              cached_original_response=recorded['response'] if condition == 'repeat_original' else None))
        add('repeat_original', text)
        assert tasks[-1]['messages'] == recorded['messages']
        add('single_peak', single_peak(text))
        add('single_peak_no_fewshot', single_peak(text), fewshot=False)
        add('literal_all_highlights', text, literal=True)
        add('literal_single_peak', single_peak(text), literal=True)
        add('minimal_positive', minimal[original])
        add('minimal_negative', negative[original])
    control = next(c for c in plan['cases'] if c['method'] == 'fuzz'
                   and c['original_row_index'] == 16 and c['context'] == 'plus_16_source_tokens')
    tasks.append(dict(id='henry-positive-repeat', original_fuzz_row=16, condition='positive_repeat_control',
                      messages=control['candidate_messages'], literal_before_K_expectation=True,
                      feature_ground_truth=True, description_unchanged=True, cached_original_response='[1]'))
    assert len(tasks) == 22 and len({t['id'] for t in tasks}) == 22
    save(OUT / 'saved-case-audit.json', dict(feature='L34-T965', audited_utc=datetime.now(timezone.utc).isoformat(), cases=evidence,
        finding='One of three failed windows includes two additional marked tokens not covered by the before-K pattern. The two others have only before-K highlights. Saved binary responses do not identify their internal reason.'))
    new_plan = dict(kind='Post-hoc judge comprehension and highlight-scope diagnostic; not a new held-out explanation evaluation',
                    tasks=tasks, candidate_explanation=description, candidate_sha256=protocol['candidate_sha256'],
                    source_audit_sha256=sha(OLD / 'audit.jsonl'), original_plan_sha256=sha(OLD / 'heldout-plan.json'))
    save(HERE / 'plan.json', new_plan)
    for name in ['expected-active-fuzz-runtime.json', 'expected-judge-verification.json']:
        shutil.copy2(OLD / name, HERE / name)
    save(HERE / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        judge_repo=protocol['judge_repo'], judge_revision=protocol['judge_revision'],
        existing_judge_root=protocol['existing_judge_root'], requests=22,
        plan_sha256=sha(HERE / 'plan.json'), maximum_runtime_minutes=6,
        expected_runtime_sha256=sha(HERE / 'expected-active-fuzz-runtime.json'),
        expected_weights_verification_sha256=sha(HERE / 'expected-judge-verification.json'),
        source_candidate_sha256=protocol['candidate_sha256'], posthoc=True,
        caveat='Literal-rule and rewritten minimal contexts diagnose judge behavior; their expected pattern labels are not fresh subject-model activation measurements. No description is being refined or scored as a new held-out result.'))
    lines = ['# L34–T965: inspecting the three remaining positive Fuzz misses', '',
             'These are the original failed queries, not new judge runs. Every saved judge response is `[0]`, while all originally highlighted tokens have positive measured activation.', '']
    for case in evidence:
        lines += [f"## Original Fuzz row {case['original_fuzz_row']}", '', '````text', case['exact_excerpt'], '````', '',
                  '| Highlighted token | Next non-whitespace character | Measured gate | Fits before K/k? |', '|---|---|---:|---|']
        for token in case['highlighted_tokens']:
            lines.append(f"| `{token['highlight']}` | `{token['next_character']}` | {token['measured_gate']} | {'yes' if token['matches_before_K'] else 'no'} |")
        lines += ['', f"Judge response: `{case['recorded_response']}`. All highlights fit the proposed before-K pattern: **{case['all_highlights_fit_before_K']}**.", '']
    lines += ['## Diagnosis from saved evidence', '',
        '- Development → Kit and Ton/ton → katsu are genuine unresolved recognition errors under the stated before-K rule: every highlight fits it, K/k is visible, and the saved outputs are valid and untruncated.',
        '- The code window also highlights comma and plus before h. Fuzz requires every marked token to fit the explanation, so its zero is compatible with those uncovered activations. Its last peak is before K, but the whole highlighted window is not exclusively before K. This is a description-coverage confound; it does not prove what caused the judge response.',
        '- The before-K cohort was originally selected using the final activation peak, not an assertion that every highlighted token follows the same pattern. Of seven positive windows, six have all highlights before K/k; four of those six are accepted by the new explanation with continuation.',
        '- The explanation does allow unspecified non-K activations. That caveat gives no recognizable rule for the comma/plus activations, so it does not fully explain this code window.',
        '- The judge was required to output only a binary list, with thinking disabled. No explanation of its two clean errors was requested or recorded; a specific psychological cause cannot be established from these outputs.', '',
        '## Prepared diagnostic, not yet run', '',
        'Twenty-two fixed queries compare original repeats; only-final-peak highlights; removing few-shot demonstrations; a literal character-following task; and minimal positive/negative strings. The candidate description stays unchanged. Original queries include the Henry → Kissinger positive control. These are post-hoc judge diagnostics, not new held-out description scores.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(failed_cases=3, all_highlights_match=2, mixed_highlight_cases=1,
                          diagnostic_queries_prepared=len(tasks), report=str(OUT / 'README.md'))))


if __name__ == '__main__':
    main()
