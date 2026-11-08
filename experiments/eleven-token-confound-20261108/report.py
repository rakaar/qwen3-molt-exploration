"""Render the measured token-confound comparison and its full native evidence."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / 'results/2026-11-08-eleven-token-confound'
initial = json.loads((OUT / 'initial/summary.json').read_text())
followup = json.loads((OUT / 'clear-slots/summary.json').read_text())
old = next(m for m in initial['final_token_matrices'] if m['variant'] == 'original' and m['ordinal'] == 11)
new = next(m for m in followup['final_token_matrices'] if m['ordinal'] == 11)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fafaf7'})


def matrix_figure(kind, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    low = .95 if kind == 'contribution' else 0
    for ax, item, label in zip(axes, [old, new], ['Original: common final token', 'Follow-up: three distinct final tokens']):
        values = np.array(item['cosines'][kind], dtype=float)
        im = ax.imshow(values, vmin=low, vmax=1, cmap='Blues')
        labels = [domain.title() + '\n' + repr(token['token']) for domain, token in zip(item['domains'], item['tokens'])]
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), [x.title() for x in item['domains']])
        ax.set_title(label, pad=14, fontsize=12)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center',
                        color='white' if values[i,j] > (low + .65 * (1-low)) else '#112536')
        ax.tick_params(length=0)
    fig.suptitle(title, fontsize=15, y=.98)
    fig.subplots_adjust(left=.10, right=.89, top=.84, bottom=.23, wspace=.38)
    fig.colorbar(im, cax=fig.add_axes([.92, .26, .018, .56]), label='Cosine similarity')
    fig.text(.5, .07, 'Eleventh month / letter / Roman integer; final input positions.\n'
             'All six initial answers are correct. No intervention, final RMSNorm or unembedding.',
             ha='center', fontsize=10, color='#42535a')
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


matrix_figure('contribution', OUT / 'different-tokens-cosines.png',
              'Full MOLT contributions remain aligned at different final tokens')
matrix_figure('residual', OUT / 'residual-cosines.png',
              'Native residual streams are less aligned than the MOLT contributions')

eleven = [c for c in followup['cases'] if c['ordinal'] == 11]
maximum = max(t['gate'] for c in eleven for t in c['active_input_tokens'])
cmap, norm = plt.get_cmap('YlOrBr'), Normalize(0, maximum)
fig, axes = plt.subplots(3, 1, figsize=(14, 5.2))
for ax, case in zip(axes, eleven):
    # Read the complete trace, including zeros, from the original capture.
    source = ROOT / 'comparisons/2026-11-08-eleven-token-confound/followup-clear-slots/remote-export/artifacts/results.json'
    raw = next(c for c in json.loads(source.read_text())['cases'] if c['id'] == case['id'])
    n = len(raw['trace'])
    for t in raw['trace']:
        i, gate = t['position'], t['gate']
        face = cmap(norm(gate)) if gate else '#efefea'
        ax.add_patch(plt.Rectangle((i, 0), .96, 1, facecolor=face,
                                  edgecolor='#236eb2' if i == n-1 else 'white', linewidth=2 if i == n-1 else 1))
        token = t['token'].replace(' ', '·').replace('\n', '↵')
        color = 'white' if gate > maximum * .65 else '#202d33'
        ax.text(i+.48, .64, token, ha='center', va='center', fontsize=9, color=color)
        ax.text(i+.48, .27, f'{i}: {gate:.3f}', ha='center', va='center', fontsize=8, color=color)
    ax.set_xlim(-.12, 12.15)
    ax.set_ylim(-.08, 1.75)
    ax.text(0, 1.40, f"{case['domain'].title()} → {case['expected_answer']}", fontsize=11, weight='bold')
    ax.axis('off')
fig.suptitle('Natural L29–T2375 activation at every input token', fontsize=15)
fig.text(.5, .025, 'Each cell: token, then zero-based position and gate. Blue outline: compared final input position. No token is edited.', ha='center', fontsize=10)
fig.subplots_adjust(left=.035, right=.99, top=.90, bottom=.09, hspace=.20)
fig.savefig(OUT / 'rewritten-input-gates.png', dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in r) + ' |\n' for r in rows)


text = '''# T2375 contribution alignment with different final input tokens

8 November 2026 · Qwen3-4B · natural activations only

The shared final ` is` token is not sufficient to explain the original high
cross-domain contribution cosine. In the clearer follow-up, the final tokens are
` called`, ` letter` and ` as`, all with different IDs. All three initial answers
are correct, and the pairwise full-contribution cosines remain 0.983179–0.988280.
This is observational geometry, not a new ablation or proof of an eleven-only feature.

![Original and rewritten contribution matrices](different-tokens-cosines.png)

## Exact rewritten eleventh prompts

'''
text += table(['Domain', 'Raw prompt', 'Final token / ID', 'Position', 'Gate', 'Contribution L2', 'Initial answer'], [
    [c['domain'], '`'+c['prompt']+'`', repr(c['final_input']['token'])+' / '+str(c['final_input']['token_id']),
     c['final_input']['position'], f"{c['final_input']['gate']:.6f}", f"{c['final_input']['contribution_norm']:.6f}", c['expected_answer']]
    for c in eleven])
text += '\n## Where the gate activates\n\n![Complete input traces](rewritten-input-gates.png)\n\n'
text += 'The month gate is active at ` is` and ` called`; alphabet at ` letter`; Roman at ` is` and ` as`. The fixed final positions, rather than peak-cosine tokens, supply the main matrix. No generated-token activation is used.\n\n'
text += table(['', 'Month: called', 'Alphabet: letter', 'Roman: as'], [
    [domain.title()] + [f'{v:.6f}' for v in row]
    for domain, row in zip(new['domains'], new['cosines']['contribution'])])
text += '\nThe corresponding residual-stream cosines are 0.604279–0.676747. Operator and contribution cosine are equal for positive scalar gates; their magnitudes differ.\n\n![Residual stream matrices](residual-cosines.png)\n\n'
text += '''## Scope and adaptive follow-up

The initial frozen pilot used 27 prompts: three domains × three phrasings ×
three ordinals (10, 11, 12). Some rewrites were ambiguous: month `named` prompted
etymology, alphabet `symbol` prompted chemical-element continuations, and Roman
`representation` often prompted clarification. These failures are retained.

After inspecting responses and gate traces, before calculating follow-up cosines,
we added a clearly labelled nine-run follow-up with endings called/letter/as.
Six prompts were new; three alphabet prompts were exact repeats. Across both
phases there are 36 runs covering 33 distinct prompts, with 369 captured input
positions. The clearer triplet was chosen for answer-slot clarity, not by cosine.

## Matched ordinal controls

Gate activity and direction are not exclusive to eleven. Correctly answered
Roman tenth/twelfth controls also activate at ` as`; original alphabet J/L and
Roman X/XII controls also activate at ` is`. The original tenth-month prompt
incorrectly answers November; the rewritten tenth-month prompt begins with a
clarification and later gives October with a contradictory explanation. Both are
flagged as failed/ambiguous capability controls rather than clean successes.

'''
text += table(['Domain', 'Ordinal', 'Final token', 'Gate', 'Native capability', 'Native response excerpt'], [
    [c['domain'], c['ordinal'], repr(c['final_input']['token']), f"{c['final_input']['gate']:.6f}",
     str(c['native_answer_review']['correct']), c['native']['response'][:100]+'…'] for c in followup['cases']])
text += '''
## Verification and measurements

- Native model and observer runs have identical generated IDs and every measured next-token logit hash, for all 36 runs.
- The three original eleventh anchors reproduce the saved native residual arrays and full first-token logits exactly.
- Independent CPU checks reproduce normalization, gate membership, raw scores, operators and contributions in both phases.
- Both remote archives were downloaded and every manifest-listed file was SHA-256 verified locally.
- No gate or MLP contribution was changed. There is no final RMSNorm or unembedding in the cosine calculation.
- All active cross-domain input-token pairs are retained, separated by equal/different token IDs. A zero contribution has undefined cosine.

## All native responses

Correctness refers to the initial target answer; it does not certify a coherent
32-token continuation. `null` means ambiguous initial-answer capability.

'''
for phase, summary in [('Initial frozen pilot', initial), ('Adaptive clear-slot follow-up', followup)]:
    text += '\n### '+phase+'\n\n'
    for c in summary['cases']:
        text += '\n#### '+c['id']+'\n\nPrompt: `'+c['prompt']+'`\n\n'
        text += 'Expected: '+c['expected_answer']+'. Native initial-answer review: '+str(c['native_answer_review']['correct'])+'. '+c['native_answer_review']['note']+'\n\n'
        text += '```text\n'+c['native']['response']+'\n```\n\n'
        text += 'Active input positions: '+', '.join(f"{t['position']} {t['token']!r}: {t['gate']:.6f}" for t in c['active_input_tokens'])+'.\n'
text += '''
## Source and data

- Source: `experiments/eleven-token-confound-20261108/`, including both frozen protocols and manual native-answer review.
- [Initial summary](initial/summary.json) · [Follow-up summary](clear-slots/summary.json)
- [Initial all-input traces](initial/all-input-token-traces.csv) · [Follow-up all-input traces](clear-slots/all-input-token-traces.csv)
- [Initial active-token pairs](initial/all-active-eleven-token-pairs.csv) · [Follow-up active-token pairs](clear-slots/all-active-eleven-token-pairs.csv)
- [Initial ordinal controls](initial/matched-ordinal-controls.csv) · [Follow-up ordinal controls](clear-slots/matched-ordinal-controls.csv)

The model and feature revisions match the earlier experiment. Full contributions
are computed in native MLP-output units using the checkpoint input/output
standardizers. Cosines are evaluated in FP64 from saved FP32 contribution vectors.
Shared task context and a stable output direction of this particular low-rank
operator remain possible explanations; the exact final-token identity is only one
confound addressed here. No new claim of language-independent abstraction is made.
'''
(OUT / 'README.md').write_text(text)
print(json.dumps({'report': str(OUT/'README.md'), 'figures': ['different-tokens-cosines.png', 'residual-cosines.png', 'rewritten-input-gates.png']}))
