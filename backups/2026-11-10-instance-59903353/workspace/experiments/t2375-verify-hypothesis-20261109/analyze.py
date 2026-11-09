"""Summarize every frozen probe without assigning post-hoc score labels."""
import json
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'results/2026-11-09-t2375-verify-hypothesis'
data=json.loads((OUT/'results.json').read_text())
assert data['status']=='complete'
precision={r['id']:r for r in json.loads((OUT/'precision-diagnostic.json').read_text())}
groups=defaultdict(list)
for c in data['cases']:groups[c['group']].append(c)
summary=dict(cases=len(data['cases']),mcp_calls=len(data['calls']),intervention=None,
    groups={group:dict(total=len(cases),any_active=sum(c['is_activated'] for c in cases),
                       final_token_active=sum(c['final_token_gate']>0 for c in cases)) for group,cases in groups.items()},
    precision_mask_disagreements=data['precision_mask_disagreements'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

def text(s):return str(s).replace('|','\\|').replace('\n',' ↵ ')
def number(value):return f'{0.0 if value == 0 else value:.5g}'
def activated(c):
    return '; '.join(f"{r['position']}: `{text(r['token'])}` = {r['gate']:.5g}" for r in c['active_tokens']) or 'none'
lines=['# L29–T2375: testing the revised water/eleven description','',
       'The actual upstream `verify_hypothesis` MCP tool was run on 51 frozen self-authored probes. This is an activation check, not a Detection/Fuzz score, a held-out benchmark, or an intervention. No responses were generated.','',
       '## Findings','',
       '- Five of six raw water probes activated; all six matched non-water controls stayed off. Isolated `water` did not activate.',
       '- In `Water is essential for life.`, the active token was ` is`, not `Water`. Water-context activation is not always literal water-token activation.',
       '- All six raw eleventh-month/alphabet/Roman prompts (word and digit forms) activated on the final ` is`; the corresponding plain positive-integer prompts stayed off.',
       '- Eleven produced the strongest gate within each matched raw month/alphabet/Roman ordinal triplet, but tenth and twelfth also often activated. This is a context-sensitive preference, not an exclusive eleven detector.',
       '- All nine chat month/alphabet/Roman probes activated somewhere; many active positions were template newlines, and the strongest position was not always the final input token.',
       '- All 51 BF16/FP32 activation masks agreed. This does not imply numerically identical gate values or identical results under a different model runtime.','',
       '## Frozen description','',data['protocol']['description'],'',
       'The description was used to design the probe texts. It was not passed to an LLM judge: `verify_hypothesis` accepts a feature ID, texts, and raw/chat formatting.','',
       '## Scope and counts','',
       '| Probe group | Prompts | Gate active anywhere | Active at final input token |',
       '|---|---:|---:|---:|']
for group,rows in summary['groups'].items():lines.append(f"| {group} | {rows['total']} | {rows['any_active']} | {rows['final_token_active']} |")
lines += ['', '## Raw ordinal comparisons','',
          'Each cell reports maximum gate / final-input-token gate. Neighboring ordinals were not assumed to be true negatives.','',
          '| Domain | Wording | Tenth | Eleventh | Twelfth |','|---|---|---:|---:|---:|']
for domain in ['month','alphabet','roman','integer']:
    for wording in ['word','digit']:
        cells=[]
        for n in [10,11,12]:
            c=next(c for c in groups['ordinal'] if c['domain']==domain and c['ordinal']==n and c['wording']==wording)
            cells.append(f"{number(c['max_activation'])} / {number(c['final_token_gate'])}")
        lines.append('| '+' | '.join([domain,wording,*cells])+' |')
lines += ['', '## Exact probes and active tokens','',
          'Positions are zero-based in the complete input passed to Qwen3, including chat template tokens. Token IDs, zero activations and full aligned traces are preserved in `results.json`. Special/control token strings use the tool’s escaped display form.','',
          '| ID | Format | Input text | Max gate | Final gate | All active input tokens (position, text, gate) |',
          '|---|---|---|---:|---:|---|']
for c in data['cases']:
    lines.append(f"| {c['id']} | {c['prompt_format']} | {text(c['prompt'])} | {number(c['max_activation'])} | {number(c['final_token_gate'])} | {activated(c)} |")
lines += ['', '## Measurement and provenance','',
 '- Native Qwen3-4B, BF16, SDPA attention; the native network is not replaced with MOLTs.',
 '- Gates are measured from the input to L29 post-attention RMSNorm, matching the dashboard’s `pre_norm` training location.',
 '- The original `SteeringEngine.inspect`, `MoltGates`, and MCP `verify_hypothesis` wrapper/backing method are unmodified.',
 '- A small session initializer permits only T2375 self-authored probes; it does not fabricate training windows or load a corpus.',
 '- A derived gate-only checkpoint contains every exact encoder row, threshold, and input-standardization tensor from the pinned release. The T2375 row and standardization match the earlier verified feature slice exactly.',
 '- The tool uses its original BF16 autocast for gate computation. The supplemental FP32 diagnostic preserves native BF16 model inputs and reports threshold-mask disagreements.',
 f"- FP32/BF16 activation-mask disagreements: {summary['precision_mask_disagreements']} of 51 prompts. Inspect `precision-diagnostic.json` before comparing values with earlier FP32 research runs.",
 f"- Model revision: `{data['protocol']['base_revision']}`.",
 f"- MOLT revision: `{data['protocol']['molt_revision']}`.",
 f"- Runtime: torch `{data['runtime']['torch']}`, transformers `{data['runtime']['transformers']}`.",
 '- Upstream source: https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/molt_agentic_mcp.py . Local source hashes and the tool schema are preserved.',
 '- These hand-authored probes assess the scope of the explanation. They do not establish general predictive accuracy or a causal mechanism.','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps(summary))
