# L29–T2375: testing the revised water/eleven description

The actual upstream `verify_hypothesis` MCP tool was run on 51 frozen self-authored probes. This is an activation check, not a Detection/Fuzz score, a held-out benchmark, or an intervention. No responses were generated.

## Findings

- Five of six raw water probes activated; all six matched non-water controls stayed off. Isolated `water` did not activate.
- In `Water is essential for life.`, the active token was ` is`, not `Water`. Water-context activation is not always literal water-token activation.
- All six raw eleventh-month/alphabet/Roman prompts (word and digit forms) activated on the final ` is`; the corresponding plain positive-integer prompts stayed off.
- Eleven produced the strongest gate within each matched raw month/alphabet/Roman ordinal triplet, but tenth and twelfth also often activated. This is a context-sensitive preference, not an exclusive eleven detector.
- All nine chat month/alphabet/Roman probes activated somewhere; many active positions were template newlines, and the strongest position was not always the final input token.
- All 51 BF16/FP32 activation masks agreed. This does not imply numerically identical gate values or identical results under a different model runtime.

## Frozen description

Activates on "water"/"Water" across lists, ingredients, and environmental descriptions, and on tokens at the boundary before eleven-related continuations. The second pattern includes contexts eliciting 11/eleven, November as the eleventh month, K as the eleventh English letter, or XI as the eleventh positive integer. Activation may occur on a preceding word, separator, or assistant-prefix token rather than the number itself; it is wording-dependent and sometimes appears for neighboring ordinals.

The description was used to design the probe texts. It was not passed to an LLM judge: `verify_hypothesis` accepts a feature ID, texts, and raw/chat formatting.

## Scope and counts

| Probe group | Prompts | Gate active anywhere | Active at final input token |
|---|---:|---:|---:|
| water | 6 | 5 | 3 |
| matched_nonwater | 6 | 0 | 0 |
| ordinal | 24 | 16 | 16 |
| list_boundary | 3 | 1 | 0 |
| chat_ordinal | 9 | 9 | 7 |
| chat_extra | 3 | 1 | 0 |

## Raw ordinal comparisons

Each cell reports maximum gate / final-input-token gate. Neighboring ordinals were not assumed to be true negatives.

| Domain | Wording | Tenth | Eleventh | Twelfth |
|---|---|---:|---:|---:|
| month | word | 0.79297 / 0.79297 | 1.1406 / 1.1406 | 0 / 0 |
| month | digit | 0.88672 / 0.88672 | 1.0234 / 1.0234 | 0 / 0 |
| alphabet | word | 0.74219 / 0.74219 | 1.8984 / 1.8984 | 0.76953 / 0.76953 |
| alphabet | digit | 0.90625 / 0.90625 | 2.2188 / 2.2188 | 0.67578 / 0.67578 |
| roman | word | 0.90625 / 0.90625 | 1.7109 / 1.7109 | 0.91797 / 0.91797 |
| roman | digit | 0.64062 / 0.64062 | 1.5547 / 1.5547 | 0.72656 / 0.72656 |
| integer | word | 0 / 0 | 0 / 0 | 0 / 0 |
| integer | digit | 0 / 0 | 0 / 0 | 0 / 0 |

## Exact probes and active tokens

Positions are zero-based in the complete input passed to Qwen3, including chat template tokens. Token IDs, zero activations and full aligned traces are preserved in `results.json`. Special/control token strings use the tool’s escaped display form.

| ID | Format | Input text | Max gate | Final gate | All active input tokens (position, text, gate) |
|---|---|---|---:|---:|---|
| water-ingredient | raw | Ingredients: flour, sugar, water | 1.3047 | 1.3047 | 6: ` water` = 1.3047 |
| water-list | raw | Essential supplies: food, shelter, water | 1.1797 | 1.1797 | 8: ` water` = 1.1797 |
| water-environment | raw | Clean air and fresh water | 0.62109 | 0.62109 | 4: ` water` = 0.62109 |
| water-capitalized | raw | Water is essential for life. | 1.125 | 0 | 1: ` is` = 1.125 |
| water-sentence | raw | Please pour the water into the glass. | 0.88281 | 0 | 3: ` water` = 0.88281 |
| water-isolated | raw | water | 0 | 0 | none |
| control-ingredient | raw | Ingredients: flour, sugar, milk | 0 | 0 | none |
| control-list | raw | Essential supplies: food, shelter, blankets | 0 | 0 | none |
| control-environment | raw | Clean air and fresh food | 0 | 0 | none |
| control-capitalized | raw | Food is essential for life. | 0 | 0 | none |
| control-sentence | raw | Please pour the milk into the glass. | 0 | 0 | none |
| control-isolated | raw | milk | 0 | 0 | none |
| month-10-word | raw | the tenth month of the year is | 0.79297 | 0.79297 | 6: ` is` = 0.79297 |
| month-10-digit | raw | the 10th month of the year is | 0.88672 | 0.88672 | 9: ` is` = 0.88672 |
| month-11-word | raw | the eleventh month of the year is | 1.1406 | 1.1406 | 7: ` is` = 1.1406 |
| month-11-digit | raw | the 11th month of the year is | 1.0234 | 1.0234 | 9: ` is` = 1.0234 |
| month-12-word | raw | the twelfth month of the year is | 0 | 0 | none |
| month-12-digit | raw | the 12th month of the year is | 0 | 0 | none |
| alphabet-10-word | raw | the tenth letter of the english alphabet is | 0.74219 | 0.74219 | 7: ` is` = 0.74219 |
| alphabet-10-digit | raw | the 10th letter of the english alphabet is | 0.90625 | 0.90625 | 10: ` is` = 0.90625 |
| alphabet-11-word | raw | the eleventh letter of the english alphabet is | 1.8984 | 1.8984 | 8: ` is` = 1.8984 |
| alphabet-11-digit | raw | the 11th letter of the english alphabet is | 2.2188 | 2.2188 | 10: ` is` = 2.2188 |
| alphabet-12-word | raw | the twelfth letter of the english alphabet is | 0.76953 | 0.76953 | 8: ` is` = 0.76953 |
| alphabet-12-digit | raw | the 12th letter of the english alphabet is | 0.67578 | 0.67578 | 10: ` is` = 0.67578 |
| roman-10-word | raw | the tenth positive integer in roman numerals is | 0.90625 | 0.90625 | 8: ` is` = 0.90625 |
| roman-10-digit | raw | the 10th positive integer in roman numerals is | 0.64062 | 0.64062 | 11: ` is` = 0.64062 |
| roman-11-word | raw | the eleventh positive integer in roman numerals is | 1.7109 | 1.7109 | 9: ` is` = 1.7109 |
| roman-11-digit | raw | the 11th positive integer in roman numerals is | 1.5547 | 1.5547 | 11: ` is` = 1.5547 |
| roman-12-word | raw | the twelfth positive integer in roman numerals is | 0.91797 | 0.91797 | 9: ` is` = 0.91797 |
| roman-12-digit | raw | the 12th positive integer in roman numerals is | 0.72656 | 0.72656 | 11: ` is` = 0.72656 |
| integer-10-word | raw | the tenth positive integer is | 0 | 0 | none |
| integer-10-digit | raw | the 10th positive integer is | 0 | 0 | none |
| integer-11-word | raw | the eleventh positive integer is | 0 | 0 | none |
| integer-11-digit | raw | the 11th positive integer is | 0 | 0 | none |
| integer-12-word | raw | the twelfth positive integer is | 0 | 0 | none |
| integer-12-digit | raw | the 12th positive integer is | 0 | 0 | none |
| recipe-boundary-10 | raw | 9. Pour the remaining sauce over the pasta, then spread it out evenly. ↵  ↵ 10. Sprinkle extra Parmesan cheese over the top. | 0 | 0 | none |
| recipe-boundary-11 | raw | 10. Pour the remaining sauce over the pasta, then spread it out evenly. ↵  ↵ 11. Sprinkle extra Parmesan cheese over the top. | 0.88672 | 0 | 17: `1` = 0.88672 |
| recipe-boundary-12 | raw | 11. Pour the remaining sauce over the pasta, then spread it out evenly. ↵  ↵ 12. Sprinkle extra Parmesan cheese over the top. | 0 | 0 | none |
| chat-month-10 | chat | What is the tenth month of the year? Answer briefly. | 1.625 | 1.625 | 16: ` ↵ ` = 1.2891; 21: ` ↵  ↵ ` = 0.54688; 23: ` ↵  ↵ ` = 1.625 |
| chat-month-11 | chat | What is the eleventh month of the year? Answer briefly. | 2.4375 | 2.4375 | 12: `?` = 0.60938; 15: `.` = 0.62109; 17: ` ↵ ` = 1.7344; 22: ` ↵  ↵ ` = 0.90234; 24: ` ↵  ↵ ` = 2.4375 |
| chat-month-12 | chat | What is the twelfth month of the year? Answer briefly. | 1.0078 | 1.0078 | 17: ` ↵ ` = 0.57031; 24: ` ↵  ↵ ` = 1.0078 |
| chat-alphabet-10 | chat | What is the tenth letter of the English alphabet? Answer briefly. | 1.0234 | 1.0234 | 17: ` ↵ ` = 0.87109; 24: ` ↵  ↵ ` = 1.0234 |
| chat-alphabet-11 | chat | What is the eleventh letter of the English alphabet? Answer briefly. | 1.8203 | 1.7266 | 18: ` ↵ ` = 1.8203; 25: ` ↵  ↵ ` = 1.7266 |
| chat-alphabet-12 | chat | What is the twelfth letter of the English alphabet? Answer briefly. | 0.56641 | 0.56641 | 25: ` ↵  ↵ ` = 0.56641 |
| chat-roman-10 | chat | What is the tenth positive integer in Roman numerals? Answer briefly. | 0.89062 | 0 | 18: ` ↵ ` = 0.89062 |
| chat-roman-11 | chat | What is the eleventh positive integer in Roman numerals? Answer briefly. | 1.9219 | 0.92969 | 19: ` ↵ ` = 1.9219; 26: ` ↵  ↵ ` = 0.92969 |
| chat-roman-12 | chat | What is the twelfth positive integer in Roman numerals? Answer briefly. | 0.74609 | 0 | 19: ` ↵ ` = 0.74609 |
| chat-extra-1 | chat | List three uses of water. | 1.3359 | 0 | 7: ` water` = 1.3359 |
| chat-extra-2 | chat | List three uses of milk. | 0 | 0 | none |
| chat-extra-3 | chat | What is the eleventh positive integer? | 0 | 0 | none |

## Measurement and provenance

- Native Qwen3-4B, BF16, SDPA attention; the native network is not replaced with MOLTs.
- Gates are measured from the input to L29 post-attention RMSNorm, matching the dashboard’s `pre_norm` training location.
- The original `SteeringEngine.inspect`, `MoltGates`, and MCP `verify_hypothesis` wrapper/backing method are unmodified.
- A small session initializer permits only T2375 self-authored probes; it does not fabricate training windows or load a corpus.
- A derived gate-only checkpoint contains every exact encoder row, threshold, and input-standardization tensor from the pinned release. The T2375 row and standardization match the earlier verified feature slice exactly.
- The tool uses its original BF16 autocast for gate computation. The supplemental FP32 diagnostic preserves native BF16 model inputs and reports threshold-mask disagreements.
- FP32/BF16 activation-mask disagreements: 0 of 51 prompts. Inspect `precision-diagnostic.json` before comparing values with earlier FP32 research runs.
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- MOLT revision: `081ce901b090b64aeecf9bc8709b57c10411e32a`.
- Runtime: torch `2.7.1+cu126`, transformers `4.57.6`.
- Upstream source: https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/molt_agentic_mcp.py . Local source hashes and the tool schema are preserved.
- These hand-authored probes assess the scope of the explanation. They do not establish general predictive accuracy or a causal mechanism.
