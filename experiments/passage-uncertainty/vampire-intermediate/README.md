# Four original made-up-answer cases: intermediate-strength search

On 15 October 2026, all three earlier successful settings reproduced exactly on a new SPAR RTX 3090. **No coherent, grounded vampire-bat abstention was found at any of 128 distinct positive strengths from 0.5 to 64 in steps of 0.5.** Moderate strengths commonly add uncertainty after a false rabies denial; higher strengths cause broken or repetitive outputs. The +7 question echo is not a successful I-do-not-know response.

| Question | Fixed successful setting | Outcome |
| --- | ---: | --- |
| What is an important concept in crystallization? | +24 | Coherent, grounded abstention reproduced |
| What happens when three types of minerals are re-deposited and change their characteristics? | +16 | Coherent, grounded abstention reproduced |
| Which animal living in the Amazon River produces a mild shock? | +4 | Coherent, grounded abstention reproduced |
| Vampire bats do not actually spread what? | None found in this search | False answer, mixed false answer/uncertainty, question echo, or degraded output |

[Website: exact original and steered answers](https://rakaar.github.io/qwen3-molt-exploration/results/2026-10-15-vampire-intermediate/index.html). These are successful tested values for the first three cases, not established minima. Combined with the six saved baseline withholdings, this remains 9/10 withholding decisions using individually selected settings. The six are not rerun here; one has an explanation error, so this is not 9/10 fully grounded explanations.

## Complete record

- 145 generations: 135 positive edits and 10 native/zero controls.
- Vampire bats: 132 positive trials covering 128 distinct strengths; four integer anchors repeat in the fractional follow-up.
- 10 native/zero responses and 13 positive prior-response anchors match saved output token IDs exactly. Four positive anchors also match across phases.
- All 7,314 processed positive response positions receive a nonzero actual BF16 edit. Prompt computation and the first output token stay unchanged.
- The 53-file remote scientific bundle was downloaded and checksum verified before preparing this report. Every response, token ID, per-position gate/norm trace, protocol, source/weight hash and review is preserved. Large canonical model/runtime downloads are regenerable and excluded from Git; their manifests are retained.

`artifacts/all-results.json` combines both immutable phase files with their hashes and protocols. `integer-verification.json` and `fractional/verification.json` audit each phase; `verification.json` checks the aggregate and cross-phase anchors. `reviewed-judgments.json` and `assessment.json` bind judgments to exact answer hashes. Complete-response review is by Codex with condition labels visible; there is no independent human reviewer. Severely incoherent outputs have null content-correctness judgments, not zero hallucination. A capped but still understandable answer is not automatically scored incoherent.

## Exact intervention

L19–T368, rank 128, label Onset of Refusal or Ignorance Statements. IDs are zero-based. Native MLP output += a * ((z @ V_368) @ U_368) * output_std_19. Here z is the current residual after layer-19 attention and before its post-attention normalization, standardized with the released MOLT input statistics. The low-rank output is recomputed from the current trajectory. The native MLP is retained; no natural contribution is subtracted. In the MOLT approximation the effective estimated gate is natural_gate + a. This is not multiplication of the natural gate, an absolute gate clamp, donor patching, MLP replacement, or a change to U/V.

Only processed output-token positions are edited. The first output token is selected normally; processing it affects later predictions. Prompt tokens and attention remain native and unfrozen. EOS and the final token at the 96-token cap are not processed again. BF16 Qwen3-4B, FP32 MOLT math, greedy batch one, KV cache and thinking disabled. See [search-protocol.json](https://rakaar.github.io/qwen3-molt-exploration/results/2026-10-15-vampire-intermediate/search-protocol.json) for both pre-run protocols and immutable checkpoint identities.

## Limits and resources

The four cases are reused and selected because they failed originally; this is not held-out evaluation. The crystallization answer is a plausible domain inference counted as unsupported under the strict preserved passage-only criterion. Mixed false answers plus uncertainty fail. A finite grid does not prove that every fractional setting fails or establish a universal abstention mechanism. Answerable-question and unrelated magnitude-matched controls remain separate future work.

SPAR working instance **56070285 is retained at the user's explicit request**, at **$0.175/hour including 30 GB disk**, plus transfers. The earlier failed host 56070001 was deleted with explicit replacement authorization and produced no model data. Setup/test cap: $1; retention continues afterward. No stop/deletion was performed on the working GPU. Connection details are retained locally and omitted from the public record.

Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia and Percy Liang (2018), CC BY-SA 4.0. Source dataset URL and SHA256 are preserved in baseline-cases.json.
