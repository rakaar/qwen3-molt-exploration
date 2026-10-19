# Seven uncertainty candidates: native query and response activations

This study keeps the exact twelve QA cases used in `confidence-gate-activation`: four correct answers, four original made-up answers and four clean natural abstentions. Each group has two questions from each original passage. Seven features are fixed from the label-only shortlist before measuring these activations. Two unsteered wording controls are excluded from the QA groups and their means.

Priority features: L12–T1667 (Uncertainty Assertion Gate), L14–T1103 (Unknown Information and Missing Details), L16–T764 (Negation of Information Availability).

Other features: L15–T2349, L23–T710, L26–T868 and L18–T74. The report switches between the priority three, other four and all seven. Each feature panel separates query-token and response-token activity; full encoded prompt activity is expandable. The last input position is shown separately before the first answer prediction. One shared numeric colour scale covers every input and output token, including the separate controls.

## Measured results

| Gate | Correct responses active | Made-up responses active | Natural abstentions active | Mean gate on abstentions |
| --- | --- | --- | --- | --- |
| L12–T1667 | 0/4 | 0/4 | 4/4 | 0.2744 |
| L14–T1103 | 0/4 | 0/4 | 4/4 | 0.0896 |
| L16–T764 | 0/4 | 0/4 | 4/4 | 0.5456 |
| L15–T2349 | 0/4 | 0/4 | 4/4 | 0.0918 |
| L23–T710 | 0/4 | 1/4 | 4/4 | 0.3843 |
| L26–T868 | 1/4 | 1/4 | 4/4 | 0.3317 |
| L18–T74 | 0/4 | 0/4 | 0/4 | 0.0000 |

The three priority gates are zero on the question tokens and final prompt position in all twelve QA cases. They can activate elsewhere in the shared prompt, so this is not a claim of complete prompt inactivity. L18–T74 fires on two question texts; L23–T710 is active at three final prompt positions and L26–T868 at one, including a made-up-answer case. Complete token values and original grades remain in the raw results.

These are selected observational contrasts, not validated confidence estimates. Response activity may track emitted missing-information wording. A contrast between response groups does not explain why a steering intervention needs different strengths. No steering or correlation calculation is performed here.

## Exact measurement and verification

Fresh native generation has no monitoring hooks and exactly reproduces all twelve historical native token sequences. Observation-only replay preserves all 312 next-token argmax predictions across fourteen cases. Hooks read the residual entering native post-attention layernorm and return None. MLP and attention remain native; BF16 model with FP32 gate arithmetic.

```
z = (residual after attention − released input mean) / released input std
raw = z dot encoder row + bias
gate = raw if raw > threshold and raw > 0; otherwise 0
```

Query-only positions are defined by tokenizer character offsets overlapping the exact final question. Instructions, passage and chat markers are preserved separately. Response colour is measured while processing an already emitted token; the final prompt position occurs before the first answer prediction. Means include zero gate values and exclude special tokens. All indices are zero-based.

Six candidates fire in at least one wording-control response. L18–T74 does not fire in either control response; its nonzero activity in the QA query tokens is preserved. No control was discarded or retuned to make it positive.

## Local rebuild

From the repository root:

```
python3 experiments/uncertainty-gate-activation/verify.py
python3 experiments/uncertainty-gate-activation/build.py
python3 -m http.server 8787 --directory results --bind 127.0.0.1
```

Open `/2026-10-19-uncertainty-gate-activation/index.html`. Browser QA uses the recorded local Chrome/Puppeteer paths, checks every query/response token and displayed gate against measurements, verifies summary tables and selector modes, and exports primary/secondary PNGs for each QA group. `prepare.py` fetches only small pinned public gate slices, not full MOLT operators. `run.py` is the exact executed GPU source and refuses to overwrite results.

Pinned model, MOLT and runtime revisions, original grading, range-request provenance, raw pre-threshold scores, full token positions, source/asset hashes and remote-download verification are preserved. Full model weights and infrastructure credentials are outside this result. The SPAR GPU is retained under the user's earlier instruction.

Selected SQuAD2.0 text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0. “Made up” means unsupported by the passage, not necessarily false outside it.
