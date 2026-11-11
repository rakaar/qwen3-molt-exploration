# Fixed L29 T2375 at every supplied input token

Result: no newly induced item-11 answer in 140 edited runs. Literal eleven, malformed text and truncated list recitations are not item retrieval. Longer completed answers were inspected; some retain the requested item with extra text. No new item-11 answer appeared even allowing sentence-form answers.

Pilot: word orders 1–2, numbered and bullets, request 5 or 12; fixed gates 2,4,8,16,24,32,40,48,56,64 (80 variants). Follow-up: word orders 3–5, same formats/requests; gates 40,48,56,60,64 (60 variants), frozen after pilot. This is adaptive exploration, not a held-out confirmation of a successful band.

Gate replacement: add (target - natural gate) * recipient operator to the native L29 MLP output at every supplied input token, including chat wrappers and assistant prefix. Native later layers run normally. Generated, candidate-continuation and terminal tokens are never edited. This is absolute post-JumpReLU gate replacement, not multiplicative scaling. Gate 2 slightly lowers a few naturally active numbered-list gates above 2.

Twenty native replays and zero-strength controls exactly match the saved revised prompts without starting-from-1. Thirteen requested-word native controls are correct; nine requested/eleventh paired controls are correct. Two native twelfth responses already return item 11 and cannot count as new successes. Baseline failures are retained.

All 13,580 edited input positions have positive measured output changes; zero generated positions edited. All-position gate/delta norms retained. Independent gate/operator reconstruction and BF16 rounding verified on 840 predetermined vector samples. Word probabilities use exact lowercase continuations at response start, not answer extraction inside sentences. Full words and word-plus-EOS are separate.

Model Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c; MOLT revision 081ce901b090b64aeecf9bc8709b57c10411e32a. Native BF16, observer FP32, eager attention, TF32 disabled, thinking off, greedy, max 32 generated tokens.

Files: index.html/explorer-data.json for the report; pilot/ and followup/ contain original results, probabilities, compressed audit vectors, source and frozen protocols; adaptive-decision.json records how the follow-up range was chosen.
