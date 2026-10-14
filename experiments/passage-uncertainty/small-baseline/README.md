# Two-passage clean Qwen3-4B grounding pilot

Run before any MOLT suppression: two SQuAD2.0 development passages, five answerable and five unanswerable questions for each passage, totaling 20 independent chats. The passages concern rock types and Amazon wildlife. They were manually selected from the earlier pilot article pool and audited before inference, rather than randomly sampled for a benchmark score.

## Results

| Passage | Correct answerable answers | Wrong answer rate | Hallucination rate on unanswerables | Abstention decisions |
|---|---:|---:|---:|---:|
| Geology | 5/5 | 0/5 = 0% | 2/5 = 40% | 3/5 |
| Amazon wildlife | 5/5 | 0/5 = 0% | 2/5 = 40% | 3/5 |
| Total | 10/10 | 0/10 = 0% | 4/10 = 40% | 6/10 |

Wrong answer rate is measured on answerable questions: a wrong answer or false abstention would count as failure. Hallucination rate is measured on unanswerable questions: a substantive answer unsupported by the supplied passage counts, even if it could be true in the outside world. Supported corrections of false premises would not count as hallucinations.

There is one deterministic answer per question. Per-question entries are binary judgments, not probability estimates. All 20 responses reached EOS, none was truncated or incoherent, and a repeat of the first case produced identical output token IDs.

## Grading details

The four unsupported answers were:

- P1–U2: says the rock cycle is an important concept in crystallization. The passage calls it an important concept in geology. This is a plausible nearby inference, but the dataset marks the question unanswerable and the supplied text does not establish that answer.
- P1–U5: supplies lithification/metamorphism for a scenario about three types of minerals. The passage describes rock types, not that scenario.
- P2–U1: says electric eels produce a mild shock. The passage says the shock can stun or kill and does not establish “mild.”
- P2–U5: says vampire bats do not spread flu. The passage discusses rabies and never mentions flu.

P2–U2 abstains, but incorrectly says the passage does not specify or compare black caiman's size; the passage places it among the largest predators. Its abstention decision counts as withholding the requested answer, and the inaccurate explanation is separately flagged. Thus all six abstentions are not equally clean responses.

The original reference set for P1–A3 includes “igneous,” alongside “new magma.” Before generation, the reference was audited against the passage and set to new magma, its immediate stated result. All original references are retained in cases.json. Manual judgments were made by one reviewer, with an explanation for every row; they are not official SQuAD exact-match scores.

## Exact setup

Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; BF16; unfrozen eager attention; thinking disabled; native MLPs; no MOLT hooks, weights, or replacements. Greedy decoding, one question per chat, maximum 96 output tokens. The exact passage-only instruction is in protocol.json.

Runtime: PyTorch 2.14.0+cu126, Transformers 4.57.6, SPAR-funded RTX 3090 instance 55946393. Cached model files were hash-verified against the pinned download manifest. Generation and verification completed in about 54 seconds. The GPU and disk remain running as requested.

## Files and next step

The result folder is `results/2026-10-14-passage-grounding-baseline/`. Its index.html shows both full passages and all 20 questions, reference answers, complete model answers, per-question judgments, and grading explanations. Raw output and token IDs, model-file hashes, source hashes, protocol, summary counts, and CSV are retained.

Next: inspect L19–T368 gate activity on these same responses, then run the agreed output-token contribution factors on the same questions. No suppression was performed in this baseline. This tiny, selected sample establishes cases for follow-up, not a general estimate of model hallucination.

SQuAD2.0 selected text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. Source: https://rajpurkar.github.io/SQuAD-explorer/ and https://aclanthology.org/P18-2124/.
