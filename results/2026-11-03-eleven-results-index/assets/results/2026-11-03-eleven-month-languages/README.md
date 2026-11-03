# Natural L29–T2375 activation across month languages

18 native prompts, each repeated with a returning-None observer; no contribution edits. Both 11th and 12th month were tested. English raw prefixes anchor the original study. Spanish, Chinese, French and Hindi each have one raw completion prefix and one no-thinking chat question per ordinal. Exact prompts, rendered input, continuations, token roles and gates are in summary.json; compact records are in activation.csv.

## Chat questions: final input gate before the first answer token

| Language | 11th month | 12th month |
|---|---:|---:|
| Spanish | 2.459084 | 1.033556 |
| Chinese | 2.369594 | 0.000000 |
| French | 2.625252 | 1.294186 |
| Hindi | 1.529316 | 0.798006 |

All eight chat answers are the correct month names in the question language. The final input token is a newline in the assistant prefix, after the user question. These values describe natural input activity, not activation on the literal ordinal word. Within the user-question text, Spanish/French/Hindi have zero gate throughout these particular questions. Chinese eleventh activates at 叫 (1.052920); the Chinese twelfth question has no active input token, including its chat prefix.

## Raw prefixes

English eleventh: 1.158554 at ` is`; Spanish: 1.243391 at ` es`; Chinese: 0.740417 at `是`. Their matched twelfth prefixes have zero gate throughout input. French and Hindi have zero gate for both ordinals in these chosen raw prefixes. French eleventh incorrectly continues December; the Hindi raw continuations do not directly answer the month. Do not use those failures as evidence that the native model cannot answer the chat question, which it answers correctly.

Earlier alternative eleventh raw prefixes in both Spanish and Chinese have zero input gate too; they are retained in summary.json as historical variants. Thus this is not a claim that every translated eleventh prompt activates the feature. Activity on some twelfth chat questions also rules out an exclusive-eleventh activation interpretation for this set.

## Verification and reproduction

All source, protocol and vector hashes checked. The observer preserved every full-vocabulary logit hash and generated token compared with native inference. The four earlier Spanish/Chinese eleventh input traces reproduced exactly. Independent CPU checks reconstructed input standardization, encoder, thresholded gates, operators and full contributions over all 430 input positions. Maximum encoder error: 0.00000310; maximum operator relative L2 error: 0.000000757. See validation.json and chat-answer-review.json.

No final RMSNorm or vocabulary projection was used in the gate measurements. Raw-prefix and chat results are reported separately; one wording per new language/format is exploratory. The study used pinned native BF16 Qwen3-4B with FP32 MOLT observation, eager attention, TF32 off and up to 24 greedy output tokens.

Source: experiments/eleven-month-languages-20261103/{run.py,protocol.json,analyze.py}. Guarded SPAR transfer helper: gpu-rentals/2026-11-02-eleven-activation/languages_remote.py. Evidence: comparisons/2026-11-03-eleven-month-languages/verified-remote-backup.tar.gz and remote-export/. The downloaded archive and every file were SHA-256 verified. No GitHub push or GPU lifecycle action was performed.
