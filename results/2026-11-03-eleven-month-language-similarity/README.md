# Eleventh-month full MOLT contribution similarity across five languages

Natural full L29–T2375 contributions at the final input token in every prompt. No new inference or interventions were needed. Completion prefixes and chat questions remain separate.

## Completion prefixes

| Language | English | Spanish | Chinese | French | Hindi |
|---|---:|---:|---:|---:|---:|
| English | 1.000000 | 0.990704 | 0.987373 | undefined | undefined |
| Spanish | 0.990704 | 1.000000 | 0.982793 | undefined | undefined |
| Chinese | 0.987373 | 0.982793 | 1.000000 | undefined | undefined |
| French | undefined | undefined | undefined | undefined | undefined |
| Hindi | undefined | undefined | undefined | undefined | undefined |

| Language | Compared position | Token | Gate | Contribution L2 |
|---|---:|---|---:|---:|
| English | 7 | " is" | 1.158554 | 17.357875 |
| Spanish | 7 | " es" | 1.243391 | 20.826672 |
| Chinese | 4 | "是" | 0.740417 | 9.513103 |
| French | 9 | " est" | 0.000000 | 0.000000 |
| Hindi | 23 | "�" | 0.000000 | 0.000000 |

## No-thinking chat questions

| Language | English | Spanish | Chinese | French | Hindi |
|---|---:|---:|---:|---:|---:|
| English | 1.000000 | 0.996344 | 0.997465 | 0.996634 | 0.993324 |
| Spanish | 0.996344 | 1.000000 | 0.994475 | 0.998530 | 0.996429 |
| Chinese | 0.997465 | 0.994475 | 1.000000 | 0.993771 | 0.991611 |
| French | 0.996634 | 0.998530 | 0.993771 | 1.000000 | 0.995414 |
| Hindi | 0.993324 | 0.996429 | 0.991611 | 0.995414 | 1.000000 |

| Language | Compared position | Token | Gate | Contribution L2 |
|---|---:|---|---:|---:|
| English | 28 | "\n\n" | 2.783611 | 68.362134 |
| Spanish | 32 | "\n\n" | 2.459084 | 61.063053 |
| Chinese | 24 | "\n\n" | 2.369594 | 51.397175 |
| French | 35 | "\n\n" | 2.625252 | 66.440515 |
| Hindi | 65 | "\n\n" | 1.529316 | 27.729281 |

## Method and limits

c = gate(z) * ((z @ V @ U) * output_std); z = (native residual - input_mean) / input_std.

Original positive full MOLT contribution in native MLP-output units; no final RMSNorm, unembedding or BF16 subtraction rounding.

- The two formats use different wording as well as a chat template; format and wording effects are not isolated.
- Raw French and Hindi have no active input token and do not supply valid eleventh-month answers in these continuations.
- No causal intervention, cross-language patching or test of a language-independent ordinal mechanism is performed.
- Cosine of a zero contribution is undefined, including its diagonal; no language-specific token search was performed.

All compared vectors, exact prompts, token IDs, input activation traces, responses and source hashes are preserved in summary.json; full vectors are in contributions.npz. CSV matrices and magnitudes are provided separately. The local findings site displays the two matrices under #languages.

Reproduce with python3 experiments/eleven-month-languages-20261103/contribution_similarity.py from the project root.
