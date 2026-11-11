# Matched follow-up: delete starting from 1

Native word-only correct: 20/30. First recognized listed word correct: 20/30. Truncated: 0/30.

All 30 frozen cases are included. Each native generation and its terminal read-only trace pass were exactly reproduced with the passive observer. Separate month calibration passed. No causal intervention was performed.

| Format | Requested item | Word-only correct | Query active | Answer-prefix active | Response text active |
|---|---:|---:|---:|---:|---:|
| numbered | 5 | 3/5 | 0/5 | 0/5 | 0/5 |
| numbered | 11 | 5/5 | 0/5 | 5/5 | 0/5 |
| numbered | 12 | 5/5 | 0/5 | 1/5 | 0/5 |
| bullets | 5 | 3/5 | 0/5 | 0/5 | 0/5 |
| bullets | 11 | 2/5 | 0/5 | 5/5 | 0/5 |
| bullets | 12 | 2/5 | 0/5 | 2/5 | 0/5 |

## All native responses

| Case | Expected | Response | Word-only correct |
|---|---|---|---|
| list-1-numbered-5 | wheel | wheel | True |
| list-1-numbered-11 | pencil | pencil | True |
| list-1-numbered-12 | drum | drum | True |
| list-1-bullets-5 | wheel | wheel | True |
| list-1-bullets-11 | pencil | pencil | True |
| list-1-bullets-12 | drum | pencil | False |
| list-2-numbered-5 | ring | ring | True |
| list-2-numbered-11 | shoe | shoe | True |
| list-2-numbered-12 | hammer | hammer | True |
| list-2-bullets-5 | ring | ring | True |
| list-2-bullets-11 | shoe | spoon | False |
| list-2-bullets-12 | hammer | hammer | True |
| list-3-numbered-5 | ticket | kite | False |
| list-3-numbered-11 | ring | ring | True |
| list-3-numbered-12 | apple | apple | True |
| list-3-bullets-5 | ticket | wheel | False |
| list-3-bullets-11 | ring | lamp | False |
| list-3-bullets-12 | apple | bell | False |
| list-4-numbered-5 | fork | fork | True |
| list-4-numbered-11 | ticket | ticket | True |
| list-4-numbered-12 | spoon | spoon | True |
| list-4-bullets-5 | fork | fork | True |
| list-4-bullets-11 | ticket | shoe | False |
| list-4-bullets-12 | spoon | spoon | True |
| list-5-numbered-5 | lamp | drum | False |
| list-5-numbered-11 | hammer | hammer | True |
| list-5-numbered-12 | ticket | ticket | True |
| list-5-bullets-5 | lamp | wheel | False |
| list-5-bullets-11 | hammer | hammer | True |
| list-5-bullets-12 | ticket | hammer | False |

## Interpretation limits

Gate color represents the scalar JumpReLU gate, not proof of an indexing function. List activations precede the later question. Response-token gates are measured when the generated token is subsequently processed, not when that token is selected. The last generated token has a separately flagged trace-only extra forward, with no additional generated token. All input template tokens and EOS are retained in raw data.

All prompts match their original case IDs, word orders, formats, requested positions and settings. The only prompt change deletes the exact substring " starting from 1". The unchanged native runner checks every generated and terminal-unused logit against its passive-observer repeat.
