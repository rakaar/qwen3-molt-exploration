# L34-T1468: activating examples of the shared decreasing gate

[Original authenticated dashboard](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/1468.html)

All 12 strongest, 6 medium and 6 lower-strength windows shown on the authenticated original dashboard; not an exhaustive corpus scan.

All 18 strongest/medium window maxima fall on the an/An token, including word fragments in Anubis, Anagram, Ankh, Annotating and Anxiety.

One strongest example activates on an, the following space and the first digit of 11-year-old. A lower example activates on the space immediately before 11.4; another activates on 9 before the final 11 in 1911.

No clear twelve-related activation example was found in the 24 displayed windows.

Square brackets mark the active token. Gate values are rounded dashboard tooltip values.

| Context | Active token | Displayed gate | Conversation / position |
|---|---|---:|---|
| about an 11-year-old | ` an` | 6.094 | 3936 / 247 |
| about an [space]11-year-old | space immediately before 11 | 4.438 | 3936 / 248 |
| about an 11-year-old | first digit `1` | 2.391 | 3936 / 249 |
| rate is at [space]11.4 percent | space immediately before 11.4 | 0.617 | 9329 / 469 |
| dynasty in 1911 | digit `9`, before the remaining `11` | 0.602 | 5243 / 865 |

The second digit of 11-year-old and both digits of 11.4 have zero displayed gate. The word-fragment examples argue for a broader lexical pattern than the label description alone suggests.

## Gate at final input ` is` in our paired experiment

| Prompt family | Native | After L29-T2375 ablation |
|---|---:|---:|
| month | 1.289993 | 0.717599 |
| alphabet | 1.084717 | 0.681166 |
| roman | 2.429366 | 1.319434 |

The examples support a broad an/An token pattern with some before-11 timing. They do not establish an abstract eleven feature or explain why its gate decreases at is across the three intervention cases. No causal role for this later transform has been tested.

This is a review of gate activation examples, not a measurement of the full operator contribution or a downstream intervention. No model inference was run.

Source token IDs, coordinates, displayed strengths, the source hash and all window texts are retained in the raw capture and T1468-examples.json.

Reproduce: `python3 experiments/eleven-downstream-gates-20261103/inspect_t1468.py`.
