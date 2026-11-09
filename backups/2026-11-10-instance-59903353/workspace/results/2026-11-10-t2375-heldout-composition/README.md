# Post-hoc composition of the T2375 held-out sets

This analysis was performed after all judge decisions were complete. It changes neither description, case selection, labels, nor judge outputs. The categories below are descriptive post-hoc diagnostics, not a new held-out score.

## What the positive cases contain

Each test has 50 positive examples and 50 negatives. All three methods share the same 50 positive windows; they are not three independent collections of positives.

| Positive group | Count | Detection: old → revised correct | Fuzz: old → revised correct |
|---|---:|---:|---:|
| Active water-containing token, including compounds | 29 | 29 → 29 | 28 → 29 |
| Other active tokens | 21 | 0 → 1 | 0 → 1 |

Water accounts for 29/50 (58%) of the positive examples. The 50 negative passages contain no water in Detection/Fuzz; Active Fuzz negatives include 29 water-containing passages but highlight inactive positions. Thus water is a majority of positives, not a majority of all Detection/Fuzz target passages.

## Eleven visibility and truncation

- None of the 100 displayed target passages per method contains literal standalone 11 or English eleven/eleventh. The revised explanation itself contains those strings; this count concerns example passages only.
- One positive passage contains November. It is the only non-water positive newly classified correctly by the revised explanation.
- Eight positive windows finish after list item 10, on punctuation or whitespace. These are plausible pre-11 boundary candidates, but their following tokens are absent, so they cannot be counted as confirmed pre-11 cases.
- Other non-water windows include partial numbers/dates and October-related contexts. Their relationship to eleven is unresolved without more context.
- All 50 positive windows stop on an active token; 50/50 stop on their maximum activation. A following 11 or November could therefore be cropped out, but this audit does not assume that continuation.

## Interpretation

- The supplied excerpts give little explicit evidence for the newly added eleven clause. This is a plausible contributor to the small observed score gains.
- However, 21 positive examples are non-water, and the revised explanation identifies only 1 of those 21. The result cannot be attributed solely to an overwhelming abundance of easy water cases.
- The scores alone do not distinguish incomplete wording, judge limitations, omitted context, or non-eleven behavior of this gate.
- The two unique corrected positives across the three tests are the November-context window and a high-water compound. The new false positive in Active Fuzz is an inactive comma in a passage with water elsewhere.
- A claim about pre-11 coverage requires verified full continuations or an independently specified evaluation of eleven-related prompts with gate labels and controls. Further description development should not tune to these now-viewed test cases.
