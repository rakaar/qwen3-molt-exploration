# L34–T965: fixed follow-up judge diagnostic

These 22 queries were prepared and frozen before this run, after inspecting the three old misses. They are post-hoc prompt diagnostics, not new held-out feature scores. The candidate explanation is unchanged wherever it is used. The separate literal-character task changes the task explicitly.

All 22 responses are valid and untruncated. Same pinned judge/runtime: verified. All four original repeats match: **True**.

| Condition | Development → Kit | Ton/ton → katsu | Code / .wait → key |
|---|---:|---:|---:|
| Original query, repeated | 0 | 0 | 0 |
| Only final peak marked; original few-shots | 0 | 1 | 1 |
| Only final peak marked; no few-shots | 0 | 1 | 0 |
| Literal before-K task; all original marks | 1 | 1 | 0 |
| Literal before-K task; final peak only | 1 | 1 | 0 |
| Minimal before-K string; original explanation/few-shots | 1 | 0 | 1 |
| Minimal non-K string; original explanation/few-shots | 0 | 0 | 0 |

## How to read this comparison

- Original positive Fuzz labels are 1 in all three windows because every originally marked token has measured positive gate activation.
- For the literal before-K task, all-highlight expectations are 1, 1, 0: the code window marks comma and plus before h as well as .wait before k. Final-peak-only expectations are 1, 1, 1.
- Minimal positive strings fit the literal before-K rule; minimal negative strings do not. Their native MOLT activation was not measured, and the explanation allows other unspecified activations. They therefore test judge interpretation, not feature specificity.
- Compare single-peak with single-peak/no-few-shot to isolate removing the demonstration bundle. Compare original with single-peak to isolate removing earlier highlight markers while preserving every example character.
- Compare literal-task outputs to rule expectations to check character recognition. This changes both the instruction and explanatory framing and cannot isolate one particular word of the original description.
- Causal interpretations of prompt changes require the original repeats to match. A failed repeat is evidence of instability and must be reported before attributing differences to the diagnostic manipulation.

## Minimal constructed strings

- Row 13, minimal_positive: `Java<< Development>> Kit (JDK).` → `[1]`.
- Row 13, minimal_negative: `Java<< Development>> Guide.` → `[0]`.
- Row 17, minimal_positive: `<< ton>>katsu is a breaded pork cutlet.` → `[0]`.
- Row 17, minimal_negative: `<< ton>>ic.` → `[0]`.
- Row 20, minimal_positive: `cv2<<.wait>>key()` → `[1]`.
- Row 20, minimal_negative: `cv2<<.wait>>()` → `[0]`.

[Original three examples and token gates](index.html#explorer)
