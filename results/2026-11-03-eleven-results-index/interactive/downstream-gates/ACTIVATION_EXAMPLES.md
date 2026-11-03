# Activating-example review of the four switched month gates

Inspected the authenticated original dashboard: 12 strongest, 6 medium and 6 lower-strength windows for each of four transforms. All 96 displayed windows were read, including exact active-token locations and following context. These are the original dashboard examples, not the separate 64-conversation preview.

**The clearest before-twelve lead is L31-T949.** Its strongest window peaks before December; another strongest window activates before 12 months, and a lower-strength example activates immediately before 12th in a horse-racing result. It also activates before other months, numbers and ordinals, so these examples do not establish twelve selectivity.

Square brackets mark the active token. A middle dot inside brackets denotes a space. Tooltip strengths are displayed rounded to three decimals.

## L30-T433: Temporal and Calendar Token Gate

[Original dashboard examples](https://amsterdam.taild80f17.ts.net/layers/layer-30/features/433.html)

Month prompt gate: 0.565827 -> 0.000000 after L29-T2375 removal.

Broad calendar, seasonal and date contexts. Peaks include spaces before 28 days, leap-year divisor 4, February 31st and December 21st, as well as autumn and calendar names. The reviewed windows do not suggest eleven-specific or twelve-specific activation.

| Short source excerpt | Displayed gate | Location | Evidence |
|---|---:|---|---|
| ` has[·]28` | 4.094 | Conversation 6955, token 346 | Space immediately before 28 days; Strongest examples (one per conversation), example 1 |
| ` by[·]4` | 3.781 | Conversation 7492, token 293 | Space immediately before leap-year divisor 4; Strongest examples (one per conversation), example 4 |
| ` December[·]21` | 3.422 | Conversation 3977, token 243 | Space immediately before December 21st; Strongest examples (one per conversation), example 10 |

## L31-T797: Winter Weather Context Gate

[Original dashboard examples](https://amsterdam.taild80f17.ts.net/layers/layer-31/features/797.html)

Month prompt gate: 0.000000 -> 0.655309 after L29-T2375 removal.

Strongest windows consistently concern snow, snowflakes, white winter landscapes and cold conditions. Peaks occur on prepositions before that content and on snow tokens themselves. This supports a winter-context lead for the month task, without establishing a twelve feature.

| Short source excerpt | Displayed gate | Location | Evidence |
|---|---:|---|---|
| ` covered[·in] deep snow` | 5.188 | Conversation 3650, token 351 | Preposition before deep snow; Strongest examples (one per conversation), example 1 |
| ` layer[·of] white` | 4.938 | Conversation 8327, token 448 | Preposition before white winter landscape; Strongest examples (one per conversation), example 3 |

## L31-T949: Temporal and Financial Timeframe Prepositions

[Original dashboard examples](https://amsterdam.taild80f17.ts.net/layers/layer-31/features/949.html)

Month prompt gate: 0.000000 -> 0.562312 after L29-T2375 removal.

The strongest window peaks on of before December. Another strongest window peaks on and before 12 months, with an intervening space. A lower-strength window peaks on the space immediately before 12th in a horse-racing result. Counterexamples include June, a space before arithmetic result 24, and to before 8th grades. This is a concrete before-twelve lead with substantial broader activation.

| Short source excerpt | Displayed gate | Location | Evidence |
|---|---:|---|---|
| ` end[·of] December` | 4.875 | Conversation 3055, token 141 | Before December; Strongest examples (one per conversation), example 1 |
| ` 6[·and] 12 months` | 4.063 | Conversation 3943, token 571 | Before 12 months; one intervening space; Strongest examples (one per conversation), example 7 |
| ` finished[·]12th` | 0.555 | Conversation 8832, token 950 | Space immediately before ordinal 12th; Lower-strength examples, example 6 |
| ` end[·of] June` | 4.844 | Conversation 2680, token 400 | Counterexample: before June; Strongest examples (one per conversation), example 2 |
| ` returns[·]24` | 1.227 | Conversation 9075, token 929 | Counterexample: space before arithmetic result 24; Medium-strength examples, example 1 |
| `6th[·to] 8th` | 1.219 | Conversation 3193, token 85 | Counterexample: to before 8th grades; Medium-strength examples, example 4 |

## L34-T84: Connector Tokens in Compounds and Idioms

[Original dashboard examples](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/84.html)

Month prompt gate: 0.000000 -> 0.587403 after L29-T2375 removal.

Strong windows include x in dimension notation, in within in-depth, or between alternatives, and email/format boundaries. No clear before-eleven or before-twelve pattern appears in the reviewed windows; the broad connector explanation remains provisional.

| Short source excerpt | Displayed gate | Location | Evidence |
|---|---:|---|---|
| ` h[·x] d` | 3.203 | Conversation 9649, token 155 | Connector x in dimension notation; Strongest examples (one per conversation), example 1 |
| `[·in]-depth information` | 2.672 | Conversation 3213, token 658 | Word fragment in before -depth; Strongest examples (one per conversation), example 4 |
| `[·or] decrease the` | 2.688 | Conversation 9552, token 349 | Connector or before an alternative; Strongest examples (one per conversation), example 3 |

## What this suggests

L31-T949 deserves a causal follow-up because it has genuine precursor timing for twelve-related completions, including a non-month ordinal example. The counterexamples remain essential: its displayed evidence also supports broad timeframe/range behavior.

L31-T797 supplies a different lead: a winter-context change could fit a December preference in this month task. This is a hypothesis about the month case, not an explanation of the alphabet or Roman flips.

Neither label descriptions nor activation examples show that these transforms cause the answer change. An appropriate next causal check would remove each newly active L31 contribution separately in the L29-T2375-off month run and measure November/December support. That intervention has not been run here.

The scope is the displayed example selection, not every activation in the corpus. Full live label descriptions matched the saved descriptions. Original page URLs, retrieved timestamps, group identities, coordinates, token IDs and displayed strengths are preserved in the raw inspection records under comparisons/2026-11-03-eleven-downstream-gates/activation-example-inspection/. Their hashes and selected evidence are in activation-example-review.json.

Reproduce the audit/report from those saved records: `python3 experiments/eleven-downstream-gates-20261103/inspect_examples.py`. No model inference or downstream intervention is performed.
