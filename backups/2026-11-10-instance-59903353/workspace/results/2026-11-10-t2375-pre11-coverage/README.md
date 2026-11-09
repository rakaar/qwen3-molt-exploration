# Confirmed pre-eleven coverage of T2375 evaluations

Post-scoring source-text audit. Descriptions, examples, labels and judge responses remain unchanged. No GPU inference or new judge calls were used.

All 100 unique scored windows were matched against the public UltraChat test_sft source, within the original held-out conversation partition. Matching allows only the known literal-versus-escaped added-token display difference. One repeated survey excerpt has two source positions; both give identical boundary classifications.

A case counts when an actually active token immediately precedes the target continuation, ignoring whitespace. In Fuzz and Active Fuzz that active boundary must also be among the highlighted positions. Each passage counts once.

| Evaluation | Standalone 11 / November / equivalents | Additional 2011 suffix cases | Total positive cases |
|---|---:|---:|---:|
| detection | 10 | 2 | 12/50 |
| fuzz | 10 | 2 | 12/50 |
| active_fuzz | 10 | 2 | 12/50 |

The same positive passages are reused across all three methods; these are not three independent sets. Active Fuzz also reuses these passages for negative controls highlighting inactive tokens elsewhere. Those controls are not additional positive pre-eleven cases.

## Breakdown of the shared positive passages

- year_2011_suffix: 2
- november: 2
- date_range_8_to_11: 1
- standalone_11: 7

## Recovered examples

| Fuzz row | Active token | Actual continuation | Category |
|---|---|---|---|
| 13 | `0` | 11.<&#124;im_end&#124;> ↵ <&#124;im_start&#124;>user ↵ Can you provide some examples of Keith Westwater's poetry that can be | year_2011_suffix |
| 39 | ` or` |  November. It involves lighting lamps, decorating homes with rangolis, and exchanging gifts and swee | november |
| 45 | `0` | 11 when Texas A&M announced their move to the Southeastern Conference (SEC) from the Big 12 Conferen | year_2011_suffix |
| 47 | `-` | 11. Using VOS360 media processing as a service, the company can deliver exceptional video quality to | date_range_8_to_11 |
| 49 | ` ` | 11. If you finish early, use the remaining time to review your answers. 12. Avoid leaving any questi | standalone_11 |
| 55 | `. ↵  ↵ ` | 11. Baked Sweet Potato: Pierce sweet potato and microwave for 5-7 minutes. Serve with toppings like  | standalone_11 |
| 57 | `. ↵  ↵ ` | 11. Install the Opener – Ensure the opener is of the correct size, has an auto-stop mechanism and is | standalone_11 |
| 68 | `. ↵  ↵ ` | 11. APPLICABLE LAW ↵ This Contract shall be governed by and construed in accordance with the laws of [ | standalone_11 |
| 69 | ` ↵ ` | 11. Charles W. Socarides ↵ 12. Edward M. Weinshel ↵ 13. Nathan M. Szajnberg ↵ 14. James L. Fosshage<&#124;im_en | standalone_11 |
| 86 | `  ↵  ↵ ` | 11. Which factors do you think an automotive manufacturer should prioritize when developing electric | standalone_11 |
| 90 | `. ↵  ↵ ` | 11. Google Analytics: This tool helps you track your website's traffic, user behavior, and other imp | standalone_11 |
| 98 | ` and` |  november could be a good choice. If you are a food enthusiast, then the sagra del cinghiale or fier | november |

## Corrections to the earlier excerpt-only audit

Eight excerpts ended after list item 10. The recovered source establishes which actually continue with item 11; an end of item 10 alone was not sufficient evidence. The visible November-the passage continues with 2nd, 1930 and must not be counted merely because November occurs in its prefix.

See counts.json for exact per-method row indexes, prefixes, continuations, activation values and highlighted-token checks. boundary-audit.json also retains inactive-boundary events. Source versions and SHA-256 hashes are in source-manifest.json.
