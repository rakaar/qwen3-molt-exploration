# Fixed T2375 gates at both chat-prefix newline tokens

Fixed post-JumpReLU scalar at both input-prefix newline tokens only. Local recipient operator retained. List-reading and generated positions unedited.

Word probability is exp(sum of FP32 token log-softmax values), evaluated in float64 to avoid probability underflow. Exact lowercase complete-word continuations; EOS, punctuation and formatting variants excluded from word-only scores. Word-then-EOS scores are separate.

Native already returns the eleventh word in 2/20 cases; those do not count as newly steered. Native requested-word answers are correct in 13/20 cases, and both requested/eleventh controls are correct in 9/20.

| Gate | Strict eleventh /20 | Newly eleventh /18 | Newly eleventh from correct native /13 | Both controls correct /9 | Truncated /20 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0 | 0 | 0 | 0 |
| 5 | 2 | 0 | 0 | 0 | 0 |
| 10 | 2 | 0 | 0 | 0 | 0 |
| 20 | 2 | 0 | 0 | 0 | 0 |
| 40 | 1 | 0 | 0 | 0 | 0 |

## Item 11 inside a sentence

Word-only counts above are a formatting-sensitive primary measure. The separate first-listed-word measure finds one newly induced item-11 answer at gate 40: list-3-bullets-12 responds “Thank you for the list. The twelfth word in the list is **ring**.” Ring is item 11; apple is item 12. The native twelfth response was bell (item 14), and the saved native eleventh control was lamp (item 10). This is a completed answer containing item 11, with extra text, on a case that failed both controls. It must be shown separately, not described as no new item-11 answers.


All 100 outputs are retained in results.json and probabilities.csv, including wrong words, number-only answers, formatting failures and truncation. Literal eleven/11 is not the eleventh listed word. All native and zero-strength replays are exact. Every steered run edits exactly two input tokens; independent CPU contribution and BF16 rounding audits pass.
