# How much does T2375 change the prediction immediately after “water”?

Across 310 firing occurrences from 245 UltraChat conversations, removing T2375 changed the model’s next-token distribution less than removing the matched active control in 206 cases (66.5%). Each comparison uses the same intact Qwen3-4B prediction. We removed one naturally gated layer-29 contribution at the token containing the final “r” of “water”. For each occurrence we selected the other active transform with the closest gate value. Every occurrence had a match.

| Transform switched off | Cases | Median KL (nats) | Mean KL (nats) |
|---|---:|---:|---:|
| T2375 | 310 | 0.00031 | 0.00471 |
| Matched active control | 310 | 0.00089 | 0.01501 |

KL is calculated over the full next-token vocabulary in the intact-to-ablated direction. Among the 123 low-T2375-KL cases (below 0.0001 nats), the intact model correctly predicted the recorded token in 113 (92%). Among the 32 high-KL cases (at least 0.01 nats), it did so in 11 (34%). The indexed token and its recorded successor had already occurred consecutively in that exact order in 38/123 low-KL cases and 0/32 high-KL cases. These are descriptive comparisons within the sample.

[View the histogram and examples on GitHub Pages](https://rakaar.github.io/qwen3-molt-exploration/sites/t2375-water-cards/#modifier-kl) · [Paired measurements](paired.csv) · [Tail counts](tails.json) · [Protocol](protocol.json) · [Analysis and intervention code](../../experiments/water-active-kl-control-20261121/)
