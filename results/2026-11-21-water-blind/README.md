# Does T2375 fire more often when “water” modifies the following noun?

We sampled 240 occurrences without replacement from 5,989 eligible occurrences in the final 11,765 conversations of UltraChat test_sft. Each case fell within the first 1,024 Qwen tokens of its conversation. Joined words and lexically hyphenated uses were excluded. All 240 causal prefixes were distinct. The sample spanned 199 prompt IDs, all distinct from those in the original dashboard collection.

One Codex text-only annotation pass labeled the grammatical role before measurement. A second independent Codex pass agreed on 239 of 240 labels. The primary analysis keeps the original coding of “Denver Water proposal” in other uses.

| Use | Firing / sampled | Firing rate | Mean gate, all | Median gate, all | Mean gate when firing |
|---|---:|---:|---:|---:|---:|
| Modifies adjacent noun | 64 / 73 | 87.7% | 1.442 | 1.501 | 1.644 |
| Other uses | 33 / 167 | 19.8% | 0.196 | 0.000 | 0.990 |

Differences below use modifier minus other. Intervals resample whole conversations 10,000 times.

| Measure | Difference | 95% interval |
|---|---:|---:|
| Firing rate (percentage points) | 67.911 | 57.381 to 77.504 |
| Mean gate | 1.246 | 1.051 to 1.428 |
| Mean gate among firings | 0.654 | 0.479 to 0.817 |

Measurement used the full native Qwen3-4B causal prefix ending exactly on “water,” the residual input to layer 29’s post-attention layer norm, and the released T2375 gate threshold. A gate above zero counts as firing. The subsequent words were available to the labeler only. Topic, wording and sentence structure can all differ between groups.

- [All 240 observed contexts, labels and gates](observations.json)
- [Sampling frame and rubric](sampling.json)
- [Frozen labels](labels.json) and [timestamped hashes](label-freeze.json)
- [Independent text-only review](annotation-review.json) and [sensitivity checks](sensitivity.json)
- [Measurement versions and checksums](measurement.json) and [36 calibration positions](calibration.json)
- [Sampling, annotation, measurement and analysis code](../../experiments/water-blind-20261121/)
- [Earlier activation-selected pilot](../2026-11-21-t2375-water-continuations/README.md)
