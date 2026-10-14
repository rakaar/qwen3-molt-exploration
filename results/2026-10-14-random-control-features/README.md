# Actual random MOLT control features

These two figures list the recorded feature IDs and saved names used in the **13 October 2026** layer-22 small→hot random controls. They were generated on **14 October** from saved results; no new model inference or GPU rental was performed.

- [Figure 1, full PNG](random-single-molt-feature-list.png) · [Zoomable SVG](random-single-molt-feature-list.svg): the same **100 MOLT IDs** appear on both sides. Left adds +4.5 times the donor contribution at `hot`; right subtracts 4.5 times the receiver contribution at `small`. These are 200 separate trials.
- [Figure 2, full PNG](random-pair-molt-feature-list.png) · [Zoomable SVG](random-pair-molt-feature-list.svg): the same **100 distinct-feature pairs** appear on both sides. Left subtracts receiver A and adds donor B; right reverses those signs. A always receives the receiver's `x_small`, and B always receives the donor's `x_hot`. These are another 200 separate trials.
- [Searchable text list](index.html) · [Single-control CSV](random-single-molt-feature-list.csv) · [Pair-control CSV](random-pair-molt-feature-list.csv) · [Complete JSON inventory](random-control-feature-inventory.json).

Rows retain the original recorded sample order. `L22–M<ID>` uses M for the MOLT/transform ID, corresponding to the same IDs previously written as T. Labels are copied by feature ID from the saved layer-22 dashboard export dated 12 October. Among these 100 IDs, 54 have the saved entry “Not observed in 10M tokens,” 42 have status complete, and four have limited evidence, marked †. No label is invented or inferred from the intervention result.

The original uniform random selection used seed **20261013**, drew 100 IDs without replacement from 2,478 non-target MOLTs, and excluded selected T2339/T941. A shuffled derangement of the same IDs supplies the pairings. These are the uniform controls: all sampled gates were zero on both exact source tokens, so their interventions were zero. The separate active-pair controls are not included in these images.

Layer 22 and full-chat token position 9 are zero-based. The native MLP was retained and attention was unfrozen. C is the complete checkpoint-normalized gate × low-rank-output contribution.

## Provenance and reproduction

Source: [build_random_feature_lists.py](../../experiments/small-hot-controls/build_random_feature_lists.py).

Inputs: [recorded results](../2026-10-13-small-hot-controls/artifacts/results.json) and [saved feature descriptions](../2026-10-13-small-hot-controls/labels.json). Their SHA-256 hashes are recorded in [figure-verification.json](figure-verification.json), along with counts, dimensions, and the builder hash. The SVG files contain all labels as searchable text.

From the repository root, with Pillow and DejaVu Sans fonts installed:

```bash
python experiments/small-hot-controls/build_random_feature_lists.py
```

The builder reads saved results only and asserts exact agreement between each displayed feature/pair and its recorded intervention terms. Original results, logits, and weights are unchanged.
