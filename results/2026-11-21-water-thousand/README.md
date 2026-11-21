# T2375 on 1,000 random water occurrences

The frame contained 5,912 occurrences in the first 1,024 rendered tokens of 1,935 eligible UltraChat test_sft conversations. The uniform sample without replacement contained 689 distinct conversations. All 1,000 model prefixes were distinct. The original dashboard corpus rows and all 199 conversations from the 240-case study were excluded. The dashboard evaluation supplied 50 activating windows and 50 zero-gate windows for the earlier pilot.

Each category summarizes its own denominator. Firing means a native T2375 gate above zero. The median includes every sampled occurrence in the category. The examples on the local cards illustrate the observed range.

| Use | Sample share | Fired / sampled | Firing rate | Median gate |
|---|---:|---:|---:|---:|
| Noun modifier | 17.9% | 157 / 179 | 87.7% | 1.47 |
| Standalone noun | 49.7% | 78 / 497 | 15.7% | 0.00 |
| Joined or inflected word | 28.0% | 58 / 280 | 20.7% | 0.00 |
| Hyphenated adjective | 1.4% | 11 / 14 | 78.6% | 1.81 |
| Fixed expression | 1.0% | 1 / 10 | 10.0% | 0.00 |
| Other uses | 2.0% | 5 / 20 | 25.0% | 0.00 |

310 of the 1,000 positions fired. The modifier-minus-standalone firing-rate difference was 72.0 percentage points (95% interval 65.6 to 78.2, resampling whole conversations).

Two Codex annotators each labeled a separate half of the sample using text alone. Labels were frozen before measurement. Fixed expressions include figurative and title uses whose classification is debatable. The joined category includes compounds, plurals, inflections and names. BPE tokens can span a whole joined word; the gate is read on the token containing the final r of water. An activation can therefore coincide with part or all of a following spelling. The measured relationship is observational.

- [Local results cards](../../sites/t2375-water-cards/index.html)
- [All observations and gates](observations.json)
- [Sampling frame](sampling.json) and [category rubric](../../experiments/water-thousand-20261121/rubric.txt)
- [Labels](labels.json), [premeasurement hashes](label-freeze.json), and [measurement verification](measurement.json)
- [Group results and uncertainty intervals](summary.json)

## Exploratory spelling comparison

Within the joined or inflected category, 58/171 positions fired when “water” began the word and 0/109 fired when a letter preceded it. This follow-up was chosen after examining the results. The groups also differ in tokenizer boundaries and vocabulary.

## Follow-up check of noun adjacency

During the next-word ablation, an audit of the 179 frozen modifier labels identified four coordinated phrases (IDs 785, 787, 934, 973) in which punctuation or a conjunction intervenes before the head noun. All four fired. Restricting the modifier group to the 175 directly adjacent cases gives 153/175 firing (87.4%), compared with 157/179 (87.7%) under the frozen labels. The [follow-up ablation](../2026-11-21-water-modifier-ablation/README.md) reports both sets.
