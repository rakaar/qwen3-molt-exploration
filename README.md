# Qwen3-4B MOLT exploration

Results, figures, and code from exploratory Qwen3-4B experiments. The original 29 September session is retained below; new intervention experiments were run on 10–15 October 2026.

**[Explore the four selected results](https://rakaar.github.io/qwen3-molt-exploration/)** — large → cold, spider → ant / eight → six, passage-grounded abstention, and refusal despite authorization. Each panel links to its full report. [Earlier research notebook](exploration.html).

## Privacy removal controls — published 20 October 2026

- [Output-only zero clamp](results/2026-10-15-privacy-zero-clamp/index.html) and [user/query timing follow-up](results/2026-10-15-privacy-query-clamp/index.html): jointly removing the estimated contributions of L11–T1596, L12–T1891 and L13–T921 preserves the exact refusal on the original unauthorized Bob/Alice prompt. All three timing conditions make genuine edits where gates are active; no synthetic code is disclosed. [The privacy overview now includes this reverse-direction negative result](results/2026-10-15-privacy-refusal-overview/index.html#zero-clamp). This is a different prompt from the positive authorized API-key intervention.

## Abstention experiment directory

**[All abstention experiments: descriptions and links](https://rakaar.github.io/qwen3-molt-exploration/results/abstention/index.html)** — 21 entries covering the native baseline, L19–T368 heatmaps, fixed-gate and additive steering, strength searches, all three token scopes, false-abstention controls, confidence/uncertainty searches and related qualification/refusal studies. Includes negative results and links to raw measurements and source in each report.

- [Three confidence-wording candidates on native 4/4/4 QA cases](results/2026-10-19-confidence-gate-activation/index.html): all three gates are zero across the twelve QA prompts and responses, but fire in a separate confidence-wording control. No useful answer-confidence score was established.
- [Seven uncertainty candidates on the same native 4/4/4 cases](results/2026-10-19-uncertainty-gate-activation/index.html): four gates activate in all four natural abstention responses and none of the eight correct/made-up responses. The priority three are zero on question tokens and the final input position. This is response-wording association, not a validated pre-answer confidence measure or a confidence–steering-strength correlation.

## New report: 19 October 2026

- [Where is CH in child? — feature absorption investigation](results/2026-10-19-ch-absorption-overview/index.html) — a silent layer-3 CH gate and active Child gate suggested absorption, but Child ablations preserve spelling and a trained layer-3 probe fails on child. A complete 36-layer activity scan and 410 active feature-position ablations find a later CH contributor; single and joint input-word removals still preserve child’s answer. Includes the seven-feature heatmap, exact tasks, probe validation failure, format-versus-letter distinction, all seven stages of code/results, and backup hashes. **Absorption remains unconfirmed; localization is unresolved.**

- [Complete twenty-question abstention comparison](results/2026-10-19-abstention-complete/index.html) — ten originally correct answers and ten selected made-up answers, each tested with input-only, output-only and input-plus-output steering of L19–T368. At one shared input-only addition **+16**, all ten made-up answers become coherent abstentions; two of the ten originally correct answers also withhold. All twenty responses remain coherent. Per-question tuning gives **10/10, 7/10 and 10/10** conversions respectively. Includes all **1,176** positive responses, the common fourteen-strength grid, eleven strength trajectories, exact additive method, false-abstention controls, historical raw inputs and full new token traces. [Code and limitations](experiments/passage-uncertainty/abstention-complete/README.md).

**Historical resource update — 16 October 2026:** SPAR instance **56070285 and its attached 30 GB disk have been removed at the user’s request**. Fresh account checks show **zero instances, zero standalone volumes and $0/hour ongoing compute/storage billing**. All 189 remote scientific/runtime files were hash-verified and backed up before removal. [Teardown and backup proof](results/2026-10-16-vast-teardown/README.md). Earlier retained-GPU records are historical.

## New report: 16 October 2026

- [L19–T368 activates during three safety refusals](results/2026-10-15-safety-refusal-gates/index.html) — full native answers and interactive token heatmaps for three fixed requests about explosives, unauthorized Wi-Fi access and poisoning. **3/3 responses refuse and show gate activity.** Includes shared-scale earlier abstention references, exact token values, PNG/SVG exports, raw measurements and pinned provenance. Native generation is unmodified; an observation-only replay matches every next-token argmax. Descriptive activation evidence, with no causal or prevalence claim. [Source and method](experiments/safety-refusal-gates/README.md).

## New experiment: 15 October 2026

- [L11–T1596: refusal despite authorization — natural evidence, causal result and poster](results/2026-10-15-privacy-refusal-overview/index.html) — three matched synthetic API-key permission pairs, measured token heatmaps and means, then complete native-versus-steered answers. Fixing L11–T1596 (Data Security and Privacy Protection) to **16**, with L13 untouched, induces coherent key-free refusal in all **3/3 authorized service variants** of one prompt template. Natural activation begins after refusal has already started. Includes exact gate-replacement formula, edit positions, strong-intervention caveat, all 114 sweep/control responses, source, raw traces and poster. [Source](experiments/privacy-refusal-overview/README.md).

- [Combined abstention story and research posters](results/2026-10-15-abstention-overview/index.html) — goal, SQuAD2.0 passage-grounding method, three native made-up versus three native abstention token heatmaps with response averages, all ten causal comparisons, the fluent-but-wrong bat example, and the exact strength-search procedure. **7/10 clear conversions** across selected baseline failures. Explicitly describes additive current-input low-rank steering (not an absolute gate replacement) and coarse sweeps with follow-ups (not strict binary search). Includes an image-generated visual summary and a precise data-rendered PNG/PDF poster. [Source and provenance](experiments/passage-uncertainty/abstention-overview/README.md).

- [Six additional passage-grounding failures: four clear abstention conversions](results/2026-10-15-six-new-abstentions/index.html) — screen 80 questions, freeze six clear passage-unsupported answers before steering, then test L19–T368 additions with unfrozen attention. **4/6 new clear successes; 7/10 combined** with the earlier four, using question-specific strengths. Successful tested additions: +16 (carbon monoxide), +4 (ship cargo), +16 (economic mobility), +8 (antigen binding). Poland has ambiguous near-misses; respiration retains an answer despite uncertainty. The respiration baseline is true external knowledge absent from the passage. Includes all 80 steering/control responses, 12 exact native/zero matches, 3,335 verified edits, full screening review and three scatter panels. [Source and methods](experiments/passage-uncertainty/six-new-cases/README.md).

- [Four original made-up answers: replay and vampire-bat strength search](results/2026-10-15-vampire-intermediate/index.html) — the three earlier coherent, grounded abstentions reproduce exactly at +24 (crystallization), +16 (minerals), and +4 (electric eel). **No clean vampire-bat abstention at 128 distinct strengths from 0.5 to 64 in steps of 0.5**. Complete four-case original/steered comparison and every failed attempt; 145 generations, 10 exact native/zero controls, 13 exact positive prior anchors, and 7,314 verified edited positions. Attention remains unfrozen. [Source and methods](experiments/passage-uncertainty/vampire-intermediate/README.md).

- [L2-matched active MOLT controls](results/2026-10-14-active-small-hot-controls/index.html#norm-matched) — dedicated figures compare 13 alternative pairs in both sign directions with the selected pair at the same intended L2 norm. Every feature ID, available dashboard label and actual applied norm is shown. Selected P(cold): **99.8653%**; best paired control: **0.000000655%**. Includes a separate figure for 24 single-feature controls and a 53-row audit table with coefficients, source gates, intended/actual norms and CSV/JSON downloads. CPU-only replot of saved trials; no new GPU inference. [Source](experiments/active-small-hot-controls/plot_norm_matched.py).

- [Original passage questions: three scatter panels](results/2026-10-15-passage-original-group-sweep/index.html) — complete additive 4, 8, 16, 32, 64 sweep on all ten original Geology/Amazon unanswerable questions, with fixed baseline cohorts of four unsupported answers and six withholdings. Three scatter-only panels show coherence, unsupported target answers, and abstention decisions. At 16: 20% unsupported answers and 100% coherence; at 32: 20% unsupported answers with 80% coherence and worse explanations; at 64 every output collapses. Includes all 70 recorded responses, source/tensor hashes, 20 exact native/zero matches and 14 prior positive anchors. [Source](experiments/passage-uncertainty/original-group-sweep/).

- [Fresh Harvard/pharmacy screen](results/2026-10-15-passage-fresh-group-sweep/index.html) — ten audited questions with a clean baseline of nine appropriate withholdings and one unsupported negative answer. The one-case conversion cohort does not become a clean abstention at 4–32. Explanation grounding worsens at 16/32; all ten outputs collapse at 64. Preserved as an exploratory screen. [Source](experiments/passage-uncertainty/fresh-group-sweep/).

- [Higher additive strengths on the three remaining questions](results/2026-10-15-passage-additive-high-strength/index.html) — coefficients 8, 16, 24, 32. Minerals abstains at 16–32; crystallization at 24/32; vampire responses remain false or mixed. All 12 positive responses are fluent, with explanation issues separately flagged. [Source](experiments/passage-uncertainty/additive-high-strength/).

- [L19–T368 additive abstention sweep](results/2026-10-15-passage-additive-gates/index.html) — retain the native contribution and add coefficients 3–8, including 3.5, at every processed output-token position on the same four unsupported-answer cases. Only the electric-eel question abstains at every tested strength (1/4); no additional question converts through 8. All 28 positive-addition responses remain fluent and reach EOS. At 7, the vampire response echoes the question and is recorded separately from abstention. Includes complete response tables, the earlier-clamp comparison, 892 verified nonzero edited positions, raw traces, and native/zero controls. [Source](experiments/passage-uncertainty/additive-gates/).

- [L19–T368 forced-gate abstention sweep](results/2026-10-15-passage-forced-gates/index.html) — force absolute gate activations 0.5–3.5 at every processed output-token position on the four previously unsupported-answer cases. Gate 3 and 3.5 yield coherent abstention for the electric-eel question (1/4 cases); the other three still receive unsupported answers. All 28 positive-gate responses remain coherent and reach EOS; eight native/zero controls reproduce the saved baseline exactly. Includes full response tables with strengths as columns, grading explanations, nonzero-edit verification, and every gate/delta trace. [Source](experiments/passage-uncertainty/forced-gates/).

## New experiment: 14 October 2026

- [Active MOLT controls for small → hot](results/2026-10-14-active-small-hot-controls/index.html) — 35 distinct active control transforms, 400 primary conditions, and all primary edits at layer 22, token 9 with unfrozen attention. All 400 controls change final logits; the selected pair gives **99.8653% P(cold)** versus a maximum of **0.1846%** among sampled active controls. Includes strength sweeps, norm-matched controls, feature names and gate values, 790 total records, lossless logits, and independent verification. [Source and methods](experiments/active-small-hot-controls/README.md).

- [L19–T368 answer-token gate heatmaps](results/2026-10-14-passage-answer-gates/index.html) — observational replay of all 10 unanswerable-question responses from the clean baseline. All four unsupported answers have zero gate across their 87 visible answer tokens; all six abstention decisions have some active tokens (68/174 total, peak gate 3.481079). Includes shared-scale PNG/PDF heatmaps, decoded tokens, every gate and contribution norm, and exact cached-greedy replay checks. No intervention was performed; this is association on a small selected sample. [Source](experiments/passage-uncertainty/answer-gates/).

- [Two-passage clean grounding baseline](results/2026-10-14-passage-grounding-baseline/index.html) — native Qwen3-4B, no MOLT intervention, two passages with five answerable and five unanswerable questions each. All 10 answerable questions were answered correctly (0% wrong answer rate); 4/10 unanswerable questions received unsupported answers (40% hallucination rate under the passage-only criterion). Full passages, all 20 complete model answers, nuanced manual judgments, tokens, hashes, and reproducible code are saved. This is a small selected pilot, not a representative model score. SPAR GPU 55946393 remains running at the user's request. [Source](experiments/passage-uncertainty/small-baseline/).

## New figures: 14 October 2026

- [Actual random MOLT control feature lists](results/2026-10-14-random-control-features/index.html) — two complete images identifying every selected layer-22 MOLT and its saved dashboard name, with 100 trials on each side. Includes single additions/subtractions, paired interventions and reversed signs, zoomable SVGs, searchable text, CSV, and provenance checks. This reformats the 13 October records; no new inference or GPU rental.

## New experiments: 12–13 October 2026

**Resource update, 13 October:** the completed experiment GPU (SPAR instance 55734307) and attached disk were deleted after verified backup. Live SPAR inventory has zero instances and zero standalone storage volumes: **$0/hour ongoing compute/storage billing**. [Backup audit and teardown proof](results/2026-10-13-vast-teardown/README.md). Historical runtime snapshots describe the earlier execution state.

- [13 October: layer-22 small→hot random-MOLT and whole-MLP controls](results/2026-10-13-small-hot-controls/index.html) — 418 recorded conditions with unfrozen attention. The selected T2339/T941 pair reaches **99.8653% P(cold)**. All 100 uniform random MOLTs are inactive at the source tokens; all three available distinct active alternative pairs also fail to flip to cold, including one at a similar correction norm. Exact donor MLP-output replacement at token 9 gives **94.1706% P(large)** and effectively zero P(cold). [Executed code and raw audit](experiments/small-hot-controls/) · [Scatterplot](results/2026-10-13-small-hot-controls/controls-scatter.png). GPU and disk subsequently deleted; zero ongoing billing verified.
- [13 October: L19–T368 extended strength and coherence sweep](results/2026-10-13-medical-disclaimer-strength-sweep/index.html) — 72 new generations with unfrozen attention, including a closer sweep between −16 and −32. Factor −30 fluently affirms qualification for general guidance while explicitly retaining medical diagnosis/treatment limits; −29.5 still denies qualification with formatting damage. The prescription prompt instead collapses into repetition at −29.5. Complete outputs, token/gate traces, assessments, and checksums are included. The SPAR GPU and disk were subsequently deleted after verified backup.
- [13 October: L19–T368 negative-factor-16 follow-up](results/2026-10-13-medical-disclaimer-strength16/index.html) — 24 matched conditions and 24 clean identity generations on the original six prompts, with unfrozen attention. Factor −16 did not yield a coherent refusal-to-answer switch on five active prompts; qualification denial remained, and four active responses had malformed openings. The inactive control was unchanged. All clean continuations match the previous rental. The working SPAR GPU was subsequently deleted after verified backup.
- [13 October: L19–T368 medical disclaimer pilot](results/2026-10-13-medical-disclaimer/) — 24 prompts screened, 133 exploratory conditions and 96 matched comparative generations with unfrozen attention. Removing or negatively steering the estimated component did not produce the requested refusal-to-answer switch. Ordinary symptom prompts were already answered and had zero target activation. [Detailed report](results/2026-10-13-medical-disclaimer/index.html) · [Gate figure](results/2026-10-13-medical-disclaimer/clean-gate-positions.png).
- [13 October: explicit ant donor → implicit spider receiver](experiments/explicit-ant-implicit-spider/) — one MOLT, L3–T2309, donor token 8 (` ant`) to receiver token 12 (` webs`), with unfrozen attention. Adding the full donor contribution at 24× reaches **99.997842% P(next token = 6)**; 32× reaches **99.999976%**. 24× is the lowest successful tested strength, not a proven minimum. [Detailed report](results/2026-10-13-explicit-ant-implicit-spider/report.html) · [Summary image](results/2026-10-13-explicit-ant-implicit-spider/molt-l3-t2309-cross-prompt-summary.png).
- [12 October: spider/ant single-layer MOLT pilot](experiments/spider-ant/) — 927 main trials and 60 follow-up controls across eight candidates. Includes L8–T2201, implicit donor at the final `ills` subtoken of `anthills`, added to receiver ` webs` at strength 16; and the separate L3–T2309 explicit-spider/explicit-ant diagnostic at strength 32. [Detailed report](results/2026-10-12-spider-ant/report.html) · [L8 image](results/2026-10-12-spider-ant/molt-l8-t2201-patch-summary.png) · [L3 explicit image](results/2026-10-12-spider-ant/molt-l3-t2309-patch-summary.png) · [Shared L27/L31 gate figure](results/2026-10-12-spider-ant/shared-arthropod-gates-L27-L31.png).
- [12 October: MOLT/transcoder residual similarity](experiments/residual-similarity/) — fixed MOLT strength 4.5 compared with 72 transcoder strengths. At transcoder strength 19.5, both give over 99% P(cold), with layer-22 token-9 full-residual cosine **0.634857913** and changes-from-clean cosine **0.608773508**. [Detailed report](results/2026-10-12-residual-similarity/report.html) · [Figure](results/2026-10-12-residual-similarity/residual-cosine-vs-strength.png).

Raw tensors, full-vocabulary logits, controls, pinned protocols, source hashes, verification records, reports and figures are preserved. [Backup scope and fresh-clone verification](BACKUP-2026-10-13.md) explains how to reassemble the largest logits file from checked chunks. These are exploratory prompt-specific measurements; semantic concept replacement remains unverified.

## New experiments: 10–11 October 2026

- [10 October: matched transcoder and MOLT manipulation](experiments/small-hot-matched/) — executed code, pinned preparation, 205 trials, 78 generations, and independent tensor audit. Includes both constrained execution and native execution with unfrozen attention.
- [11 October: single-layer-22 transcoder strength search](experiments/layer22-transcoder/) — 463 trials with unfrozen attention. Two features reach **99.75234% P(cold)** at 19.5×; six reach **99.80232%** at 4.5× and **99.92866%** at about 6.10417×. The same-run two-transform MOLT control reaches **99.82693%** at 4.5×.
- [Interactive matched report](https://small-hot-molt-experiment.rakaar.chatgpt.site/) · [Layer-22 follow-up](https://small-hot-molt-experiment.rakaar.chatgpt.site/layer22-transcoder/) · [Two-feature explanatory image](results/2026-10-11-layer22-transcoder/two-feature-intervention.png).

The new outcome is next-token probability on the original prompt, not accuracy across prompts. Both scientific runners preserve the executed computation; added preparation helpers regenerate ignored selected-weight assets. Read each experiment's README and saved runtime before rerunning.

## Original session: 29 September 2026

**[Open the original HTML results summary](https://rakaar.github.io/qwen3-molt-exploration/exploration.html)** · [Downloadable summary](exploration.html)

The HTML is self-contained: its plots and gate-explorer data are embedded, so a downloaded copy works offline.

## Findings

- **Reconstruction:** released MOLT layers 17 and 25 reduced standardized squared error against the saved-mean baseline by 73.43% and 80.59%, respectively, on 64 held-out UltraChat `test_sft` conversations (58,424 tokens).
- **Full layer-17 replacement:** the model retained the same numerical/geographic answers in three qualitative questions, and answered the Dallas → state → capital question with Austin. This is not a broad capability evaluation.
- **Texas-labelled transcoder features:** eight of 73 retrieved candidates activated at the Dallas token. The search was not exhaustive; activation alone is not a causal graph.
- **Published small → hot intervention:** the unchanged “opposite of small” prompt switches its top next token from `large` to `cold`. At strength 4.5, P(cold)=99.774%, compared with the community reference's 99.822%.
- **MOLT gate response:** at the `small` token, the combined intervention changes layer 21 from 4 to 5 active gates and layer 22 from 3 to 5. Gate identities change as well as counts. Layers are zero-based.

**Not established:** Qwen MOLT Jacobian fidelity, a causal MOLT equivalent of the transcoder intervention, an unrestricted intervention result, or a validated multihop attribution graph.

The feature intervention freezes attention patterns and pins MLP outputs through layer 22 to their baseline plus decoder edits. MOLTs at layers 21/22 were passive observers of captured inputs; they did not drive the answer. The model's MOLT inputs are before post-attention normalization, as required by Georg's checkpoints.

## Contents

| Experiment | Report | Raw data |
|---|---|---|
| Held-out reconstruction | [Report](results/reconstruction-layers17-25/report.md) | [JSON](results/reconstruction-layers17-25/reconstruction.json) |
| Three-question layer replacement | [Report](results/layer17-three-questions/report.md) | [JSON](results/layer17-three-questions/comparison.json) |
| Dallas response | [Report](results/dallas-one-word/report.md) | [JSON](results/dallas-one-word/comparison.json) |
| Texas feature activations | [Report](results/texas-features/report.md) | [JSON](results/texas-features/activations.json) |
| Dallas MOLT gates | [Report](results/dallas-molt-gates/report.md) | [JSON](results/dallas-molt-gates/gates.json) |
| Small-to-hot reproduction | [Report](results/small-hot-reproduction/report.md) | [JSON](results/small-hot-reproduction/reproduction.json) |
| Layers 21/22 MOLT comparison | [Report](results/small-hot-molt-gates-21-22/report.md) | [JSON](results/small-hot-molt-gates-21-22/comparison.json) |

## Resume on a future GPU

No credentials, SSH details, or GPU lifecycle commands are included. Choose and authorize a GPU separately. The original run used an RTX 5090, Python 3.12.14, PyTorch 2.11.0+cu128, and Transformers 4.57.6. See the recorded [environment](results/reconstruction-layers17-25/environment.txt); it is a provenance snapshot, not a portable lockfile.

```bash
git clone https://github.com/rakaar/qwen3-molt-exploration.git
cd qwen3-molt-exploration
python -m venv .venv
source .venv/bin/activate
# Install a PyTorch build appropriate for your GPU. Original CUDA wheel:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/Goreg12345/crosslayer-transcoder.git
git -C crosslayer-transcoder checkout 2611214e6cf46cd96d4d689bcee369dd4fb27b30
python download_models.py
```

For the small-to-hot reproduction and gate comparison:

```bash
python prepare_small_hot.py
python reproduce_small_hot.py
python fetch_molt_gates_21_22.py
python compare_molt_gates_21_22.py
python report_molt_gates_21_22.py
```

This downloads the base model, about 2.9 MB of selected transcoder data, and about 52 MB of MOLT gate data. It does not require full MOLT transform matrices. The summary and existing measurements can be inspected without a GPU.

For full MOLT reconstruction and Dallas experiments:

```bash
python download_layers_17_25.py
bash run_reconstruction.sh
python compare_layer17_responses.py
python compare_dallas.py
python fetch_texas_encoders.py
python measure_texas_features.py
python measure_dallas_molts.py
python plot_texas_features.py
python plot_dallas_molts.py
```

The two full MOLT checkpoints add about 8.44 GB. Reconstruction activations can be regenerated from the pinned dataset and sample seed; they are not in Git. A compact raw-results/code archive was also backed up locally before GPU teardown. The user chose to regenerate the large reconstruction activation cache later; that cache and redownloadable model weights are intentionally excluded. Small prompt activations and gate tensors are preserved.

For a CPU-only rebuild of the HTML:

```bash
python build_summary.py
python -m http.server 8000
```

Open `http://localhost:8000`. Generating the HTML uses only the Python standard library and existing images.

## Provenance and attribution

- [MOLT article](https://transformer-circuits.pub/2025/bulk-update/index.html)
- Base model: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`.
- [Georg Lange's MOLTs](https://huggingface.co/georglange/qwen3-4b-molt): `eed10292a96521837f5d65b36fdfa3bf5bea0037`.
- [Georg's training code](https://github.com/Goreg12345/crosslayer-transcoder/tree/2611214e6cf46cd96d4d689bcee369dd4fb27b30): unchanged model/normalization implementation.
- [Hanna's transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders): `94d176260ac39ce2f882b8b09aba8c118df29bb3`.
- [Community intervention notebook](https://github.com/wdk0082/llm-circuits/blob/af8a37ad836dc401a0ec5dbb71307d8d7ab68028/notebooks/multilingual.ipynb): feature IDs, strengths, and steering implementation. This is not an Anthropic-released Qwen experiment.

Upstream MIT license notices are preserved in [licenses/](licenses/). Model weights retain their original repository terms. `small_hot_upstream.py` contains attributed upstream helpers; function hashes are recorded in the reproduction results.

Published experiment scripts retain the original computation. Their workspace roots were changed from the temporary GPU path to the script directory. The reconstruction runner was adjusted to regenerate an absent activation cache even when a saved result manifest exists. Plot/report builders only read saved results. A fresh GPU rerun of this packaged version has not been performed.

Historical per-experiment reports describe the state at execution time. Consult `archive-status.json` for the final backup and GPU lifecycle status. Some JSON includes pre-run recipe status text nested within provenance; measured run fields and controls are the authoritative outcome.
