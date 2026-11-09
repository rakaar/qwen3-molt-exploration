# L29–T2375 explanation refinement

**10 November held-out update:** the [complete same-judge comparison](../2026-11-10-t2375-detection-fuzz-heldout/README.md) is now available. Detection changed from 79/100 to 80/100 (1 improved, 0 worsened); Fuzz changed from 78/100 to 80/100 (2 improved, 0 worsened). Both descriptions stayed frozen, all 100 cases per method were included, and the cached judge/settings match the local Active Fuzz run. The larger held-out evaluation shows modest gains, rather than the large percentage-point gains seen in the small training diagnostic.

**Later update, 9 November:** a [fresh local-judge Active Fuzz comparison](../2026-11-09-t2375-active-fuzz-local-judge/README.md) is now complete on the same 100 saved held-out cases. Both descriptions were rerun: original 78/100, revised 79/100, with two corrections and one regression. This is a separate judge build/runtime and does not replace the historical dashboard score or establish an Active Fuzz improvement.

The revised water + eleven-boundary explanation improved two fixed **training diagnostic** scores. This is an initial paired test, not a new held-out benchmark result or an explanation of the transform's causal computation.

| Method | Original explanation | Revised explanation | Change | Paired outcomes |
|---|---:|---:|---:|---|
| Detection | 26/32 = 81.25% | 29/32 = 90.625% | +9.375 percentage points | 3 corrected, 0 regressed |
| Fuzz | 26/32 = 81.25% | 28/32 = 87.5% | +6.25 percentage points | 2 corrected, 0 regressed |

There were no parse failures or false positives in either arm. Each method used 16 activating and 16 inactive cases, so accuracy equals balanced accuracy. The sample is small; no claim of a statistically established generalization gain is made.

## Revised explanation

**Water and eleven-context boundaries**

> Activates on "water"/"Water" across lists, ingredients, and environmental descriptions, and on tokens at the boundary before eleven-related continuations. The second pattern includes contexts eliciting 11/eleven, November as the eleventh month, K as the eleventh English letter, or XI as the eleventh positive integer. Activation may occur on a preceding word, separator, or assistant-prefix token rather than the number itself; it is wording-dependent and sometimes appears for neighboring ordinals.

The 68-word text was frozen before seeing T2375's scoring-case text or new judge outputs. Its SHA-256, calculated over the stripped UTF-8 text, is `dc149e8b122d8c553276c5807972a529ab4e5495d246363cd16463e6cbf6c00f`. The draft uses the inherited water explanation and our independent ordinal experiments. It does not claim the gate is exclusive to eleven, that water and eleven share a mechanism, or that this describes what `UVx` computes.

## What improved

Detection newly recognized the activating contexts containing **“There were 11 lurkers”**, the transition into recipe step **11**, and **“Digital watercooler”**. Fuzz newly accepted the highlighted punctuation/newlines immediately before recipe step 11 and the `water` subtoken of `waterfalls`.

Token precision remains incomplete: Fuzz still rejected the activating space before `11` in the lurker example. Both methods still missed the `2011` census context, a Spider-Man `#101` context, and a Chinese Maze Murders context. These are residual errors, not evidence that adding those phrases indiscriminately would be a good explanation.

## Matched comparison

- Source: [dashboard manual investigation, layer 29 feature 2375](https://amsterdam.taild80f17.ts.net/tools?feature=2375&layer=29), authenticated.
- Tool: `score_explanation`, advertised model `qwen3.8-27b-nvfp4`.
- Both arms have identical scorer runtime, cache, vocabulary, selection and manifest hashes.
- The same 16 positive and 16 negative training-window identities were returned in both arms. All 32 contexts, labels, highlight selections and gate arrays matched within each method.
- Windows contain up to 64 tokens, aligned to conversation boundaries in the current manual tool. This differs from the original campaign's peak-ending evaluation windows.
- All Qwen added-token strings are displayed as ordinary text in both arms.
- One frozen candidate was tried. No rewrite was made after its feedback.
- The canonical dashboard explanation and historical scores were not overwritten. No new GPU was rented or dashboard volume modified.

The [reduced result extraction](training-comparison.json) records metrics, paired checks, corrected window identities and server-side diagnostic IDs. It is not a full copy of the raw audit. Full result and audit files were saved by the dashboard under its `molt-manual-playground/layer-29/training-scores/2375` directory, named by the two diagnostic IDs.

## Historical held-out scores remain separate

| Method | Historical original explanation | Revised explanation under the original judge deployment |
|---|---:|---|
| Detection | 79/100 = 79% | Not run |
| Fuzz | 77/100 = 77% | Not run |
| Active Fuzz | 77/100 = 77% | Not run |
| Intruder | 3/20 = 15% | Explanation-independent; not rerun |

Intruder does not receive an explanation, so a textual rewrite cannot directly change its predictions. The manual tool exposes only Detection and Fuzz; it cannot establish an Active Fuzz improvement.

We downloaded the authenticated original feature JSON containing all labels and the manifest. We also retrieved all 270 published request attempts for T2375 from dataset revision `5160cb8fb34db41545e8ceabb3fb1ea047b519e4`. Every one of the 300 binary result rows matches an exact published request; cached/shared positives explain why request counts differ from decision counts. The stored predictions reproduce the historical confusion counts and accuracies.

`heldout_replay.py` has prepared the paired 300-case comparison offline. It will refresh both explanations under the same judge and escape special-token strings in both arms, rather than compare a new score directly against a differently rendered historical run. There are 500 unique arm/request combinations after deduplication. Intruder is excluded because it does not consume the explanation. This exact original-judge replay has not run; the later local-judge Active Fuzz comparison is documented separately above.

The historical judge hostname no longer resolves. The working manual tool reports a server-local API at `127.0.0.1:8000/v1`; its external port was not reachable. A reachable judge URL/tunnel is needed to execute the held-out comparison. The persistent Vast dashboard volume has published assets and weights, but its saved inventory does not contain the original per-case scorer audits. Renting storage-attached compute would not by itself restore judge access.

## Live Vast volume inspection — 9 November 2026

With a separate $0.50 cap and explicit temporary-instance cleanup authorization, we rented instance **59766907** on the dashboard volume's physical host. Its approved rate was **$0.1096296/hour** including an 8 GB container disk. Volume **58518928** was mounted at `/data`; the stored dashboard was verified at `/data/molt-dashboard`. The SSH host fingerprint matched the instance's provider-supplied startup log.

A read-only walk of the entire mounted volume checked **102,616 file names**, with **zero search errors** and **zero `scorer-runtime.json` matches**. The dashboard application and model inference were not started. The inspection code and output were written only to the temporary container's `/tmp` directory. Its three output/source files were downloaded and independently checked against remote SHA-256 digests.

The temporary instance and its container disk were then deleted. Live provider lists confirmed that instance 59766907 was absent and that volume 58518928 remained present and unattached. Temporary-instance billing is now zero; the preserved dashboard volume retains its existing storage charge. The separate experiment instance 58952920 was not changed.

No runtime file was found in the other existing experiment instance or the relevant local SPAR workspace either, and no matching Amsterdam/Geor/New York SSH alias was configured locally. **The full three-method original-judge replay has not run.** An exact replay of that deployment requires its checkpoint/runtime or reachable endpoint. A fresh paired comparison does not require that endpoint: the later local-judge Active Fuzz run above uses offline vLLM inference.

Inspection receipts and verified backups: `gpu-rentals/2026-11-09-t2375-heldout/`. The CLI did not emit a structured deletion receipt; deletion was established independently using fresh instance and volume lists, not inferred from the command's return alone.

## Sources and code

- Current GitHub code was checked on the **MOLT branch**, not `master`. The [setup script](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/setup_molt_autointerp.sh#L6) pins **EleutherAI/Delphi** to the revision above. The [full scorer implementation](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/run_molt_layer_explanations.py#L322) implements Intruder, Detection and both fuzz variants; its [default model](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/run_molt_layer_explanations.py#L645) is Qwen3.8-27B-NVFP4. [Runtime overrides](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/tools/molt_scorer_runtime.py#L9) load `base_url`, `model` and `context_limit` from a separate output-directory `scorer-runtime.json`. Identifying the framework/model from GitHub does not establish access to that live server. See `github-source-verification.json` for fetched blob identities.
- [Published explanations and metrics](https://huggingface.co/datasets/georglange/qwen3-4b-molt-interpretations/tree/5160cb8fb34db41545e8ceabb3fb1ea047b519e4)
- [Authenticated original feature and scores](https://amsterdam.taild80f17.ts.net/layers/layer-29/autointerp-layer/features/2375.html)
- [Public different-token and residual-stream findings](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-08-eleven-token-confound/)
- Local code: `experiments/t2375-explanation-refinement-20261109/`.
- Local full labeled feature and manifest: `comparisons/2026-11-09-t2375-explanation/original-campaign/`.
- Local prepared requests: `heldout-plan.json`; this is preparation, not a scored result.
