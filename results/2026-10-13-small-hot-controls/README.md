# Layer-22 small→hot MOLT controls — 13 October 2026

The selected two-MOLT edit reproduces **99.865258% P(cold)** on the unchanged receiver prompt. Uniform random controls are all inactive at the selected source tokens; active alternatives are more informative. Exact whole-MLP donor replacement does not flip the answer.

## Prompts, position, and edits

Receiver: `What is the opposite of "small"? Reply with only the word, nothing else.`

Donor: `What is the opposite of "hot"? Reply with only the word, nothing else.`

Layer and positions are zero-based. The edited position is **9 in the full chat sequence**: `small` in the receiver, `hot` in the donor. Both inputs have 29 tokens; next-token logits are read at position 28. Thinking is disabled in the standard Qwen chat template.

For feature f, define its clean contribution:

```text
z = (pre-normalization residual after layer-22 attention - input_mean) / input_std
C_f(x) = gate_f(z) × ((z @ V_f) @ U_f) × output_std

Selected edit:
native receiver MLP_output[22,9] +=
    -4.5 × C_2339(clean receiver)
    +4.5 × C_941(clean donor)

Whole-MLP control:
native receiver MLP_output[22,9,:] = native clean donor MLP_output[22,9,:]
```

T2339 is labelled **Small-Size Adjectives and Idiomatic Phrases**, rank 32. T941 is **Thermal State and Heat Action Gate**, rank 64. Labels describe gate activation, not necessarily the meaning of the low-rank output.

The model retains its native MLP in additive conditions. A full gate × low-rank output contribution is subtracted or added. U/V weights and attention are not modified. The whole-MLP control copies all 2,560 native output values at the one position, before residual addition; the receiver residual remains. Its replacement strength is 1, not 4.5. Every other directly written MLP position is unchanged.

## Results

P(cold) and P(large) are full-vocabulary next-token probabilities for lowercase token IDs 87072 and 16767. Uppercase variants are separate tokens; these probabilities are not normalized over two choices.

| Condition | P(cold), % | P(large), % |
|---|---:|---:|
| Clean receiver | 2.45892e-10 | 89.9155 |
| Clean donor | 97.0680 | 1.29566e-13 |
| Selected pair, −4.5 receiver T2339 +4.5 donor T941 | **99.8653** | 1.02481e-8 |
| Selected hot only, +4.5 donor T941 | 2.51770e-7 | 65.3821 |
| Selected small only, −4.5 receiver T2339 | 0.0158910 | 0.0807012 |
| Selected pair, reversed signs | 1.25862e-11 | 18.2028 |
| Exact native donor MLP-output replacement | **7.00034e-10** | **94.1706** |

There are **418 recorded conditions** in 60 batches. Fixed seed 20261013 selects 100 uniform IDs without replacement from 2,478 non-target MOLTs before intervention outcomes. Each ID is tested individually at +4.5 donor and −4.5 receiver strength. A shuffled derangement of the same IDs supplies 100 distinct-feature pairs, tested in both sign directions. All 400 uniform trials have zero edits because every sampled gate is inactive at both exact source tokens. They recover the clean logits exactly; this alone is weak selectivity evidence.

All other gate-active features were also tested. Receiver-active non-target IDs are T8 and T55; donor-active IDs are T8 and T71. All four applicable individual edits and all three distinct-feature active pairs were measured. None makes cold the top token, and none reaches 99% P(cold).

The alternative pair −4.5 receiver T8 +4.5 donor T71 has actual correction norm **63.0534**, versus **65.3121** for the selected pair, but P(cold) is only **7.01581e-13** (fraction). This is a useful similar-magnitude comparison, not an exact norm-matched control. Active alternative MOLTs have rank 512. Fixed strength does not imply equal perturbation size.

The original selected-pair result was 99.826932%. The new result differs by 0.038326 percentage points with a different CUDA build and batch size. Both reproduce the over-99% outcome. This remains a single-prompt experiment; the controls do not establish broad semantic selectivity or a general concept-replacement mechanism.

## Evidence and independent audit

- `protocol.json`: formulas, pinned revisions, position, and sampling procedure.
- `artifacts/selection.json`: all sampled IDs, pairings, and all 2,480 clean gates for both inputs, saved before intervention outcomes.
- `artifacts/conditions-before-execution.json`: all intervention specifications before model execution.
- `artifacts/results.json` / `conditions.csv`: every condition, source gate, intended/actual norm, probability, and top token.
- `artifacts/next-token-logits.npz`: exact BF16 logit bit patterns, deduplicated into 14 unique rows; `logit_row` maps each condition to a row. Decode with `(bits.astype(np.uint32) << 16).view(np.float32)`.
- `artifacts/direct-write.npz`: before/after output vectors, indexed separately by `vector_row`.
- `artifacts/contributions.npz` and `artifacts/clean-state.pt`: fixed contribution vectors and clean activation captures.
- `verification.json`: independent CPU softmax, vector algebra, BF16 rounding, random-seed reconstruction, exact whole-MLP replacement, identity/repeat checks, and file hashes. Maximum probability disagreement is 6.81e-8 in fraction units.
- `remote-export-checksums.json`, `archive-record.json`, and `transfer-verification.json`: verified download record.
- `prepare-and-run.log`: execution log. No cloud credentials are included.

All forward passes use eight identical prompts, seven experimental slots and a clean identity row. Batch clean logits match bitwise across the experiment. Upstream residual, normalized MLP input, and native MLP output are identical before every receiver intervention. The layer-22 residual-add boundary and unchanged direct writes outside token 9 are checked. Ordinary attention executes throughout: layers up to 22 are unchanged, while downstream attention responds to the intervention.

## Reproduce

Use a fresh copy of this folder for a new inference run: preparation overwrites saved clean artifacts. A 24GB RTX 3090 was used. Install a CUDA-compatible PyTorch build separately; the measured run used Python 3.12.3, PyTorch 2.14.0+cu126, and Transformers 4.57.6, with BF16 model inference and FP32 MOLT algebra. These are recorded runtime versions, not a guarantee of bitwise reproduction across hardware.

```bash
pip install -r requirements.txt
python download_base.py
python prepare.py
python run.py
python verify.py
python report.py
```

`download_base.py` retrieves and hashes Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`. The compact public selected asset is included with SHA-256 `cff6543981e529a6185ba456031cd42e3c5f7dc9570d66bbdd3a57de80d5070e`. `prepare.py` verifies the pinned MOLT config/header and retrieves needed U/V tensors by checked HTTP ranges from georglange/qwen3-4b-molt revision `eed10292a96521837f5d65b36fdfa3bf5bea0037`. The measured control tensors are also preserved in `weights/control-uv.safetensors`. Base model weights are excluded.

CPU-only audit and report regeneration on the saved evidence need NumPy and Matplotlib:

```bash
python verify.py
python report.py
```

The executed `common.py`, `prepare.py`, and `run.py` are preserved byte-for-byte, and their hashes are recorded. `report.py` only reads results; plotting and website changes do not alter the experiment.

## Resource state

After completion and verified GitHub backup, the user authorized teardown. SPAR Vast instance 55734307 and its attached disk were deleted on 13 October 2026. The post-deletion account inventory contains zero instances and zero standalone storage volumes, confirming **$0/hour ongoing compute/storage billing**. See `teardown.json`; the retained `resource-state.json` describes the earlier execution-time snapshot. Provider connection helpers are excluded from the public package.
