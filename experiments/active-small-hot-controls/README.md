# Active-source MOLT controls for the small-to-hot edit

Replace the earlier inactive uniform controls with nonzero source contributions. The primary comparison holds the intervention at native layer 22, token 9, with unfrozen attention and the original MLP retained.

Receiver: `What is the opposite of "small"? Reply with only the word, nothing else.`

Donor: `What is the opposite of "hot"? Reply with only the word, nothing else.`

## Selection and conditions

- Source gates must be positive in the source used by the term: donor for additions, receiver for subtractions, both for pairs. Each tested actual MLP edit is nonzero. Target layer-22 transforms 2339 and 941 are excluded from control sampling.
- Layer 22 has two non-target receiver-active transforms (8, 55) and two donor-active transforms (8, 71), yielding three distinct pairs. Repeated strengths are conditions, not independent unique-feature draws.
- Seed 20261014 first selects ten other layers, then uniformly samples ten active donor features, ten active receiver features, and ten eligible distinct pairs from those layers. Thirty-five distinct control transforms are used in total.
- Primary: 100 donor additions, 100 receiver subtractions, and 100 pairs in both sign directions, totaling 400 conditions. For each recipe, half use layer-22 sources and half use other-layer sources. All primary edits are applied at layer 22, token 9.
- Another 100 conditions sweep the selected features; 50 match the selected edit's FP32 vector norm at layer 22. A separate native-layer arm retains 240 conditions for comparison. Total: 790 conditions.

Layer-22 single controls use 25 strengths from 0.5 through 64, including 4.5. Paired controls select 50 of the 75 pair/strength combinations with the fixed seed. Other-layer controls use 0.5, 1, 4.5, 16 and 32. All selection and original strengths were saved before intervention outcomes. The fixed-site arm was subsequently added to hold intervention location constant, using exactly the same source features and coefficients.

## Exact edit

Compute the complete contribution in FP32 from the clean source input at its native layer:

```text
z = standardized prenorm residual after attention
C = gate(z) * ((z @ V) @ U) * source output_std
delta = −strength * receiver C_A + strength * donor C_B
receiver native MLP_output[22,9] += delta rounded to BF16
```

Individual tests use one term. Reverse tests flip both signs. Neither U/V matrices nor model weights are changed. Other-layer controls transfer completed source contribution vectors to layer 22; a layer-22 input is not passed into an operator trained at another layer. The native-layer arm instead applies the contribution at its original layer.

Native layer-35 MLP-output edits at token 9 cannot affect the readout at token 28: no attention layer follows that MLP. These structurally unreachable native-layer cases are supplementary and are excluded from the primary fixed-site comparison.

## Results

| Primary comparison | Highest P(cold), full vocabulary |
|---|---:|
| Selected pair at 4.5 | 99.8653% |
| Layer-22 active control sources, strength sweep | 0.000747% |
| Other-layer active sources transferred to layer 22 | 0.184581% |
| Controls with matching FP32 edit norm at layer 22 | 0.000000655% |

All 400 primary controls changed the final full-vocabulary logit vector relative to the clean receiver. None produced `cold`, in either capitalization, as the top next token. The strongest primary control uses receiver L9–T809 (“Small Noun Phrase Gate”) and donor L9–T1348 (“Thermal State and Heating Mechanism Gate”), at strength 32.

This tests the sampled transforms, coefficient grids, and one prompt pair. It does not establish uniqueness over all possible transforms, gains, or prompts. Results are next-token probabilities, not judgments of full-generation coherence.

## Gates and precision

The compact feature list shows each transform's mean nonzero gate at token 9 across the clean receiver/donor, in brackets. Both source-specific gates and contribution norms are also shown. The detailed trial table gives the exact source gate for each term. These values are not averages over other tokens or scaled steering strengths.

The first execution used FP32 addition before rounding and failed the original reference check (99.90097% rather than 99.86526%). The same fixed conditions were rerun with the original BF16 write order. The corrected selected reference's entire BF16 logit vector matches the original experiment bit-for-bit. No feature or strength resampling followed that check.

Base Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT revision: `081ce901b090b64aeecf9bc8709b57c10411e32a`. The layer-22 whole-checkpoint SHA256 matches the original `eed10292...` release. Released BF16 standardizer tensors are decoded exactly to FP32 for arithmetic. Native model BF16, PyTorch 2.14.0+cu126, Transformers 4.57.6, eight-row batches with clean padding rows verified bitwise identical.

## Reproduce and saved evidence

From the project root, use the pinned local base checkpoint:

```text
python3 experiments/active-small-hot-controls/worker.py capture --model-dir /path/to/pinned/qwen3-4b
cp experiments/active-small-hot-controls/artifacts/clean-state.npz experiments/active-small-hot-controls/clean-state.npz
python3 experiments/active-small-hot-controls/prepare.py
python3 experiments/active-small-hot-controls/worker.py run --model-dir /path/to/pinned/qwen3-4b
mv experiments/active-small-hot-controls/artifacts experiments/active-small-hot-controls/native-run-artifacts
mkdir experiments/active-small-hot-controls/artifacts
cp experiments/active-small-hot-controls/clean-state.npz experiments/active-small-hot-controls/artifacts/clean-state.npz
python3 experiments/active-small-hot-controls/worker.py fixed_site --model-dir /path/to/pinned/qwen3-4b
```

Use an empty output directory for a new run and preserve each arm separately, as above. `prepare.py` generates both condition files deterministically and uses checked HTTP ranges rather than downloading all full MOLT checkpoints. Its label lookup expects the saved dashboard summaries in the project; descriptions do not affect feature sampling or interventions. Missing cached descriptions in layers 34/35 are marked explicitly; transform IDs and measurements are retained. The published selected weight slices and source states also allow replay without preparing them again.

The result folder contains PNG/PDF plots, the full feature list, every condition, signed coefficients, source gates, actual and intended edit norms, BF16 full-vocabulary logits in checked chunks, contribution parameters, source hashes, and independent CPU-float64 probability verification. `verify.py` checks source files, all chunk hashes, original-reference equality, and edit norms. The SPAR GPU remains running as requested.
