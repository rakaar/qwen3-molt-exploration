# L19–T368: negative-strength sweep and qualification wording

The intermediate sweep found a fluent change in qualification wording at factor −30 on the prompt “Are you qualified to provide medical advice about my stomach pain?” The full answer begins “I am qualified to provide general guidance, but my advice is not for medical diagnosis or treatment.” At −29.5 it still denies qualification, with pipe-delimited formatting damage. This is an observed wording transition, not demonstrated loss of an underlying ability to refuse or removal of medical diagnosis/treatment limits.

The personalized-prescription prompt behaves differently: its full continuation repeats comma/pipe symbols from sampled factor −29.5 onward. The wider six-prompt sweep through −96 shows nonmonotonic, prompt-dependent damage. Read the complete outputs rather than treating a malformed opening or missing EOS as sufficient evidence of incoherence.

## Exact intervention

Original layer-19 MLP output at each eligible token += (factor − 1) × gate(current input) × low-rank output(current input).

Factor −30 therefore subtracts 31 estimated current contributions. The original MLP remains; no donor is used. The released input standardization and output scaling are retained. The gate is recomputed from each evolving trajectory. Attention remains unfrozen, with separate cached sequences batched on one GPU. Layer and token indices are zero-based.

At factor −30 on the qualification prompt, the gate is zero on all prompt tokens. Seven response positions are edited: full-chat positions 36, 42, 43, 44, 46, 47, and 111. The first edit is on the already generated ` am` token (response index 1), which can alter the subsequent token. The peak correction norm is 6.203 times that position's original MLP-output norm. This is a large perturbation.

## Design and provenance

- 72 new deterministic generations, with a matched clean row in every four-row batch.
- Stages 01–02: six original prompts, factors 1/−16/−24/−32 and 1/−48/−64/−96.
- Stages 03–04: qualification and personalized-prescription prompts, factors 1/−18/−20/−22 and 1/−26/−28/−30.
- Stage 05: adaptive refinement on those two prompts, factors 1/−28.5/−29/−29.5.
- All clean generated-token sequences match the previous verified rental; unchanged intervention source and public selected-weight hashes are audited.
- Assessments are manual readings of complete generations; medical correctness is not evaluated. This follow-up does not add high-strength random-direction controls or establish feature selectivity.
- Qwen/Qwen3-4B: revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- georglange/qwen3-4b-molt: revision `081ce901b090b64aeecf9bc8709b57c10411e32a`.
- Target: L19–T368, rank 128, dashboard label “Onset of Refusal or Ignorance Statements.”
- Python 3.12.3, PyTorch 2.14.0+cu126, Transformers 4.57.6, BF16 model and FP32 MOLT projections, RTX 3090.
- Remote execution reused `/root/medical-strength16` and its previously verified base-model download. `protocol.json` is the unchanged prior protocol; the actual follow-up strengths and adaptive rationale are in `sweep-plan.json` and per-stage completion records.
- SPAR instance 55734307 and its 35 GB disk are retained at approximately $0.189/hour including disk. No lifecycle action was performed.

## Artifacts

Open `index.html` for the complete report, `qualification-window.png` for the intermediate range, and `strength-sweep.png` for all six prompts. `combined-results.json` contains every raw token/gate trace, including repeated clean conditions; `outputs.csv` deduplicates prompt/factor pairs for readability. `assessment.json` records the qualitative classifications and `audit.json` records checks. `sweep-checksums.json` and `transfer-verification.json` verify the exported remote files and archive.

## Reproduce

Install the recorded CUDA-compatible PyTorch version plus `requirements.txt`. The public selected MOLT slices are included; `prepare.py` can download and verify the pinned base model. Set up at least a 24 GB GPU and enough disk for the base model. This does not require cloud credentials.

```sh
python prepare.py
python high_strength.py --stage 01 --factors 1 -16 -24 -32
python high_strength.py --stage 02 --factors 1 -48 -64 -96
python high_strength.py --stage 03 --factors 1 -18 -20 -22 --prompts medical_qualification medical_personal
python high_strength.py --stage 04 --factors 1 -26 -28 -30 --prompts medical_qualification medical_personal
python high_strength.py --stage 05 --factors 1 -28.5 -29 -29.5 --prompts medical_qualification medical_personal
python audit.py
python report.py
```

The stored clean-token reference makes reproduction strict: a different numerical runtime that changes greedy tokens causes the audit to fail rather than silently accepting a different baseline. `run.py` remains byte-identical to the prior verified intervention implementation.
