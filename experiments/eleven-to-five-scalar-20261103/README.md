# Scalar-only eleventh-to-fifth month sweep

Donor: `the eleventh month of the year is`; its natural L29–T2375 gate at final ` is` is 1.1585536003112793.

Recipient: `the fifth month of the year is`; its native gate is 0 and its initial answer is May. Only its final input token (position 6) is edited. The recipient operator UVx is recomputed from the native recipient state and remains identical across all strengths; neither donor operator nor donor full contribution is transferred. Generated tokens receive no edits.

The predeclared fixed gates are 0, 0.5, 1, the exact donor scalar, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80 and 90. The native model remainder is retained. The FP32 edit is `(fixed_gate - natural_gate) * recipient_operator`, followed by one BF16 cast at the selected local MLP output. Other local outputs are checked bit-exact. Greedy native Qwen3-4B uses the pinned checkpoints, eager attention, TF32 off and a 32-token continuation limit.

The exact donor scalar retains May. Gates 20, 30 and 40 produce an initial November followed by explicit self-correction back to May. Gate 20 is the lowest successful value tested for an initial-November criterion, not a global minimum; gates between 15 and 20 were not tested. There is no clean November-only answer in this sweep. Higher gates 50–70 begin with “the month of may”; 80–90 begin with “11” and continue into a question.

Native/observer and zero controls match exactly. Donor-scalar and maximum-gate repetitions match exactly. A separately logged follow-up reloaded the model and tested only 0, the donor scalar, and 20; all generated tokens and recorded full-vocabulary logit hashes match the primary runs. The additional source is under experiments/eleven-to-five-scalar-repeat-20261103. This is an adaptive confirmation, not additional predeclared strengths.

The task helpers scalar_remote.py and scalar_repeat_remote.py live under gpu-rentals/2026-11-02-eleven-activation and verify live SPAR instance identity before SSH. Both downloads and every preserved file are SHA-256 checked. They have no lifecycle cleanup commands. No GitHub publication or GPU stop/delete was performed.

Local reproduction after collecting the saved evidence:

```sh
python3 experiments/eleven-to-five-scalar-20261103/analyze.py
python3 experiments/eleven-to-five-scalar-20261103/plot.py
```

Results, all 17 continuations, probabilities, validation and the plot are in results/2026-11-03-eleven-to-five-scalar/. Scientific raw backups are in comparisons/2026-11-03-eleven-to-five-scalar/ and comparisons/2026-11-03-eleven-to-five-scalar-repeat/.

Reported May/November case sums cover two exact next-token renderings per month, with a leading space and lowercase/title case. They are not semantic probabilities summing every possible answer format. All four individual probabilities remain in summary.json.
