# Fifth-element scalar-gate sweep

Prompt: `the fifth element in the periodic table is`. Feature: L29–T2375.

No tested strength gives an initial sodium or aluminum/aluminium answer. Sodium peaks among tested values at gate 10: 0.334640%, versus 0.001780% natively. Aluminum/aluminium peaks at 0.001521%. High strengths produce unrelated starts such as `eletrum`, followed by a return to boron.

Only final input token ` is`, position 7, is edited. Keep recipient UVx; no donor vector and no generated-token edits.

Natural fifth-element gate: 0.000000000. Scalar replacement uses the same 17 fixed gates as the preceding month sweep.

The fifth element is boron, the eleventh is sodium, and the thirteenth is aluminum/aluminium. All three native capability checks pass. These are spelled-ordinal raw completion prefixes, not chat questions.

| Gate | Initial word | P(boron) | P(sodium) | P(aluminum + aluminium) |
|---:|---|---:|---:|---:|
| 0.000000 | boron | 84.100488% | 0.001780% | 0.000068% |
| 0.500000 | boron | 85.795743% | 0.001715% | 0.000057% |
| 1.000000 | boron | 85.992425% | 0.001828% | 0.000053% |
| 1.158554 | boron | 85.721460% | 0.002196% | 0.000060% |
| 2.000000 | boron | 90.816356% | 0.001605% | 0.000041% |
| 3.000000 | boron | 92.560668% | 0.001857% | 0.000036% |
| 5.000000 | boron | 93.654201% | 0.004237% | 0.000043% |
| 10.000000 | boron | 20.821436% | 0.334640% | 0.001521% |
| 15.000000 | what | 0.627024% | 0.084529% | 0.001319% |
| 20.000000 |  ?\n | 0.294175% | 0.022655% | 0.001258% |
| 30.000000 |  ?\n | 0.074101% | 0.002238% | 0.000509% |
| 40.000000 | eletrum | 0.029944% | 0.001488% | 0.000161% |
| 50.000000 | eletrum | 0.021020% | 0.001255% | 0.000072% |
| 60.000000 | eletrum | 0.024292% | 0.001269% | 0.000059% |
| 70.000000 | eletrum | 0.022150% | 0.001595% | 0.000053% |
| 80.000000 | eletrum | 0.012708% | 0.001141% | 0.000030% |
| 90.000000 | eletrum | 0.007625% | 0.000908% | 0.000020% |

Probabilities sum exact lowercase/title-case leading-space answer prefixes, multiplying token probabilities for multi-token spellings; no subsequent punctuation or EOS. Chemical symbols and other phrasings are excluded. They are not semantic answer probabilities.

Native BF16 Qwen3-4B; pinned model/MOLT; FP32 gate/operator and edit with one BF16 cast; eager attention; TF32 off; greedy 32-token continuations. No extra normalization or donor operator copy.

## Native capability controls

### fifth

Prompt: `the fifth element in the periodic table is`.
Final input gate: 0.000000000.

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### eleventh

Prompt: `the eleventh element in the periodic table is`.
Final input gate: 3.026236534.

```text
 sodium, which has an atomic number of 11. The atomic number is the number of protons in the nucleus of an atom. So, sodium has
```

### thirteenth

Prompt: `the thirteenth element in the periodic table is`.
Final input gate: 0.000000000.

```text
 aluminum, which has an atomic number of 13. the atomic number is the number of protons in the nucleus of an atom. so, the number
```

## Full sweep responses

### Gate 0.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 0.500000000

```text
 boron, which has an atomic number of 5. The atomic number is the number of protons in the nucleus of an atom. So, boron
```

### Gate 1.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 1.158553600

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 2.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 3.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 5.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 10.000000000

```text
 boron, which has an atomic number of 5. The sixth element is carbon, with an atomic number of 6. The seventh element is nitrogen,
```

### Gate 15.000000000

```text
 what? The fifth element in the periodic table is Boron. Its symbol is B and its atomic number is 5. Boron is a chemical element with
```

### Gate 20.000000000

```text
 ?
The fifth element in the periodic table is Boron. Its symbol is B and its atomic number is 5. Boron is a chemical element with atomic
```

### Gate 30.000000000

```text
 ?
The fifth element in the periodic table is Boron. Its symbol is B and its atomic number is 5. Boron is a chemical element with atomic
```

### Gate 40.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

### Gate 50.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

### Gate 60.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

### Gate 70.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

### Gate 80.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

### Gate 90.000000000

```text
 eletrum, but it's not on the periodic table. What is the fifth element? The fifth element in the periodic table is Boron. But wait
```

## Validation and scope

{
  "status": "pass",
  "sweep_conditions": 17,
  "array_files_checked": 27,
  "audited_edit_arrays": 21,
  "probability_max_absolute_error": 1.8832136294832935e-07,
  "native_capability_controls": 3,
  "native_observer_and_zero_exact": true,
  "repeated_gates_exact": [
    1.1585536003112793,
    20,
    90
  ],
  "recipient_operator_identical_across_gates": true,
  "bf16_edits_exact": true,
  "input_only_mask_exact": true,
  "candidate_prefix_scores_reconstructed": true,
  "no_generated_token_edits": true
}

All arrays and the remote archive are SHA-256 verified. CPU analysis reconstructs gate/operator values, exact BF16 edits, input masks and exact-prefix probabilities. Native/observer/zero controls and three repeated strengths match exactly.

This deterministic exploratory sweep covers one prompt. An initial word change is not a stable abstract ordinal mechanism; full continuations are preserved.

Reproduce: `python3 experiments/eleven-fifth-element-scalar-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-eleven-fifth-element-scalar/`. No publication or GPU cleanup was performed.
