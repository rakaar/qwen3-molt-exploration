# Fixed-source layer-22 MOLT strength sweep

Question: Can adding only the donor hot-associated transform produce P(cold) >= 99%, if its strength is increased beyond 4.5? Does subtracting only 0.5 times the receiver small-associated contribution help?

Receiver: `What is the opposite of "small"? Reply with only the word, nothing else.`

Donor: `What is the opposite of "hot"? Reply with only the word, nothing else.`

Only layer 22, zero-based prompt token 9, is directly changed. The original model and MLP remain in place. All attention is unfrozen. The full donor contribution is computed once, then held fixed:

```
MLP_receiver[22,9,:] += -beta * C_2339(clean receiver)
                       +alpha * C_941(clean donor)
```

`C_f(x)` means the entire gated low-rank output, with the checkpoint's input and output standardizers. It is not just a gate, and x is not multiplied by alpha. T2339 has rank 32 and the dashboard label Small-Size Adjectives and Idiomatic Phrases. T941 has rank 64 and label Thermal State and Heat Action Gate. Labels describe gate contexts and are not proof of output semantics.

The protocol records checkpoint revisions, 25 prespecified hot strengths through 512, beta values 0, 0.5, 4.5, a separate 20-strength small-only control, and eight local threshold refinements per observed 99% crossing. The small coefficient denotes subtracted clean contribution, not a target gate value. The model runs BF16 with FP32 source-vector math. Fixed eight-row batches always contain an unmodified receiver; all comparisons use the same runtime. Prompt token9 is the only direct intervention, including full-prefix greedy decoding.

## Reproduce

Install PyTorch 2.14.0+cu126 from the official CUDA 12.6 wheel index, then requirements.txt. A 24 GB GPU is sufficient. Download the selected asset from the immutable public URL below into weights/molt-22-selected.safetensors, and verify SHA256 against protocol.json:

https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/4b99e8a75a08efa9f683c7c802499c4dc747d519/experiments/residual-similarity/assets/molt-22-selected.safetensors

```
python download_base.py
python run.py
python verify.py
python report.py
```

`prefetch_base.py` is an optional standard-library parallel downloader; it verifies Hub LFS hashes. `range_prefetch.py` can resume partial downloads using verified HTTP ranges. Cloud transport/credentials are excluded from this scientific source.

Raw full-vocabulary next-token BF16 logits are stored losslessly as uint16 bits in artifacts/next-token-logits.npz. Decode with `(bits.astype(np.uint32) << 16).view(np.float32)`. The independent CPU verifier recomputes full-vocabulary softmax, source math, intended deltas, actual BF16 writes, identities, norms, and source/artifact hashes. It records outcomes in verification.json.

A local crossing refinement does not prove a global minimum. Failing through 512 does not prove impossibility at every strength. Success on one prompt pair does not establish generalization or semantic selectivity.
