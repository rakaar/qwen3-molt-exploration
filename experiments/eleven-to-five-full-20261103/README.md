# Gate and UVx transplantation: eleventh month into fifth month

Recipient: `the fifth month of the year is`. Donor: `the eleventh month of the year is`. Feature: L29–T2375, native BF16 Qwen3-4B with the same pinned model and MOLT files as the preceding scalar-only sweep.

Freeze the donor operator from its unmodified final input ` is` position 7. At recipient final input position 6, replace only the selected MOLT contribution:

    new local MLP output = native recipient MLP output
                           - native recipient gate * recipient UVx
                           + fixed gate * frozen donor UVx

Calculate the edit in FP32, then cast that one local output to BF16 once. Every other local MLP output is checked unchanged; generated positions receive no edit. The natural recipient gate is zero. At gate 1.1585536003112793 the inserted vector exactly equals the saved natural eleventh-month contribution, in native MLP-output units. No extra vector normalization or final RMSNorm is applied to the patch.

The same 17 fixed gates as the previous scalar-only sweep are tested: 0, 0.5, 1, 1.1585536003112793, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90. Only the replacement operator changes between the two experiments. Identical gates are not identical edit norms: donor UVx is 2.505470 times the recipient operator norm.

Controls include native/observer/zero runs, a scalar-only donor-value run reproducing the previous experiment, and recipient-operator controls matched to the intended donor-patch norm at the natural donor value and first initial-November strength. The latter control strength is chosen by a prespecified rule after seeing the sweep, rather than being an independently preselected gate. The natural full patch, first initial-November strength, and maximum gate are repeated exactly. Full generated token sequences and full-vocabulary logit hashes match in those repetitions.

The unamplified full donor contribution retains May. Donor-operator gates 10 and 15 start with November and explicitly correct back to May. Equal-norm recipient-operator steering also gives that same continuation at the first-flip comparison. This does not demonstrate an advantage specific to the donor direction in these controls. Higher donor-operator gates produce numbers or repeated eleventh-month text; all 32-token continuations are retained.

Reproduce local audit and figure after collecting the scientific raw archive:

```sh
python3 experiments/eleven-to-five-full-20261103/analyze.py
python3 experiments/eleven-to-five-full-20261103/plot.py
```

Results: `results/2026-11-03-eleven-to-five-full/`. Raw source, protocol, vectors, logits and hash manifest: `comparisons/2026-11-03-eleven-to-five-full/`. The archive and every remote file were SHA-256 verified locally. The guarded task helper is `gpu-rentals/2026-11-02-eleven-activation/full_remote.py`; it verifies SPAR instance identity and spending headroom and contains no lifecycle cleanup action. No GitHub publication or GPU stop/delete was performed.
