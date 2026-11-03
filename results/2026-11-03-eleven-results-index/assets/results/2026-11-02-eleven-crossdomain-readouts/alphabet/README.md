# English alphabet: three-part L29 MLP readout

Exact raw prompt: `the eleventh letter of the english alphabet is`; final input position 8, token ` is`.

The three primary vectors are native L29 MLP output M, positive intended L29-T2375 contribution c, and the actual edited output BF16(M - c). The contribution is displayed with its original positive sign; the gate-off intervention subtracts it.

All three use `W_out @ (gamma * vector / sqrt(mean(native_final_residual^2) + epsilon))`. The denominator comes from the whole native final residual for this prompt, after layer 35 at the same position. Learned final RMSNorm gamma is included. The three vectors share that scale and are not independently normalized.

These are direct vocabulary scores, not probabilities or complete-model answers. Incoming residual and later-layer computations are excluded from these three MLP-part projections.

K and L target rows are complete single-token variants.

![Three-part vocabulary readout](three-part-readout.png)

| Token | Full MLP | Removed MOLT | After removal |
|---|---:|---:|---:|
| <code>·k</code> | +1.017619 | +0.220251 | +0.797707 |
| <code>·K</code> | +0.970533 | +0.070220 | +0.900769 |
| <code>·l</code> | +0.545366 | +0.067791 | +0.477930 |
| <code>·L</code> | +0.481297 | -0.152826 | +0.633878 |

The highest full-MLP readout token is <code>·ele</code>; the highest remainder token is <code>·ele</code>.

For the displayed contrast <code>·k</code> versus <code>·l</code>, the direct margin changes from +0.472253 to +0.319777. The saved complete-model first-token margin changes from +4.500000 to -2.625000. This first-token comparison does not give complete-answer probabilities.

The native MLP vector reproduced exactly on the new forward pass, and the native final RMSNorm output reconstructed exactly. CPU auditing passed over 130 saved output-matrix rows, including the highest and lowest tokens for every part and all target tokens.

The ideal FP32 decomposition is additive to within 0.00000054 over the full vocabulary. The primary display uses the actual BF16 remainder. Its maximum score difference from ideal M-minus-c is 0.00187640.

The direct readout is a diagnostic of these saved vectors. It does not by itself explain the full generated-answer change or establish a general add-one mechanism.

[All vocabulary scores and ranks](all-vocabulary-scores.csv) · [Target scores](target-token-scores.csv) · [CPU audit](audit.json)

The CSV preserves raw tokenizer-piece strings and explicitly labels unmapped output rows. Ranks use competition ranking, so exact ties share a rank.

## Recorded native continuation

> k. the eleventh letter of the greek alphabet is kappa. the eleventh letter of the russian alphabet is ё. the eleventh letter of the

## Recorded final-input-only gate-off continuation

> l. the eleventh letter of the english alphabet is l. the eleventh letter of the english alphabet is l. the eleventh letter of the english alphabet

These historical continuations hit the 32-token cap and can drift. This readout analysis adds a fresh native prefill to reproduce the vectors and obtain the contextual normalization scale; it does not rerun generation or gate-off inference.


## 1  Full MLP output

| Rank | Token | Score |
|---:|---|---:|
| 1 | <code>·ele</code> | +2.253535 |
| 2 | <code>·nov</code> | +2.072665 |
| 3 | <code>·November</code> | +1.973366 |
| 4 | <code>·Nov</code> | +1.972656 |
| 5 | <code>ele</code> | +1.937195 |
| 6 | <code>Ele</code> | +1.875349 |
| 7 | <code>November</code> | +1.860719 |
| 8 | <code>Nov</code> | +1.851499 |
| 9 | <code>nov</code> | +1.830976 |
| 10 | <code>_ele</code> | +1.750205 |
| 11 | <code>·Ele</code> | +1.697858 |
| 12 | <code>·novel</code> | +1.697411 |
| 13 | <code>(ele</code> | +1.690963 |
| 14 | <code>十一</code> | +1.651051 |
| 15 | <code>·XI</code> | +1.599518 |
| 16 | <code>·november</code> | +1.578547 |
| 17 | <code>·Novel</code> | +1.577663 |
| 18 | <code>KI</code> | +1.499934 |
| 19 | <code>XI</code> | +1.485714 |
| 20 | <code>·eleven</code> | +1.480461 |
| 21 | <code>ki</code> | +1.457304 |
| 22 | <code>oji</code> | +1.431338 |
| 23 | <code>owie</code> | +1.415172 |
| 24 | <code>-Nov</code> | +1.392281 |
| 25 | <code>(wrapper</code> | +1.370585 |
| 26 | <code>LI</code> | +1.351910 |
| 27 | <code>·Nikki</code> | +1.350281 |
| 28 | <code>k</code> | +1.347778 |
| 29 | <code>·wrapper</code> | +1.341177 |
| 30 | <code>·thank</code> | +1.313088 |

## 2  Removed MOLT contribution

| Rank | Token | Score |
|---:|---|---:|
| 1 | <code>·nov</code> | +1.279739 |
| 2 | <code>·ele</code> | +1.147511 |
| 3 | <code>nov</code> | +1.130893 |
| 4 | <code>·Nov</code> | +1.099133 |
| 5 | <code>Nov</code> | +1.041943 |
| 6 | <code>·novel</code> | +1.011423 |
| 7 | <code>ele</code> | +0.979902 |
| 8 | <code>_ele</code> | +0.966579 |
| 9 | <code>-Nov</code> | +0.914134 |
| 10 | <code>Ele</code> | +0.913743 |
| 11 | <code>·Novel</code> | +0.911373 |
| 12 | <code>·November</code> | +0.903179 |
| 13 | <code>·Ele</code> | +0.889493 |
| 14 | <code>November</code> | +0.866066 |
| 15 | <code>(ele</code> | +0.838080 |
| 16 | <code>·november</code> | +0.836605 |
| 17 | <code>十一</code> | +0.804470 |
| 18 | <code>illi</code> | +0.798721 |
| 19 | <code>lli</code> | +0.789718 |
| 20 | <code>·novels</code> | +0.765261 |
| 21 | <code>XI</code> | +0.748904 |
| 22 | <code>·novembre</code> | +0.743550 |
| 23 | <code>·XI</code> | +0.731839 |
| 24 | <code>·ноября</code> | +0.726270 |
| 25 | <code>新型</code> | +0.726012 |
| 26 | <code>eli</code> | +0.717891 |
| 27 | <code>·EI</code> | +0.701129 |
| 28 | <code>·نوفمبر</code> | +0.697136 |
| 29 | <code>·elev</code> | +0.677576 |
| 30 | <code>小说</code> | +0.671534 |

## 3  MLP after removal

| Rank | Token | Score |
|---:|---|---:|
| 1 | <code>·ele</code> | +1.106124 |
| 2 | <code>·November</code> | +1.069978 |
| 3 | <code>·Nin</code> | +1.048049 |
| 4 | <code>k</code> | +1.046592 |
| 5 | <code>·ups</code> | +1.028022 |
| 6 | <code>KI</code> | +1.007341 |
| 7 | <code>ОН</code> | +1.005237 |
| 8 | <code>LI</code> | +0.999324 |
| 9 | <code>November</code> | +0.994736 |
| 10 | <code>·ki</code> | +0.981480 |
| 11 | <code>·Often</code> | +0.977033 |
| 12 | <code>Ele</code> | +0.961296 |
| 13 | <code>ele</code> | +0.956947 |
| 14 | <code>:n</code> | +0.949409 |
| 15 | <code>;n</code> | +0.935640 |
| 16 | <code>ki</code> | +0.932499 |
| 17 | <code>·often</code> | +0.931791 |
| 18 | <code>often</code> | +0.924342 |
| 19 | <code>Occurred</code> | +0.909824 |
| 20 | <code>lements</code> | +0.907555 |
| 21 | <code>li</code> | +0.905533 |
| 22 | <code>Fil</code> | +0.900910 |
| 23 | <code>·K</code> | +0.900769 |
| 24 | <code>·eleven</code> | +0.898659 |
| 25 | <code>·Nov</code> | +0.874361 |
| 26 | <code>{n</code> | +0.872532 |
| 27 | <code>(wrapper</code> | +0.868978 |
| 28 | <code>·XI</code> | +0.867537 |
| 29 | <code>,n</code> | +0.863497 |
| 30 | <code>(n</code> | +0.860113 |

## Lowest-scoring tokens for the removed contribution

| Token | Score |
|---|---:|
| <code>·thirteen</code> | -0.823728 |
| <code>esan</code> | -0.729970 |
| <code>特朗</code> | -0.705862 |
| <code>xz</code> | -0.688473 |
| <code>cce</code> | -0.688347 |
| <code>·bz</code> | -0.686473 |
| <code>筏</code> | -0.675643 |
| <code>ISO</code> | -0.661241 |
| <code>gow</code> | -0.656783 |
| <code>_SENS</code> | -0.649030 |
| <code>パー�</code> | -0.644851 |
| <code>加盖</code> | -0.643713 |
| <code>(keys</code> | -0.642054 |
| <code>谟</code> | -0.639913 |
| <code>·LZ</code> | -0.631066 |
| <code>·qw</code> | -0.629793 |
| <code>CCA</code> | -0.625764 |
| <code>·Saturdays</code> | -0.618770 |
| <code>�</code> | -0.614396 |
| <code>ду</code> | -0.612849 |
| <code>黯</code> | -0.612472 |
| <code>anos</code> | -0.612446 |
| <code>除夕</code> | -0.610524 |
| <code>·ISO</code> | -0.603117 |
| <code>erable</code> | -0.594261 |
| <code>·warmth</code> | -0.592658 |
| <code>正版</code> | -0.592404 |
| <code>·gag</code> | -0.590473 |
| <code>hev</code> | -0.589973 |
| <code>January</code> | -0.589845 |
