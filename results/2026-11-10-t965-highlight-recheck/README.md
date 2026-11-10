# T965: all-original versus final-only highlights

The example text, continuation and few-shot demonstrations are identical. Both frozen descriptions are tested under both highlight scopes.

| Condition | Development → Kit | Ton / ton → katsu | .wait → key |
|---|---:|---:|---:|
| all_original / old | 0 | 0 | 0 |
| all_original / new | 0 | 0 | 0 |
| final_only / old | 0 | 0 | 0 |
| final_only / new | 0 | 1 | 1 |

Historical responses reproduced: 9/9. Old-description final-only conditions were not previously run and are new comparisons.

These are judge answers, not native gate values. All retained marked tokens have measured positive activation in the original source windows. The code window also contains non-K activations when all original highlights are retained.

The same pinned Qwen3.8-27B-NVFP4 weights and runtime settings are used. This is a fixed post-hoc recheck, not a new held-out feature benchmark.
