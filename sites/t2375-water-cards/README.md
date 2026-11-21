# When does T2375 fire on “water”?

[Open the GitHub Pages report](https://rakaar.github.io/qwen3-molt-exploration/sites/t2375-water-cards/). The page includes the activation-selected pilot, two blind-labeled random samples, the paired noun-modifier ablation and the active-control KL comparison.

From the repository root, run `python3 sites/t2375-water-cards/build.py` to rebuild the static page and its figure and data downloads from the committed results. The site build uses the Python standard library. Open `sites/t2375-water-cards/index.html` in a browser.

| Study | Measurements and analysis |
|---|---|
| 32 activating pilot positions | [Matched examples and gate values](../../results/2026-11-21-t2375-water-continuations/) |
| 240 whole-word occurrences | [Frozen labels, measurements and sensitivity checks](../../results/2026-11-21-water-blind/) |
| 1,000 occurrences including joined forms | [Sampling frame, measured gates and group comparisons](../../results/2026-11-21-water-thousand/) |
| 179 paired ablations | [Case-level probabilities, figure and verification](../../results/2026-11-21-water-modifier-ablation/) |
| 310 firing occurrences | [Immediate next-token KL, matched active controls and tail analysis](../../results/2026-11-21-water-active-kl-control/) |
| 179 noun-modifier occurrences | [Earlier follow-up measuring KL by token position](../../results/2026-11-21-water-modifier-kl/) |

The results directories contain the inputs for regenerating the summaries and figures. For sampling again from the original UltraChat corpus, download the parquet file specified in the [source manifest](../../comparisons/2026-11-10-t2375-pre11-coverage/source-manifest.json) beside that manifest. Place the Qwen3-4B tokenizer from the revision in its tokenizer-config URL at the manifest’s repository-relative `tokenizer_file` path. The sampling scripts verify their SHA-256 hashes. Native inference used the pinned Qwen3-4B model and the [L29–T2375 weights](../../experiments/water-ice-probe-20261112/weights/L29-T2375.npz).
