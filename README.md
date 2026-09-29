# Qwen3-4B MOLT exploration

Results, figures, and code from a small exploratory session on 2026-09-29.

**[Open the HTML results summary](https://rakaar.github.io/qwen3-molt-exploration/)** · [Downloadable summary](index.html)

The HTML is self-contained: its plots and gate-explorer data are embedded, so a downloaded copy works offline.

## Findings

- **Reconstruction:** released MOLT layers 17 and 25 reduced standardized squared error against the saved-mean baseline by 73.43% and 80.59%, respectively, on 64 held-out UltraChat `test_sft` conversations (58,424 tokens).
- **Full layer-17 replacement:** the model retained the same numerical/geographic answers in three qualitative questions, and answered the Dallas → state → capital question with Austin. This is not a broad capability evaluation.
- **Texas-labelled transcoder features:** eight of 73 retrieved candidates activated at the Dallas token. The search was not exhaustive; activation alone is not a causal graph.
- **Published small → hot intervention:** the unchanged “opposite of small” prompt switches its top next token from `large` to `cold`. At strength 4.5, P(cold)=99.774%, compared with the community reference's 99.822%.
- **MOLT gate response:** at the `small` token, the combined intervention changes layer 21 from 4 to 5 active gates and layer 22 from 3 to 5. Gate identities change as well as counts. Layers are zero-based.

**Not established:** Qwen MOLT Jacobian fidelity, a causal MOLT equivalent of the transcoder intervention, an unrestricted intervention result, or a validated multihop attribution graph.

The feature intervention freezes attention patterns and pins MLP outputs through layer 22 to their baseline plus decoder edits. MOLTs at layers 21/22 were passive observers of captured inputs; they did not drive the answer. The model's MOLT inputs are before post-attention normalization, as required by Georg's checkpoints.

## Contents

| Experiment | Report | Raw data |
|---|---|---|
| Held-out reconstruction | [Report](results/reconstruction-layers17-25/report.md) | [JSON](results/reconstruction-layers17-25/reconstruction.json) |
| Three-question layer replacement | [Report](results/layer17-three-questions/report.md) | [JSON](results/layer17-three-questions/comparison.json) |
| Dallas response | [Report](results/dallas-one-word/report.md) | [JSON](results/dallas-one-word/comparison.json) |
| Texas feature activations | [Report](results/texas-features/report.md) | [JSON](results/texas-features/activations.json) |
| Dallas MOLT gates | [Report](results/dallas-molt-gates/report.md) | [JSON](results/dallas-molt-gates/gates.json) |
| Small-to-hot reproduction | [Report](results/small-hot-reproduction/report.md) | [JSON](results/small-hot-reproduction/reproduction.json) |
| Layers 21/22 MOLT comparison | [Report](results/small-hot-molt-gates-21-22/report.md) | [JSON](results/small-hot-molt-gates-21-22/comparison.json) |

## Resume on a future GPU

No credentials, SSH details, or GPU lifecycle commands are included. Choose and authorize a GPU separately. The original run used an RTX 5090, Python 3.12.14, PyTorch 2.11.0+cu128, and Transformers 4.57.6. See the recorded [environment](results/reconstruction-layers17-25/environment.txt); it is a provenance snapshot, not a portable lockfile.

```bash
git clone https://github.com/rakaar/qwen3-molt-exploration.git
cd qwen3-molt-exploration
python -m venv .venv
source .venv/bin/activate
# Install a PyTorch build appropriate for your GPU. Original CUDA wheel:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/Goreg12345/crosslayer-transcoder.git
git -C crosslayer-transcoder checkout 2611214e6cf46cd96d4d689bcee369dd4fb27b30
python download_models.py
```

For the small-to-hot reproduction and gate comparison:

```bash
python prepare_small_hot.py
python reproduce_small_hot.py
python fetch_molt_gates_21_22.py
python compare_molt_gates_21_22.py
python report_molt_gates_21_22.py
```

This downloads the base model, about 2.9 MB of selected transcoder data, and about 52 MB of MOLT gate data. It does not require full MOLT transform matrices. The summary and existing measurements can be inspected without a GPU.

For full MOLT reconstruction and Dallas experiments:

```bash
python download_layers_17_25.py
bash run_reconstruction.sh
python compare_layer17_responses.py
python compare_dallas.py
python fetch_texas_encoders.py
python measure_texas_features.py
python measure_dallas_molts.py
python plot_texas_features.py
python plot_dallas_molts.py
```

The two full MOLT checkpoints add about 8.44 GB. Reconstruction activations can be regenerated from the pinned dataset and sample seed; they are not in Git. A compact raw-results/code archive was also backed up locally before GPU teardown. The user chose to regenerate the large reconstruction activation cache later; that cache and redownloadable model weights are intentionally excluded. Small prompt activations and gate tensors are preserved.

For a CPU-only rebuild of the HTML:

```bash
python build_summary.py
python -m http.server 8000
```

Open `http://localhost:8000`. Generating the HTML uses only the Python standard library and existing images.

## Provenance and attribution

- [MOLT article](https://transformer-circuits.pub/2025/bulk-update/index.html)
- Base model: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`.
- [Georg Lange's MOLTs](https://huggingface.co/georglange/qwen3-4b-molt): `eed10292a96521837f5d65b36fdfa3bf5bea0037`.
- [Georg's training code](https://github.com/Goreg12345/crosslayer-transcoder/tree/2611214e6cf46cd96d4d689bcee369dd4fb27b30): unchanged model/normalization implementation.
- [Hanna's transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders): `94d176260ac39ce2f882b8b09aba8c118df29bb3`.
- [Community intervention notebook](https://github.com/wdk0082/llm-circuits/blob/af8a37ad836dc401a0ec5dbb71307d8d7ab68028/notebooks/multilingual.ipynb): feature IDs, strengths, and steering implementation. This is not an Anthropic-released Qwen experiment.

Upstream MIT license notices are preserved in [licenses/](licenses/). Model weights retain their original repository terms. `small_hot_upstream.py` contains attributed upstream helpers; function hashes are recorded in the reproduction results.

Published experiment scripts retain the original computation. Their workspace roots were changed from the temporary GPU path to the script directory. The reconstruction runner was adjusted to regenerate an absent activation cache even when a saved result manifest exists. Plot/report builders only read saved results. A fresh GPU rerun of this packaged version has not been performed.

Historical per-experiment reports describe the state at execution time. Consult `archive-status.json` for the final backup and GPU lifecycle status. Some JSON includes pre-run recipe status text nested within provenance; measured run fields and controls are the authoritative outcome.
