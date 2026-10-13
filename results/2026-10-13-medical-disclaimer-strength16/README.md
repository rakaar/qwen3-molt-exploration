# L19–T368 negative-factor-16 follow-up

Factor −16 did not produce a coherent refusal-to-answer switch on any of the five active prompts. Qualification denial remained, and four active outputs had malformed openings. The inactive ordinary symptom control was unchanged. This follow-up tests whether the original pilot stopped at too low a negative strength. Read [the complete report](index.html) for the actual conclusion and all generated outputs. The original −8 endpoint was a pilot cutoff, not a demonstrated limit.

The original six clean-selected prompts are reused: five active medical/qualification or missing-information cases, and an ordinary inactive stomach-pain control. Four independent conditions share each forward pass with identical prompt lengths: coefficient factors 1, 0, −8 and −16. A separate four-row clean batch checks the unchanged reference. The copied `run.py` has the same SHA-256 as the original pilot source.

```
original MLP output[token] += (factor - 1) * gate(current x) * lowrank(current x)
```

Factor −16 means subtracting seventeen current estimated contributions. There is no donor. Original MLPs remain in use and attention stays unfrozen. The layer-19 pre-normalization input, checkpoint standardization and output scaling follow the original intervention exactly. This changes a learned component estimate, not a native gate inside the original model.

The protocol retains the complete original prompt list and original control descriptions for provenance; this follow-up runs only the six IDs in `selection.json` at factors 1, 0, −8 and −16, plus clean identity repeats. The broader controls listed in the inherited protocol belong to the original pilot. `extension.py` preserves 24 intervention generations and 24 clean identity generations. The raw records include complete chat templates, generated tokens, all consumed-token gate values, full-chat positions, norms and edit positions. `assessment.json` records a manual reading of visible outputs; clinical correctness is not evaluated.

## Reproduce

Use an authorized GPU with about 24 GB VRAM, a compatible PyTorch build and the pinned requirements from the original pilot. From a copy of this directory, run:

```
python prepare.py
python extension.py
python export.py
python verify.py
python report.py
```

The selected public MOLT slices are preserved. The runtime uses PyTorch 2.14.0+cu126 and Transformers 4.57.6, as in the pilot; the lightweight Ubuntu image changes Python from 3.12.14 to 3.12.3. Runtime package versions are recorded in `artifacts/runtime-packages.txt`. The optional parallel range downloader only changes transfer orchestration and verifies the same immutable model hashes. Preparation downloads the base model from its immutable pinned revision and verifies the bytes. `verify.py` compares unchanged intervention source against the sibling original pilot directory. Keep both directories when reproducing the verification. The report uses the preserved manual `assessment.json`. Running here replaces saved artifacts, so preserve the originals first.

The checksum-verified tar archive is retained locally rather than duplicated in Git. Selected public tensors and all scientific source/results are backed up. The reused loader has an unused full-layer-22 download helper, which this experiment does not call.

## GPU lifecycle

The user explicitly instructed that this GPU and storage remain after the task for follow-up tests. Do not stop or delete it without a new explicit instruction. `resource-state.json` records the latest verified state and rate, and `rental-record.json` identifies the task instance. No credentials or SSH connection details are published.
