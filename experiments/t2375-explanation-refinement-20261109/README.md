# T2375 explanation refinement

`candidate-explanation.txt` was frozen before examining held-out case text. `protocol.json` records its initial state and stripped-text hash. The completed manual dashboard pilot and its limitations are in `results/2026-11-09-t2375-explanation-refinement/README.md`.

`extract_requests.py` retrieves only this feature's requests from the pinned public dataset using verified HTTP byte ranges. It does not infer labels. The authenticated original `features/2375.json` supplies per-case labels and historical predictions; the public export supplies exact prompts and settings, not responses.

Prepare the full matched comparison without inference:

```sh
python3 experiments/t2375-explanation-refinement-20261109/heldout_replay.py
```

The script verifies the frozen text, replays the old metric counts, matches all 300 binary rows to exact exported requests, and escapes every added token in both comparison arms using the saved upstream `token_display.py`. The local upstream source is `artifacts/molt-dashboard-vast-20261028/source/`. The input files are recorded by SHA-256 in the generated plan.

After obtaining a reachable endpoint for the original judge, execution is explicit:

```sh
python3 experiments/t2375-explanation-refinement-20261109/heldout_replay.py \
  --execute --base-url CURRENT_JUDGE_URL
```

An API key, if required, is read from `vLLM_API_KEY`; never place a key in a command argument, repository or result file. The script requires the original advertised model alias, deduplicates identical requests, resumes recorded work, retains invalid results, and reports both confusion counts and paired corrections/regressions. It does not rent GPUs or change any dashboard explanation. Execution has not yet been performed.
