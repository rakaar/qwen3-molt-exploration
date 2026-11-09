# The evaluation cut off the clue

Research page: https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-10-t2375-evaluation-cutoff/

Open index.html locally, or rebuild using `python3 render.py` in this folder. The page is self-contained and uses only study-data.json and template.html for rebuilding.

The page includes three actual saved dashboard examples, all 12 original and extended evaluation passages in two columns of six, exact old/new descriptions, per-case judge responses and aggregate scores, plus the Active Fuzz inactive-highlight control. No XI or K cases are claimed in this subset.

The full-source build entry point is experiments/t2375-evaluation-cutoff-site-20261110/build.py in the local research project. No model inference is performed by either build.

Browser interactions, source hashes, responsive layout, prompt fidelity and scores were checked. See site-validation.json and site-manifest.json.
