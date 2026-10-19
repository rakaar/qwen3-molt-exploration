# CH absorption overview

This CPU-only report connects seven frozen spelling experiments. It separates
an absorption-like activation pattern from causal evidence, documents the failed
child diagnostic probe, and preserves the later confidence-contribution result.

Build from this repository root:

    python3 experiments/ch-absorption-overview/build.py
    python3 -m http.server 8765

Open results/2026-10-19-ch-absorption-overview/index.html. The builder reads saved
JSON and copied figures; it does not load models or run inference.

The displayed seven-feature heatmap is the exact user-selected PNG. Its SHA-256
and every measured input hash are in overview-provenance.json. The original
scientific plotting code is results/2026-10-19-ch-crosslayer-activation/build.py.

backup.py was used to copy the scoped source records into this publishing
checkout from the original local experiment tree. It records copied hashes and
intentional local-only files; it is not needed to rebuild the overview.
