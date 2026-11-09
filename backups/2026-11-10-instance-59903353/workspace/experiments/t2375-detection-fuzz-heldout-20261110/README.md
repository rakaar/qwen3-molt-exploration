# Frozen-description held-out Detection and Fuzz

Run both original frozen descriptions on all 100 saved cases per method, using
the exact cached local judge and settings from the earlier Active Fuzz run.
The evaluation includes 400 fresh classifications and no selective retries.

`protocol.json` records the original description hashes, judge revision and
input-plan hash. `run.py` checks these identities, re-hashes the cached judge,
verifies runtime settings against Active Fuzz and sends test text directly to
the tokenizer and judge. Logs expose progress counts only. No test examples
were manually inspected during preparation or scoring of this run.

`analyze.py` verifies every saved request against the frozen plan, recomputes
the paired outcome counts, and checks repeated requests against Active Fuzz.
The run does not alter the prior source, result files, or descriptions.

Results: `results/2026-11-10-t2375-detection-fuzz-heldout/README.md`.
The original GPU's authorized 3:00 AM IST compute stop is unchanged.
