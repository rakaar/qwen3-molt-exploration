# Backup before deleting old experiment instance 58952920

These archives preserve the research code and results collected from the old, stopped SPAR experiment instance. They cover residual-stream tracing, downstream gate changes, T2375/T1468 coactivation, T1468 ablation and rescue, and the later token-confound experiments.

`manifest.json` records SHA-256 hashes for every archive. The seven run archives were checked against their original remote-transfer hashes, and their internal transfer manifests were verified again before publication. `local-code-and-reports.tar.gz` adds the corresponding local analysis scripts and report artifacts; individual file hashes are in `code-and-report-files.json`.

To restore, download an archive, check its SHA-256 hash against the manifest, and extract it into a new directory. The run archives include prompts, protocols, source, measurements, and available dependency/model provenance. Large downloadable base-model caches are not duplicated here.

The running scoring instance 59903353 and dashboard volume 58518928 are outside this deletion scope.
