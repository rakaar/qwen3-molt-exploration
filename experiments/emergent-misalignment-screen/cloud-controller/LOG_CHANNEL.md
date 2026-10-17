# Existing-worker task transfer

No Portal, tunnel, new host, GPU lifecycle change, or GPU setup is used here.
`frame-log-metadata` verifies SPAR account 710735 and exact existing worker
56373998/label before the official request_logs endpoint. Its fixed request is
`{"tail":"10000"}`; EMSCREEN2 selection happens only in bounded memory. The
official CLI documents the tail field. Provider metadata remains private; S3 receives no
Authorization header. The managed proxy and verified default TLS are required.

The first long probe was truncated at 500 physical stdout bytes. EMSCREEN2 uses
header/data/end records, <=256 decoded gzip bytes per data record, and <=480
bytes per full stdout line (the GPU sender uses <=440). A complete archive has
at most 8192 data records plus header/end. Compressed payload <=2 MiB, raw tar
<=16 MiB, each S3 fetch <=8 MiB. Asynchronous-object HTTP404 or bounded S3 XML
AccessDenied/NoSuchKey is polled at most 30 times at two-second intervals. No
credential or URL is changed; other failures stop. Raw error text is suppressed.

The parser requires exact instance/run ID, a 16-hex snapshot ID, one consistent
header, every indexed chunk, an end marker, compressed/raw lengths and SHA256,
valid bounded gzip, task-only tar members, and complete embedded file hashes.
Conflicting duplicate chunks, traversal, links, arbitrary logs, credentials,
unknown response/control fields, and incomplete snapshots fail closed. Task
backups are private 0700/0600. Original inputs are compared byte-for-byte with
commit `21da619829635b97b1797600e6b3e5bd3ef5e468` after transfer. Only
`native-deadline-v2-status.json` and `control-provenance.json` were added to the
existing artifact-name allowlist. Native receipt acceptance is not a claim that
native responses supplied account-owner fields; receipt/source audit is separate.

Save fresh actual platform observations and functional proxy evidence using
`network_evidence.py` before retrieving. Never rewrite unknown to enforced.

```sh
python3 experiments/emergent-misalignment-screen/cloud-controller/log_retrieve.py \
  --kind probe --snapshot-id 412252d51a209e31 --nonce 9e508dd38e0b8a9c \
  --expect-compressed-sha256 af8ae0d0d021c1aca91f648d88df493ae896bd8e75aafeac46718c1762f0ee22 \
  --expect-raw-sha256 df27ea8128cfb4c1d2bf26a0121cc7796b1c11530d61607d8e03fecfed2d0360

python3 experiments/emergent-misalignment-screen/cloud-controller/log_retrieve.py \
  --kind archive --snapshot-id 6fc81ba246309285 \
  --destination /workspace/work/backups/emscreen2-initial-6fc81ba246309285 \
  --expect-compressed-sha256 2a42c47d2ccb8ba29aa91654c37d25b93d9a116375f011955c5d73ae76686c80 \
  --expect-raw-sha256 fb2aa92de9a43bff59530dc4e61df86f8557c45f946f0b1e6858063f840bb0fc
```

Omit `--snapshot-id` to select the latest complete verified archive; an incomplete
or oversized fetch is never accepted. No arbitrary provider logs are persisted.
The real probe nonce and both hashes were verified; the initial real archive
verified eight files and all four original input files. It contained setup-failed
status and no CUDA smoke. This transfer proof alone never marks readiness.
