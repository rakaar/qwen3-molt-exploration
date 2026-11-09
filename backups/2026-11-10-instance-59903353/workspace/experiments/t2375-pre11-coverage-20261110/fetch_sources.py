"""Recover public source text for a post-scoring coverage audit; no inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'comparisons/2026-11-10-t2375-pre11-coverage'
DATA.mkdir(exist_ok=True)
revision = '8049631c405ae6576f93f445c6b8166f76f5505a'
name = 'data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet'
expected = 'c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70'
url = f'https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/{revision}/{name}'
path = DATA / Path(name).name
partial = path.with_suffix('.partial')
if partial.exists() and hashlib.sha256(partial.read_bytes()).hexdigest() == expected:
    partial.rename(path)
if not path.exists():
    response = requests.get(url, stream=True, timeout=(30, 90))
    response.raise_for_status()
    with partial.open('wb') as handle:
        for chunk in response.iter_content(2**20):
            handle.write(chunk)
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == expected
    partial.rename(path)
assert path.stat().st_size == 81176553
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
model_revision = '1cfa9a7208912126459214e8b04321603b3df60c'
config_url = f'https://huggingface.co/Qwen/Qwen3-4B/resolve/{model_revision}/tokenizer_config.json'
config = DATA / 'tokenizer_config.json'
if not config.exists():
    response = requests.get(config_url, timeout=30)
    response.raise_for_status()
    json.loads(response.content)
    config.write_bytes(response.content)
tokenizer = ROOT / 'comparisons/2026-11-02-eleven-contribution-readout/tokenizer.json'
manifest = dict(
    collected_at_utc=datetime.now(timezone.utc).isoformat(),
    dataset='HuggingFaceH4/ultrachat_200k', revision=revision, split='test_sft',
    dataset_url=url, dataset_file=path.name, dataset_bytes=path.stat().st_size,
    dataset_sha256=expected, tokenizer_config_url=config_url,
    tokenizer_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    tokenizer_file=str(tokenizer.relative_to(ROOT)),
    tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    note='Current pinned public corpus; exact token matching to saved evaluation windows is required before classifying any continuation.',
)
(DATA / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
