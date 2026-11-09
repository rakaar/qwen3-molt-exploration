"""Match every scored passage to source tokens and recover its continuation.

No model inference, judge calls, description edits, or changes to scored cases.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'comparisons/2026-11-10-t2375-pre11-coverage'
OUT = ROOT / 'results/2026-11-10-t2375-pre11-coverage'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'artifacts/molt-dashboard-vast-20261028/source'))
from crosslayer_transcoder.dashboard.token_display import escape_added_tokens

feature_path = ROOT / 'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json'
feature = json.loads(feature_path.read_text())
settings = json.loads((feature_path.parent.parent / 'manifest.json').read_text())['settings']
test_conversations = set(settings['test_conversations'])
tokenizer = Tokenizer.from_file(str(ROOT / 'comparisons/2026-11-02-eleven-contribution-readout/tokenizer.json'))
config = json.loads((DATA / 'tokenizer_config.json').read_text())
template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(config['chat_template'])
vocab = {i: escape_added_tokens(tokenizer.decode([i], skip_special_tokens=False))
         for i in range(tokenizer.get_vocab_size())}

def fingerprint(row):
    return hashlib.sha256(json.dumps([row['str_tokens'], row['activations']], ensure_ascii=False).encode()).hexdigest()

windows = {}
tails = defaultdict(list)
for method in ('detection', 'fuzz', 'active_fuzz'):
    for row_index, row in enumerate(feature['methods'][method]['rows']):
        key = fingerprint(row)
        if key not in windows:
            windows[key] = dict(str_tokens=row['str_tokens'],
                                canonical_tokens=[escape_added_tokens(t) for t in row['str_tokens']],
                                activations=row['activations'],
                                occurrences=[], matches=[])
            tails[tuple(windows[key]['canonical_tokens'][-4:])].append(key)
        windows[key]['occurrences'].append(dict(method=method, row_index=row_index,
                                                activating=row['activating'],
                                                highlighted_positions=row.get('highlighted_positions')))

rows = pq.read_table(DATA / 'test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet').slice(0, 11345).to_pylist()
total_tokens = 0
for conversation, source in enumerate(rows):
    rendered = template.render(messages=source['messages'], tools=None, add_generation_prompt=False,
                               bos_token=config.get('bos_token'), eos_token=config.get('eos_token'))
    # Encode the full source conversation so a clipped cache boundary does not
    # prevent observing the recorded source continuation.
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    budget = min(1024, 10_000_000 - total_tokens)
    cached_ids = ids[:budget]
    total_tokens += len(cached_ids)
    if conversation not in test_conversations:
        continue
    displayed = [vocab[i] for i in cached_ids]
    for stop in range(4, len(displayed) + 1):
        for key in tails.get(tuple(displayed[stop-4:stop]), ()):
            entry = windows[key]
            start = stop - len(entry['str_tokens'])
            if start < 0 or displayed[start:stop] != entry['canonical_tokens']:
                continue
            active = []
            for position, gate in enumerate(entry['activations']):
                if gate <= 0:
                    continue
                absolute = start + position
                future = ids[absolute+1:absolute+65]
                active.append(dict(window_position=position, conversation_position=absolute,
                                   token=entry['str_tokens'][position], gate=gate,
                                   prefix=tokenizer.decode(ids[max(0,absolute-24):absolute+1], skip_special_tokens=False),
                                   following_tokens=[vocab[i] for i in future],
                                   following_token_ids=future,
                                   following_text=tokenizer.decode(future, skip_special_tokens=False)))
            entry['matches'].append(dict(conversation=conversation, prompt_id=source['prompt_id'],
                                         window_start=start, window_stop=stop,
                                         window_token_ids=ids[start:stop], active_positions=active))
    if conversation and conversation % 1500 < 4:
        print(json.dumps(dict(conversation=conversation, matched=sum(bool(w['matches']) for w in windows.values()),
                              total=len(windows))), flush=True)

assert total_tokens == 10_000_000, total_tokens
unmatched = [k for k, w in windows.items() if not w['matches']]
ambiguous = [k for k, w in windows.items() if len(w['matches']) != 1]
result = dict(recovered_at_utc=datetime.now(timezone.utc).isoformat(),
              feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              corpus_manifest=json.loads((DATA / 'source-manifest.json').read_text()),
              original_collection_conversations=11345, original_collection_tokens=total_tokens,
              exact_displayed_token_match_after_added_token_escape=True,
              added_token_note='Some original rows leave think tags literal; compare after the saved source escape_added_tokens function only. Original scored strings are retained unchanged.',
              heldout_conversations_only=True,
              total_unique_windows=len(windows), unmatched=unmatched,
              nonunique_match_windows=ambiguous, windows=windows)
(OUT / 'recovered-windows.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(unique_windows=len(windows), unmatched=len(unmatched), nonunique=len(ambiguous))))
assert not unmatched, 'Cannot classify unmatched windows'
for key, window in windows.items():
    if not any(window['activations']):
        continue
    if any(a > 0 and 'water' in t.lower() for t,a in zip(window['str_tokens'],window['activations'])):
        continue
    ids = [(r['method'],r['row_index']) for r in window['occurrences'] if r['activating']]
    print('ROWS', ids)
    for a in window['matches'][0]['active_positions']:
        print(repr(a['prefix'][-85:]), 'NEXT', repr(a['following_text'][:125]))
