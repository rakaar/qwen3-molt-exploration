import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
from hashlib import sha256

source = Path('results/2026-11-10-t2375-pre11-coverage/recovered-windows.json')
feature = Path('comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json')
data = json.loads(source.read_bytes())
assert sha256(feature.read_bytes()).hexdigest() == data['feature_sha256']
assert not data['unmatched']
cases = {}
inactive = {}
for w in data['windows'].values():
    for match in w['matches']:
        for a in match['active_positions']:
            if 'water' not in a['token'].casefold():
                continue
            assert len(w['matches']) == 1
            pos = a['window_position']
            assert (a['token'], a['gate']) == (w['str_tokens'][pos], w['activations'][pos])
            key = (match['conversation'], a['conversation_position'])
            if key in cases:
                assert cases[key] == a
            cases[key] = a
        for pos, (t, gate) in enumerate(zip(w['str_tokens'], w['activations'])):
            if 'water' in t.casefold() and gate == 0:
                inactive[match['conversation'], match['window_start'] + pos] = ''.join(w['str_tokens'][max(0,pos-10):pos+10])

# Qualitative codes assigned after reading both sides of each occurrence.
nouns = {(605,218),(840,458),(2230,691),(3390,307),(3457,342),(4260,144),
         (4260,182),(6413,913),(6730,298),(7178,385),(7291,492),(8002,198),
         (8494,747),(8548,713),(11259,901)}
joined = {(183,401),(2230,670),(6138,547)}
hyphenated = {(3322,1023),(6730,311)}
fixed = {(5600,766)}
assert nouns | joined | hyphenated | fixed <= cases.keys()
categories = defaultdict(list)
rows = []
def cell(s):
    return s.replace('|','\\|').replace('\n',' ↵ ').strip()
for (c,p), a in sorted(cases.items()):
    key = (c,p)
    cat = ('Noun modifier' if key in nouns else 'Joined word' if key in joined else
           'Hyphenated adjective' if key in hyphenated else 'Fixed expression' if key in fixed else 'Standalone noun')
    categories[cat].append(a['gate'])
    prefix = a['prefix'][:-len(a['token'])][-62:]
    cont = a['following_text'].split('<|im_end|>')[0][:62]
    rows.append(f"| {c} / {p} | {cell(prefix)} **{cell(a['token'])}** | {cell(cont)} | {cat} | {a['gate']:.3f} |")

out = Path('results/2026-11-21-t2375-water-continuations/README.md')
summary = [f'| {cat} | {len(vals)} | {median(vals):.2f} |' for cat,vals in sorted(categories.items())]
out.write_text('\n'.join(['# What follows an active water token in L29–T2375?', '',
    f'This first pass uses {len(cases)} distinct active token positions across {len(set(c for c,p in cases))} UltraChat test conversations. Selection was based on the existing detection and fuzz evaluation windows. Overlapping observations count once per conversation and token position. Each water hit has a unique match to the original conversation, including the continuation beyond the saved window. Gate values come from the activation record. Following text comes from the corpus.',
    '', '## Continuations', '', '| Observed use | Positions | Median gate |','|---|---:|---:|',*summary,'',
    'Noun modifier includes productive phrases such as “water scarcity” and “water activities.” Joined words include “waterways,” “watercolor,” and “watersheds.” The token for the last of these is “ waters.” The fixed expression is “high-water mark,” with the gate on the token “-water.” Categories were assigned by reading the passages. These descriptive counts cannot estimate firing rates by context because the evaluation sampled on activation.',
    '', '## All active water positions', '', 'The bold token carries the gate value. The continuation column starts at the next token. Whitespace is simplified for display. Indices are zero-based.',
    '', '| Conversation / position | Prefix ending at active token | Observed continuation | Use | Gate |','|---|---|---|---|---:|',*rows,
    '', '## Nearby zero-gate occurrences', '', 'These are the other tokens containing “water” with a measured gate of zero in the same set of windows. They form a convenience comparison.',
    '', '| Conversation / position | Excerpt |','|---|---|',*[f'| {c} / {p} | {cell(ctx)} |' for (c,p),ctx in sorted(inactive.items())],
    '', '## Sources', '', '- [Matched windows, token identities, gates and continuations](../2026-11-10-t2375-pre11-coverage/recovered-windows.json)',
    '- [Original evaluation cases](../../comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json)',
    '- [Corpus and tokenizer versions](../2026-11-10-t2375-pre11-coverage/source-manifest.json)','']))
print(len(cases),len(set(c for c,p in cases)),len(inactive),summary)
print(Counter(a['token'] for a in cases.values()))
