"""Count explicitly defined pre-eleven coverage after recovering source text.

This is a post-hoc coverage audit. It neither updates the frozen descriptions
nor submits new judge requests. Year-2011 suffixes are reported separately.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'comparisons/2026-11-10-t2375-pre11-coverage'
OUT = ROOT / 'results/2026-11-10-t2375-pre11-coverage'
recovery = json.loads((OUT / 'recovered-windows.json').read_text())
assert not recovery['unmatched']
source_path = ROOT / 'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json'
feature = json.loads(source_path.read_text())
rows = pq.read_table(DATA / 'test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet').to_pylist()
tokenizer = Tokenizer.from_file(str(ROOT / 'comparisons/2026-11-02-eleven-contribution-readout/tokenizer.json'))
config = json.loads((DATA / 'tokenizer_config.json').read_text())
template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(config['chat_template'])
encoded = {}

def classify(ids, absolute):
    before = tokenizer.decode(ids[max(0, absolute-30):absolute+1], skip_special_tokens=False)
    following = tokenizer.decode(ids[absolute+1:absolute+65], skip_special_tokens=False)
    if re.match(r'^\s*11(?!\d)', following):
        if re.search(r'\d$', before):
            if re.search(r'(?<!\d)20$', before):
                family = 'year_2011_suffix'
            else:
                family = 'other_number_ending_11'
        elif re.search(r'(?<!\d)8-$', before):
            family = 'date_range_8_to_11'
        else:
            family = 'standalone_11'
    elif re.match(r'^\s*eleven(?:th)?\b', following, re.I):
        family = 'eleven_word'
    elif re.match(r'^\s*november\b', following, re.I):
        family = 'november'
    elif re.match(r'^\s*xi\b', following, re.I):
        family = 'xi'
    elif re.match(r'^\s*k\b', following, re.I):
        family = 'k'
    else:
        return None
    return dict(family=family, prefix=before, following=following,
                next_token_ids=ids[absolute+1:absolute+17])

audit = {}
for key, window in recovery['windows'].items():
    matching_classifications = []
    matching_examples = []
    for match in window['matches']:
        conversation = match['conversation']
        if conversation not in encoded:
            text = template.render(messages=rows[conversation]['messages'], tools=None,
                                   add_generation_prompt=False, bos_token=config.get('bos_token'),
                                   eos_token=config.get('eos_token'))
            encoded[conversation] = tokenizer.encode(text, add_special_tokens=False).ids
        ids = encoded[conversation]
        start, stop = match['window_start'], match['window_stop']
        assert ids[start:stop] == match['window_token_ids']
        events = []
        for position in range(stop-start):
            event = classify(ids, start+position)
            if event:
                event.update(window_position=position, token=window['str_tokens'][position],
                             gate=window['activations'][position], conversation=conversation,
                             conversation_position=start+position)
                events.append(event)
        matching_classifications.append([(e['window_position'],e['family']) for e in events])
        matching_examples.append(events)
    assert all(x == matching_classifications[0] for x in matching_classifications), 'Ambiguous source match changes coverage classification'
    # A repeated survey excerpt has two source locations. Both give the same
    # classification; no arbitrary source choice affects the count.
    audit[key] = dict(occurrences=window['occurrences'], events=matching_examples[0],
                      source_matches=len(window['matches']),
                      all_source_matches_agree_on_classification=True)

methods = {}
strict_families = {'standalone_11','date_range_8_to_11','eleven_word','november','xi','k'}
for method in ('detection','fuzz','active_fuzz'):
    cases = []
    counts = Counter()
    for key, entry in audit.items():
        for occurrence in entry['occurrences']:
            if occurrence['method'] != method:
                continue
            events = [e for e in entry['events'] if e['gate'] > 0]
            shown = occurrence['highlighted_positions']
            relevant = events if method == 'detection' else [e for e in events if e['window_position'] in shown]
            if occurrence['activating'] and relevant:
                families = sorted({e['family'] for e in relevant})
                for family in families:
                    counts[family] += 1
                cases.append(dict(row_index=occurrence['row_index'], fingerprint=key,
                                  families=families, active_boundary_events=relevant))
    negative_context_rows = [o['row_index'] for e in audit.values() if any(x['gate'] > 0 for x in e['events'])
                             for o in e['occurrences'] if o['method']==method and not o['activating']]
    methods[method] = dict(total_decisions=100, positive_decisions=50,
                           confirmed_pre11_positive_cases=len(cases),
                           strict_eleven_or_equivalent_cases=sum(any(f in strict_families for f in c['families']) for c in cases),
                           families=dict(counts), cases=sorted(cases,key=lambda c:c['row_index']),
                           negative_highlight_decisions_reusing_pre11_active_contexts=negative_context_rows)

fingerprint_sets = [{c['fingerprint'] for c in m['cases']} for m in methods.values()]
assert fingerprint_sets[0] == fingerprint_sets[1] == fingerprint_sets[2]
result = dict(performed_at_utc=datetime.now(timezone.utc).isoformat(), posthoc=True,
              source_feature_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
              matched_unique_windows=recovery['total_unique_windows'], unmatched_windows=0,
              repeated_source_windows=len(recovery['nonunique_match_windows']),
              repeated_source_matches_agree_on_coverage=True,
              rule='Count an evaluated positive passage when a gate-active token is immediately followed by 11, eleven/eleventh, November, XI, or K, allowing whitespace only. Fuzz/Active Fuzz must actually highlight that gate-active boundary. Count one passage once. Separate 11 as the suffix of 2011 from standalone eleven or its equivalents.',
              methods_share_same_positive_pre11_contexts=True, methods=methods)
(OUT / 'counts.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT / 'boundary-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
lines = ['# Confirmed pre-eleven coverage of T2375 evaluations','',
         'Post-scoring source-text audit. Descriptions, examples, labels and judge responses remain unchanged. No GPU inference or new judge calls were used.','',
         'All 100 unique scored windows were matched against the public UltraChat test_sft source, within the original held-out conversation partition. Matching allows only the known literal-versus-escaped added-token display difference. One repeated survey excerpt has two source positions; both give identical boundary classifications.','',
         'A case counts when an actually active token immediately precedes the target continuation, ignoring whitespace. In Fuzz and Active Fuzz that active boundary must also be among the highlighted positions. Each passage counts once.','',
         '| Evaluation | Standalone 11 / November / equivalents | Additional 2011 suffix cases | Total positive cases |',
         '|---|---:|---:|---:|']
for name, m in methods.items():
    lines.append(f"| {name} | {m['strict_eleven_or_equivalent_cases']} | {m['families'].get('year_2011_suffix',0)} | {m['confirmed_pre11_positive_cases']}/50 |")
lines += ['', 'The same positive passages are reused across all three methods; these are not three independent sets. Active Fuzz also reuses these passages for negative controls highlighting inactive tokens elsewhere. Those controls are not additional positive pre-eleven cases.','',
          '## Breakdown of the shared positive passages','']
for family,n in methods['fuzz']['families'].items():
    lines.append(f'- {family}: {n}')
lines += ['', '## Recovered examples','', '| Fuzz row | Active token | Actual continuation | Category |', '|---|---|---|---|']
for case in methods['fuzz']['cases']:
    e=case['active_boundary_events'][-1]
    display=lambda s:s.replace('\n',' ↵ ').replace('|','&#124;')
    lines.append(f"| {case['row_index']} | `{display(e['token'])}` | {display(e['following'][:100])} | {e['family']} |")
lines += ['', '## Corrections to the earlier excerpt-only audit','',
          'Eight excerpts ended after list item 10. The recovered source establishes which actually continue with item 11; an end of item 10 alone was not sufficient evidence. The visible November-the passage continues with 2nd, 1930 and must not be counted merely because November occurs in its prefix.','',
          'See counts.json for exact per-method row indexes, prefixes, continuations, activation values and highlighted-token checks. boundary-audit.json also retains inactive-boundary events. Source versions and SHA-256 hashes are in source-manifest.json.','']
(OUT / 'README.md').write_text('\n'.join(lines))
(OUT / 'source-manifest.json').write_bytes((DATA / 'source-manifest.json').read_bytes())
manifest = dict(files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json'])
(OUT / 'local-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({name:{k:v for k,v in m.items() if k!='cases'} for name,m in methods.items()},indent=2))
