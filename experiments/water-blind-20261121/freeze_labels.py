"""Record the text-only judgments made on the seeded, ordered sample."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
out=Path(__file__).resolve().parents[2]/'results/2026-11-21-water-blind'
yes=set(map(int,'''1 10 11 16 21 22 25 26 27 30 31 33 34 35 38 46 51 55 59 67 68 72 74 79 83 84 85 93 102 108 109 112 116 123 125 128 130 134 140 142 144 147 151 152 157 158 159 166 168 172 175 177 178 182 183 184 185 189 191 198 200 202 207 214 215 220 221 224 226 230 236 238 240'''.split()))
notes={5:'Possessive enclitic before surface.',28:'Repellent functions as an adjective modifying neoprene material.',38:'White water is the medium of rafting.',46:'Cornstarch and water jointly modify mixture.',93:'Air and water jointly modify pollution.',99:'Flowing is a participle with water as its subject.',116:'White water is the medium of rafting.',117:'Denver Water is an organization name modifying proposal.',154:'Verb with a direct object.',159:'Water Regatta is the event type in this name.',166:'Air and water jointly modify quality.',179:'Verb with a direct object.',201:'Trickling is a participle with water as its subject.',209:'Water and preserve jointly modify views, but preserve intervenes before the head views.',212:'Consuming is participial before techniques.',223:'Imperative verb.'}
s=(out/'sample.json').read_bytes()
rows=json.loads(s)
assert {r['id'] for r in rows} == set(range(1,241))
labels=[dict(id=r['id'],conversation=r['conversation'],position=r['position'],next_noun_modifier=r['id'] in yes,note=notes.get(r['id'],'')) for r in rows]
assert not (out/'gates.json').exists() and not (out/'labels.json').exists()
b=(json.dumps(labels,indent=2)+'\n').encode()
(out/'labels.json').write_bytes(b)
(out/'label-freeze.json').write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),annotator='Codex, one text-only pass',sample_sha256=hashlib.sha256(s).hexdigest(),labels_sha256=hashlib.sha256(b).hexdigest(),modifier=len(yes),other=len(rows)-len(yes)),indent=2)+'\n')
