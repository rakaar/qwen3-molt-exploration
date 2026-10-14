"""Select two hand-audited SQuAD2 passages before clean model inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/tmp/molt-squad-dev-v2.json')
EXPECTED = '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == EXPECTED
data = json.loads(DATA.read_text())
selection = [
    ('Geology', 0, ['5a57c667770dc0001aeefd66', '5a57c667770dc0001aeefd67',
                    '5a57c667770dc0001aeefd68', '5a57c667770dc0001aeefd69',
                    '5a57c667770dc0001aeefd6a']),
    ('Amazon_rainforest', 10, ['5a0c7fb2f5590b0018dab444', '5a0c7fb2f5590b0018dab445',
                             '5a0c7fb2f5590b0018dab446', '5a0c7fb2f5590b0018dab447',
                             '5a0c7fb2f5590b0018dab448']),
]
passages, cases = [], []
for passage_index, (title, index, unknown_ids) in enumerate(selection, 1):
    article = next(a for a in data['data'] if a['title'] == title)
    passage = article['paragraphs'][index]
    passage_id = f'P{passage_index}'
    passages.append({'id': passage_id, 'article_title': title, 'paragraph_index': index,
                     'text': passage['context']})
    known = [q for q in passage['qas'] if not q['is_impossible']]
    unknown = [next(q for q in passage['qas'] if q['id'] == i) for i in unknown_ids]
    assert len(known) == len(unknown) == 5
    for group, questions in [('answerable', known), ('unanswerable', unknown)]:
        for j, q in enumerate(questions, 1):
            assert bool(q['is_impossible']) == (group == 'unanswerable')
            for answer in q['answers']:
                start = answer['answer_start']
                assert passage['context'][start:start+len(answer['text'])] == answer['text']
            gold = list(dict.fromkeys(a['text'] for a in q['answers']))
            reference = gold[0] if gold else 'No supported answer in the passage.'
            note = ''
            if q['id'] == '572657d9dd62a815002e8232':
                reference = 'New magma (igneous rock may crystallize subsequently).'
                note = 'Original annotators include igneous; the immediate product stated in the passage is new magma.'
            cases.append({'id': f'{passage_id}-'+('A' if group == 'answerable' else 'U')+str(j),
                          'source_id': q['id'], 'passage_id': passage_id,
                          'question': q['question'], 'is_answerable': group == 'answerable',
                          'original_gold_answers': q['answers'], 'reference_answer': reference,
                          'reference_audit_note': note})
payload = {'dataset': 'SQuAD2.0 development set', 'source_url': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
           'source_sha256': EXPECTED, 'license': 'CC BY-SA 4.0',
           'attribution': 'Pranav Rajpurkar, Robin Jia, and Percy Liang, 2018',
           'selection': 'Two manually selected, reference-audited passages from the prior pilot article pool; nonrandom, selected before generation.',
           'screening': ['Jacksonville paragraph 0 has a city-versus-metropolitan population ambiguity.',
                         'Packet_switching paragraph 20 has duplicate ambiguous partnership questions.',
                         'These candidate passages were inspected without running the model and not selected.'],
           'passages': passages, 'cases': cases}
assert len(cases) == 20
(ROOT/'cases.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
print('Saved 2 passages, 10 answerable questions, and 10 unanswerable questions before inference.')
