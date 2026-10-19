"""Create deterministic, stem-family-disjoint CH datasets from a local dictionary."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT/'protocol.json').read_text())
OUT = ROOT/'artifacts'
OUT.mkdir(exist_ok=True)
TEMPLATES = {
    'fewshot_upper': 'Give the first two letters of each word in uppercase.\n\ngarden: GA\nwindow: WI\ntable: TA\n',
    'fewshot_lower': 'Give the first two letters of each word in lowercase.\n\ngarden: ga\nwindow: wi\ntable: ta\n',
    'transfer_upper': 'Write the first two letters of each word. Use uppercase letters only.\n\nplanet: PL\nriver: RI\nmusic: MU\n',
}


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def category(word):
    if word.startswith('ch'): return 'positive_ch'
    if word.startswith('c'): return 'c_not_h'
    if word.startswith('h'): return 'h_initial'
    if 'ch' in word: return 'internal_ch'
    return 'other'


def prepare():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args()
    assert not (OUT/'dataset.json').exists()
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    raw=json.loads((ROOT/'sources/word-families.json').read_text())
    word_to_family={r['word']:r['family'] for r in raw}
    for word in ['cheef']:
        word_to_family.setdefault(word,word)
    families=defaultdict(list)
    for word,family in word_to_family.items(): families[family].append(word)
    reserved_families={family for word,family in word_to_family.items()
        if any(word.startswith(prefix) for prefix in P['diagnostic_positive_prefixes'])
        or word in P['diagnostic_negative_words']}
    strings={}
    for token in range(len(tok)):
        surface=tok.decode([token])
        word=surface.removeprefix(' ')
        if re.fullmatch(r'[A-Za-z]{2,20}',word) and surface in [word,' '+word]:
            strings[surface]=token
    eligible={}
    for word,family in sorted(word_to_family.items()):
        forms=[]
        for spelling in sorted({word,word.title(),word.upper()}):
            for leading in ['', ' ']:
                surface=leading+spelling
                if surface in strings:
                    forms.append({'surface':surface,'token_id':strings[surface],
                        'case':'lower' if spelling.islower() else 'upper' if spelling.isupper() else 'title',
                        'leading_space':bool(leading)})
        if forms: eligible[word]={'word':word,'family':family,'category':category(word),'forms':forms}
    # Exact membership in the actual chat prompt is checked before admitting a word.
    rejected=[]
    for word,item in eligible.items():
        valid=[]
        for form in item['forms']:
            ok=True
            for prefix in TEMPLATES.values():
                user=prefix+form['surface']+':'
                chat=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
                start=chat.index(user)+len(prefix);end=start+len(form['surface'])
                positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>start and a<end]
                if len(positions)!=1 or enc['input_ids'][positions[0]]!=form['token_id']:
                    ok=False;break
            if ok:valid.append(form)
            else:rejected.append({'word':word,'surface':form['surface']})
        item['forms']=valid
    eligible={w:v for w,v in eligible.items() if v['forms']}
    by_family=defaultdict(list)
    for item in eligible.values():by_family[item['family']].append(item)
    family_category={f:category(sorted(items,key=lambda x:(len(x['word']),x['word']))[0]['word'])
                     for f,items in by_family.items()}
    assert all(len({int(i['word'].startswith('ch')) for i in items})==1 for items in by_family.values())
    positive=sorted(f for f in by_family if f not in reserved_families and family_category[f]=='positive_ch')
    assert len(positive)>=20, 'Too few independent CH stem groups for the planned split.'
    rng=random.Random(P['seed'])
    selected={'positive_ch':positive}
    rep_length=lambda f:min(len(i['word']) for i in by_family[f])
    pos_lengths=[rep_length(f) for f in positive];rng.shuffle(pos_lengths)
    for cat in P['negative_categories']:
        pool=sorted(f for f in by_family if f not in reserved_families and family_category[f]==cat)
        tie={f:rng.random() for f in pool};chosen=[]
        for length in pos_lengths:
            if not pool:break
            match=min(pool,key=lambda f:(abs(rep_length(f)-length),tie[f],f))
            chosen.append(match);pool.remove(match)
        selected[cat]=sorted(chosen)
    family_split={f:'diagnostic' for f in reserved_families if f in by_family}
    for cat,group in selected.items():
        group=group.copy();rng.shuffle(group)
        n=len(group);a=max(1,int(n*.70));b=max(a+1,int(n*.85));b=min(b,n-1)
        assert n>=5
        for i,f in enumerate(group):family_split[f]='train' if i<a else 'validation' if i<b else 'test'
    rows=[]
    for word,item in sorted(eligible.items()):
        if item['family'] not in family_split:continue
        split=family_split[item['family']]
        for form in item['forms']:
            templates=P['templates_train']+(P['templates_transfer'] if split in ['test','diagnostic'] else [])
            for template in templates:
                prefix=TEMPLATES[template];user=prefix+form['surface']+':'
                chat=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
                start=chat.index(user)+len(prefix);end=start+len(form['surface'])
                positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and b>start and a<end]
                assert len(positions)==1 and enc['input_ids'][positions[0]]==form['token_id']
                rows.append({'id':f'row-{len(rows):05d}','word':word,'family':item['family'],
                    'category':item['category'],'label':int(word.startswith('ch')),'split':split,
                    'template':template,'is_transfer':template in P['templates_transfer'],
                    **form,'user':user,'chat':chat,'prompt_token_ids':enc['input_ids'],
                    'target_position':positions[0]})
    main=[r for r in rows if not r['is_transfer']]
    family_sets={s:{r['family'] for r in main if r['split']==s} for s in ['train','validation','test','diagnostic']}
    for a,fa in family_sets.items():
        for b,fb in family_sets.items():
            if a!=b:assert not fa&fb
    for r in main:
        if any(r['word'].startswith(p) for p in P['diagnostic_positive_prefixes']):assert r['split']=='diagnostic'
    details={}
    for split in family_sets:
        subset=[r for r in main if r['split']==split]
        details[split]={'rows':len(subset),'words':len({r['word'] for r in subset}),
            'families':len({r['family'] for r in subset}),'positive_rows':sum(r['label'] for r in subset),
            'positive_families':len({r['family'] for r in subset if r['label']}),
            'category_rows':dict(Counter(r['category'] for r in subset)),
            'category_families':{c:len({r['family'] for r in subset if r['category']==c}) for c in selected}}
    dump('dataset.json',rows)
    dump('dataset-summary.json',{'rows_total':len(rows),'main':details,
        'transfer_rows':sum(r['is_transfer'] for r in rows),'all_eligible_dictionary_words':len(eligible),
        'eligible_nonreserved_ch_families':len(positive),'rejected_context_forms':len(rejected),
        'family_split_disjoint':True,'reserved_prefixes':P['diagnostic_positive_prefixes'],
        'family_split':family_split,'templates':TEMPLATES,'seed':P['seed']})
    dump('tokenizer-verification.json',{'files':{name:hashlib.sha256((args.model_dir/name).read_bytes()).hexdigest()
        for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']}})
    print(json.dumps({'rows':len(rows),'splits':details,'transfer_rows':sum(r['is_transfer'] for r in rows)},indent=2),flush=True)


if __name__=='__main__':prepare()
