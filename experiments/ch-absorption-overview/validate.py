"""Validate publication links, frozen claims, copied hashes, and credential exclusions."""
import argparse,csv,hashlib,json,re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        if tag in ['a','img','script','link']:
            ref=attrs.get('href') if tag in ['a','link'] else attrs.get('src')
            if ref:self.refs.append(ref)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    report=root/'results/2026-10-19-ch-absorption-overview';backup=json.loads((report/'backup-manifest.json').read_text())
    for f in backup['copied_files']:
        p=root/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
    pages=[report/'index.html']+[root/'results'/('2026-10-19-'+n)/'index.html' for n in [
        'ch-prefix-activation','ch-child-input-ablation','ch-child-all-occurrences-ablation','ch-prefix-probe',
        'ch-crosslayer-activation','ch-alllayer-causal-screen','ch-child-joint-ablation']]
    missing=[];checked=0
    for page in pages:
        parser=Links();parser.feed(page.read_text())
        for ref in parser.refs:
            url=urlsplit(ref)
            if url.scheme or url.netloc:continue
            target=(page.parent/unquote(url.path)).resolve() if url.path else page
            if target.is_dir():target=target/'index.html'
            if not target.exists():missing.append({'page':str(page.relative_to(root)),'href':ref});continue
            assert target.is_relative_to(root)
            if url.fragment and target.suffix=='.html':
                t=Links();t.feed(target.read_text());assert url.fragment in t.ids,(page,ref)
            checked+=1
    assert not missing,json.dumps(missing,indent=2)
    facts=json.loads((report/'facts.json').read_text())
    gate_proof=json.loads((report/'input-word-gates-verification.json').read_text())
    assert len(gate_proof['selected_features'])==8 and gate_proof['input_token_gate_values']==96
    assert gate_proof['original_seven_rows_unchanged'] and gate_proof['no_new_inference']
    assert gate_proof['L28_independent_cpu_gate_max_abs_error']<3e-5
    for path,digest in gate_proof['sources'].items():assert sha(root/path)==digest,path
    def gate_rows(path):
        with path.open(newline='') as stream:return list(csv.DictReader(stream))
    original_gates=gate_rows(root/'results/2026-10-19-ch-crosslayer-activation/input-word-gates.csv')
    updated_gates=gate_rows(report/'assets/input-word-gates.csv')
    def key(row):return (row['case_id'],row['position'],row['feature'])
    updated_lookup={key(row):row for row in updated_gates}
    assert len(updated_gates)==len(updated_lookup)==96 and len(original_gates)==84
    assert all(updated_lookup[key(row)]==row for row in original_gates)
    added=json.loads((report/'L28-T1123-input-gates.json').read_text())
    for row in added['records']:
        saved=updated_lookup[(row['case_id'],str(row['position']),row['feature'])]
        assert float(saved['gate'])==row['gate'] and row['gate']>0
    assert facts['gates']['L28-T1123']['child_gates']=={
        task:next(r['gate'] for r in added['records'] if r['task']==task and r['word']=='child')
        for task in ['uppercase','lowercase']}
    assert facts['probe']['test']['auroc']==0.9620606986555471
    assert facts['gates']['L3-T2334']['child_gates']=={'uppercase':0.,'lowercase':0.}
    assert len(facts['probe']['child_diagnostics'])==2 and all(d['prediction']==0 for d in facts['probe']['child_diagnostics'])
    assert facts['screen']['active_feature_site_pairs']==410
    assert all(r['answer']=='CH' for r in facts['taskrows']['upper'])
    assert all(r['answer']=='child: ch' for r in facts['taskrows']['lower'])
    assert next(r for r in facts['taskrows']['upper'] if r['condition']=='joint')['target_probability']==0.9227643013000488
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')
    flagged=[]
    for f in backup['copied_files']:
        p=root/f['path']
        if p.suffix in ['.json','.py','.md','.txt','.html','.csv'] and secret.search(p.read_text(errors='replace')):flagged.append(f['path'])
    assert not flagged,{'credential_like_files':flagged}
    assert not any(x['path'].endswith('.npz') and '/weights/' in x['path'] for x in backup['copied_files'])
    output={'copied_files_hash_verified':len(backup['copied_files']),'copied_bytes':backup['copied_bytes'],
        'local_links_checked':checked,'reports_checked':len(pages),'missing_links':missing,
        'frozen_measurement_claims_verified':True,'credential_pattern_matches':0,
        'input_word_gate_figure':{'features':8,'gate_values':96,'original_values_unchanged':84,
            'added_L28_values_verified':12,'added_gates_cpu_max_abs_error':gate_proof['L28_independent_cpu_gate_max_abs_error']},
        'model_weights_and_large_caches_excluded':True,'no_new_inference':True}
    (report/'publication-validation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
