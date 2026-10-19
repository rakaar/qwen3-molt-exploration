"""Copy the CH investigation's code and compact scientific records into its Git checkout."""
import argparse,hashlib,json,shutil
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
NAMES=['ch-prefix-activation','ch-child-input-ablation','ch-child-all-occurrences-ablation',
       'ch-prefix-probe','ch-crosslayer-activation','ch-alllayer-causal-screen','ch-child-joint-ablation']
EXTENSIONS={'.py','.json','.csv','.txt','.md','.png','.svg','.html','.npz'}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish-root',type=Path,required=True);a=ap.parse_args()
    assert (a.publish_root/'.git').exists()
    roots=[PROJECT/'experiments'/n for n in NAMES]+[PROJECT/'results'/('2026-10-19-'+n) for n in NAMES]
    roots.append(PROJECT/'comparisons/2026-10-19-ch-later-layer-search')
    kept=[];omitted=[]
    for root in roots:
        assert root.is_dir(),root
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=p.relative_to(PROJECT);size=p.stat().st_size
            reason=None
            if p.suffix not in EXTENSIONS:reason='Transport archive, log, or runtime-only file; scientific records kept separately.'
            elif p.suffix=='.npz' and any(x in p.parts for x in ['weights','gates','gate-subsets','capture']):reason='Released weights or activation cache retained locally; source hashes and extraction code preserved.'
            elif p.name=='activations.npz' or size>5_000_000:reason='Large native activation cache retained locally and reproducible from pinned model; dataset, fitted probe, predictions, metrics and hashes preserved.'
            if reason:
                omitted.append({'path':str(rel),'bytes':size,'reason':reason});continue
            dest=a.publish_root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            digest=sha(p);assert sha(dest)==digest
            kept.append({'path':str(rel),'bytes':size,'sha256':digest})
    out=a.publish_root/'results/2026-10-19-ch-absorption-overview';out.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'Seven CH spelling experiments, their published measurements, and the selected later-layer label survey.',
        'copied_files':kept,'copied_count':len(kept),'copied_bytes':sum(x['bytes'] for x in kept),
        'omitted_local_files':omitted,'omitted_bytes':sum(x['bytes'] for x in omitted),
        'large_weights_and_native_activation_caches_preserved_locally':True,
        'no_new_inference_or_lifecycle_action':True}
    (out/'backup-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'BACKUP.md').write_text('''# CH absorption investigation backup

This GitHub backup preserves all seven stages' code, exact spelling prompts,
probe dictionary and licenses, train/validation/test/diagnostic membership,
fitted linear-probe coefficients, predictions, metrics, natural-gate measurements,
ablation outcomes and compact intervention vectors, scientific plots, and
verification/provenance JSON. `backup-manifest.json` records copied hashes and
every intentionally omitted local path.

Released model/MOLT weights, full gate matrices, transport archives, runtime logs,
and large native activation caches remain in the original local experiment tree.
They were not uploaded. Existing manifests may name these locally retained files;
their absence from GitHub is intentional, not an incomplete experiment run.
The source protocols pin the checkpoint revisions needed to regenerate them.

The source scripts retain their recorded computation and hashes. Some old report
builders require the omitted local inputs to repeat all independent numerical
checks. The new overview can be rebuilt using the saved JSON and figures alone.

No new GPU inference or resource lifecycle action was performed for publication.
''')
    print(json.dumps({'files':len(kept),'MB':round(manifest['copied_bytes']/1e6,2),'omitted_MB':round(manifest['omitted_bytes']/1e6,2)}))

if __name__=='__main__':main()
