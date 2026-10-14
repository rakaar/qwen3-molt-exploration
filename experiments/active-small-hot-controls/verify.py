"""Independently verify writes, source hashes, and full-vocabulary probabilities."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('result_dir',type=Path);root=ap.parse_args().result_dir
 R=json.loads((root/'artifacts/results.json').read_text());assert R['status']=='complete'
 for name,digest in R['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 rows=R['rows'];count=len(rows);assert count in [550,240]
 edits=np.load(root/'artifacts/edits.npz');assert edits['actual'].shape==(count,2560)
 clean_bits=np.load(root/'artifacts/clean-state.npz')['receiver_logits_bits']
 clean_sha=hashlib.sha256(clean_bits.tobytes()).hexdigest()
 probabilities={};offset=0;max_error=0
 for chunk in R['logit_chunks']:
  path=root/'artifacts'/chunk['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==chunk['sha256']
  bits=np.load(path)['bits'];assert bits.shape==(chunk['rows'],151936) and bits.dtype==np.uint16
  decoded=(bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
  for j in range(len(bits)):
   row=rows[offset+j];assert row['logit_row']==offset+j
   assert hashlib.sha256(bits[j].tobytes()).hexdigest()==row['logit_sha256']
   z=decoded[j]-decoded[j].max();p=np.exp(z);p/=p.sum()
   assert int(p.argmax())==row['top5'][0]['id']
   error=abs(float(p[87072])-row['P_cold']);max_error=max(max_error,error);assert error<3e-6
   assert all(g['gate']>0 for g in row['source_gates']) and row['nonzero_actual_edit']
   assert np.isclose(np.linalg.norm(edits['actual'][offset+j].astype(np.float64)),row['actual_delta_norm'],rtol=1e-5,atol=1e-5)
   probabilities[row['id']]={'P_cold':float(p[87072]),'P_large':float(p[16767]),'logits_changed_from_clean':row['logit_sha256']!=clean_sha}
  offset+=len(bits)
 assert offset==count and all(R['clean_batch_checks'])
 old_path=Path('results/2026-10-13-small-hot-controls/artifacts/results.json')
 old=json.loads(old_path.read_text());old_selected=next(r for r in old['rows'] if r['id']=='selected_pair')
 selected=next((r for r in rows if r['id']=='selected_pair_4.5'),None)
 if selected:assert selected['logit_sha256']==old_selected['logit_sha256']
 result={'status':'verified','conditions':count,'all_actual_edits_nonzero':True,'all_source_gates_positive':True,
   'selected_reference_logits_bitwise_equal_to_original':True if selected else None,'all_source_hashes_match':True,
   'all_clean_batch_rows_bitwise_equal':True,'maximum_gpu_vs_cpu64_probability_difference':max_error,
   'probability_method':'CPU float64 softmax of saved full-vocabulary BF16 logits','probabilities':probabilities}
 (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Verified',count,'conditions, edit norms, and full-vocabulary probabilities.')
 if selected:print('Selected P(cold), CPU float64:',probabilities['selected_pair_4.5']['P_cold'])
if __name__=='__main__':main()
