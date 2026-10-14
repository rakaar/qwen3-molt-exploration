"""Restore the verified active-control NPZ files from the GitHub snapshot."""
import hashlib,json,subprocess
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[2]
 metadata=json.loads((root/'results/2026-10-14-active-small-hot-controls/data-backup.json').read_text())
 for name,record in metadata['files'].items():
  path=root/name
  if path.exists():
   assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'], 'Local file differs; preserve it before restoring: '+name
   continue
  data=subprocess.check_output(['git','show',metadata['git_commit']+':'+name],cwd=root)
  assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
  path.parent.mkdir(parents=True,exist_ok=True)
  temporary=path.with_suffix(path.suffix+'.restoring');temporary.write_bytes(data);temporary.replace(path)
 print('Verified/restored',len(metadata['files']),'binary files from',metadata['git_commit'])
if __name__=='__main__':main()
