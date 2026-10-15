"""Read-only inventory of this rental's scientific/runtime files; omit caches and redownloadable assets."""
from pathlib import Path
import hashlib,json
roots=['vampire-intermediate','six-new-cases','privacy-activation-pilot','privacy-zero-clamp','privacy-query-clamp','privacy-authorized-activation','privacy-candidate-activation-screen','privacy-system-api-key','privacy-api-key-steering','safety-refusal-gates-20261015','safety-refusal-causal-20261015']
exclude={'models','runtime-wheels','runtime-wheel-parts','__pycache__','.git','.cache'}
rows=[]
for name in roots:
 root=Path('/root')/name
 if not root.exists():continue
 for p in sorted(root.rglob('*')):
  if p.is_file() and not any(part in exclude for part in p.relative_to(root).parts):
   with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   rows.append({'path':str(p.relative_to('/root')),'bytes':p.stat().st_size,'sha256':digest})
print(json.dumps({'roots':roots,'excluded_redownloadable_or_temporary_dirs':sorted(exclude),'files':rows},indent=2))
