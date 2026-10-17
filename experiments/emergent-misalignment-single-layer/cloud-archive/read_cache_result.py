import urllib.request,ssl,time,re
from pathlib import Path
class NR(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None
u=Path('/workspace/work/vast-storage-research/cache-result-url').read_text()
o=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://proxy:8080'}),urllib.request.HTTPSHandler(context=ssl.create_default_context()),NR())
for n in range(10):
 try:
  with o.open(u,timeout=15) as r:b=r.read(65537)
  assert len(b)<=65536
  break
 except urllib.error.HTTPError:
  if n==9:raise SystemExit('Bounded metadata export unavailable')
  time.sleep(2)
# This is a requested du directory-size listing, never startup/environment logs.
lines=b.decode(errors='strict').splitlines()
valid=[x for x in lines if re.fullmatch(r'[0-9.]+[KMGT]?\s+\.?/?[A-Za-z0-9_./-]+',x)]
print({'bounded_bytes':len(b),'directory_sizes':valid[:80]})
Path('/workspace/work/vast-storage-research/cache-du.txt').write_text('\n'.join(valid)+'\n')
Path('/workspace/work/vast-storage-research/cache-du.txt').chmod(0o600)
