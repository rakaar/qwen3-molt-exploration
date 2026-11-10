import json,os,signal,time
from pathlib import Path
root=Path(__file__).resolve().parent;p=json.loads((root/'extended-watchdog.json').read_text())
def active(pid,expected):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return fields[19]==expected and fields[0]!='Z'
 except FileNotFoundError:return False
m=p['monitor_pid'];w=p['worker_pid']
assert active(m,p['monitor_start']) and active(w,p['worker_start'])
os.kill(m,signal.SIGSTOP)
(root/'extended-watchdog-ready.json').write_text(json.dumps(dict(ready=True,deadline_unix=p['deadline_unix']))+'\n')
while active(w,p['worker_start']) and time.time()<p['deadline_unix']:time.sleep(2)
expired=active(w,p['worker_start'])
if expired and active(m,p['monitor_start']):
 os.killpg(m,signal.SIGTERM);os.kill(m,signal.SIGCONT);time.sleep(10)
 if active(w,p['worker_start']):os.killpg(m,signal.SIGKILL)
elif active(m,p['monitor_start']):os.kill(m,signal.SIGKILL)
(root/'extended-watchdog-outcome.json').write_text(json.dumps(dict(deadline_expired=expired,finished_unix=time.time()))+'\n')
