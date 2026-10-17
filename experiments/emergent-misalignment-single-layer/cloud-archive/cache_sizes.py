import sys,json,re
sys.path.insert(0,'/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-single-layer')
import replacement_control as c
c.network();client=c.official_client();state=c.state()
instance=56373998 if len(sys.argv)==1 or sys.argv[1]=='source' else state['instance_id']
if instance==56373998:c.source(client)
else:c.target(client,state)
command='du -sb workspace/em-screen-venv workspace/.hf_home/hub/models--Qwen--Qwen3-4B'
c.account(client);response=client.put('/instances/command/'+str(instance)+'/',json_data={'command':command})
response.raise_for_status();url=response.json().get('result_url')
if not isinstance(url,str):raise SystemExit('Metadata result URL absent')
body=c.bounded_s3(url)
if len(body)>65536:raise SystemExit('Metadata exceeds 64KiB')
values={}
for line in body.decode().splitlines():
 match=re.fullmatch(r'(\d+)\s+.*?/diff/(workspace/(?:em-screen-venv|\.hf_home/hub/models--Qwen--Qwen3-4B))',line)
 if match:values['/'+match[2]]=int(match[1])
result={'instance_id':instance,'directory_apparent_bytes':values,'bounded_response_bytes':len(body)}
c.write(('source' if instance==56373998 else 'target')+'-cache-sizes.json',result)
print(json.dumps(result))
