"""Start the authorized remote Judge once, retaining every saved valid decision."""
import json
import shlex
import subprocess
from probe_judge_service import ROOT, conversation_key

BOOTSTRAP = r'''
import datetime,json,os,pathlib,subprocess,sys
payload=json.load(sys.stdin)
root=pathlib.Path('/srv/encbank/encbank_new_backbones_formal_20260915')
out=root/'judge_gpt6_astra'/'formal_new_models'
out.mkdir(parents=True,exist_ok=True)
receipt=out/'remote_worker_launch.json'
if receipt.exists():
    old=json.loads(receipt.read_text())
    cmd=pathlib.Path('/proc')/str(old['pid'])/'cmdline'
    if cmd.exists() and b'locomo_judge_remote_worker.py' in cmd.read_bytes():
        print(json.dumps(dict(already_running=True,**old)))
        raise SystemExit(0)
now=datetime.datetime.now(datetime.timezone.utc)
stamp=now.strftime('%Y%m%d-%H%M%S')
history=out/'launch_history'/stamp
history.mkdir(parents=True)
for name in ('worker_status.json','status.json','remote_worker_launch.json'):
    p=out/name
    if p.exists(): (history/name).write_bytes(p.read_bytes())
env=os.environ.copy();env['MIDCACHE_JUDGE_API_KEY']=payload.pop('key')
stdout=out/('worker-'+stamp+'.stdout.log');stderr=out/('worker-'+stamp+'.stderr.log')
with stdout.open('ab') as o,stderr.open('ab') as e:
    process=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B',str(root/'locomo_judge_remote_worker.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)
data=dict(pid=process.pid,at=now.isoformat(),host='gpu-node1',user=os.environ.get('USER'),script='locomo_judge_remote_worker.py',stdout=str(stdout),stderr=str(stderr),credential_persisted=False,model='gpt-6-astra')
receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
'''

def main():
    command='/srv/encbank/Paper_Evolve/.venv/bin/python -c '+shlex.quote(BOOTSTRAP)
    key=conversation_key()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',command],
        input=json.dumps({'key':key}),text=True,capture_output=True,timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr.replace(key,'[REDACTED]')[:500])
    receipt=json.loads(result.stdout)
    (ROOT/'judge_gpt6_astra'/'remote_worker_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
