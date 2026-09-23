"""Same authorized API-key transport; secret sent through SSH stdin only."""
import json, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'comem_new_backbones_formal_20260915'))
from probe_judge_service import conversation_key

BOOTSTRAP=r'''
import datetime,json,os,subprocess,sys
from pathlib import Path
payload=json.load(sys.stdin)
p=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
out=p/'judge_gpt6_astra';out.mkdir(parents=True,exist_ok=True)
receipt=out/'launch.json'
if receipt.exists():
    old=json.loads(receipt.read_text());cmd=Path('/proc')/str(old['pid'])/'cmdline'
    if cmd.exists() and b'judge_watch.py' in cmd.read_bytes():print(json.dumps(dict(already_running=True,**old)));raise SystemExit(0)
    raise SystemExit('Previous watcher exited; inspect before resuming.')
tmp=p/'cache/judge-tmp';tmp.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['MIDCACHE_JUDGE_API_KEY']=payload.pop('key');env['TMPDIR']=str(tmp)
env['PYTHONPATH']=str(p);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
with (out/'watch.stdout.log').open('ab') as o,(out/'watch.stderr.log').open('ab') as e:
    child=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-u','-B',str(p/'judge_watch.py')],cwd=p,env=env,stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),credential_persisted=False,model='gpt-6-astra',reasoning='low')
receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''

def main():
    key=conversation_key()
    cmd='/srv/encbank/Paper_Evolve/.venv/bin/python -c '+shlex.quote(BOOTSTRAP)
    r=subprocess.run(['ssh','-o','BatchMode=yes','gpu-node1',cmd],input=json.dumps({'key':key}),text=True,capture_output=True,timeout=40)
    assert r.returncode==0,r.stderr.replace(key,'[REDACTED]')[:500]
    receipt=json.loads(r.stdout);(ROOT/'judge_watch_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
