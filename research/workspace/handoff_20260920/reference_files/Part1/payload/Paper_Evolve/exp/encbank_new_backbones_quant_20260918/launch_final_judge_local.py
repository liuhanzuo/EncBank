"""Launch prepared final Judge; credentials stay in transport/process memory."""
import json, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'encbank_new_backbones_formal_20260915'))
from probe_judge_service import conversation_key
STAGE='/tmp/qcm-q18-final-judge-20021-20260919-0840'
BOOTSTRAP=r'''
import datetime,json,os,subprocess,sys
from pathlib import Path
payload=json.load(sys.stdin)
p=Path('/tmp/qcm-q18-final-judge-20021-20260919-0840')
receipt=p/'launch.json'
if receipt.exists():
    old=json.loads(receipt.read_text());cmd=Path('/proc')/str(old['pid'])/'cmdline'
    if cmd.exists() and b'final_judge_local_runner.py' in cmd.read_bytes():
        print(json.dumps(dict(already_running=True,**old)));raise SystemExit(0)
    raise SystemExit('Previous local Judge exited; inspect before resuming.')
assert (p/'prepared.json').exists() and not (p/'parent_exit.json').exists()
env=os.environ.copy();env['MIDCACHE_JUDGE_API_KEY']=payload.pop('key')
env['TMPDIR']=str(p/'tmp');env['PYTHONPATH']=str(p)
env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
with (p/'parent.stdout.log').open('ab') as o,(p/'parent.stderr.log').open('ab') as e:
    child=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B','-u',str(p/'final_judge_local_runner.py'),'parent'],cwd=p,env=env,stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=str(p),cpu_only=True,workers=4,credential_persisted=False,model='gpt-6-astra',reasoning='low')
receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''
def main():
    key=conversation_key()
    command='/usr/bin/python3 -I -B -c '+shlex.quote(BOOTSTRAP)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',command],input=json.dumps({'key':key}),encoding='utf-8',capture_output=True,timeout=40)
    assert r.returncode==0,r.stderr.replace(key,'[REDACTED]')[:500]
    result=json.loads(r.stdout)
    (ROOT/'delivery/final_judge_local/launch.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
