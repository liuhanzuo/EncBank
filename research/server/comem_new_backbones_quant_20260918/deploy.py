"""Copy explicit non-secret package files and launch one CPU admission owner."""
import json, shlex, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'

def main():
    subprocess.run([sys.executable,str(ROOT/'make_plan.py')],check=True)
    manifest=json.loads((ROOT/'package_manifest.json').read_text())
    check="from pathlib import Path; p=Path("+repr(REMOTE)+"); assert p.resolve().is_relative_to(Path('/srv/encbank')); p.mkdir(parents=True,exist_ok=True)"
    subprocess.run(['ssh','-o','BatchMode=yes','gpu-node1',PY+' -c '+shlex.quote(check)],check=True,timeout=40)
    files=[ROOT/n for n in manifest]+[ROOT/'package_manifest.json',ROOT/'cpu_checks.json']
    subprocess.run(['scp','-q',*[str(p) for p in files],'gpu-node1:'+REMOTE+'/'],check=True,timeout=240)
    launch=r'''
import datetime,json,os,subprocess
from pathlib import Path
p=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
r=p/'coordinator_launch.json'
if r.exists():
    old=json.loads(r.read_text());cmd=Path('/proc')/str(old['pid'])/'cmdline'
    if cmd.exists() and b'coordinator.py' in cmd.read_bytes():
        print(json.dumps(dict(already_running=True,**old)));raise SystemExit(0)
    raise SystemExit('Previous owner exited. Inspect its receipts before restarting.')
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (p/'coordinator.stdout.log').open('ab') as out,(p/'coordinator.stderr.log').open('ab') as err:
    child=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B','-u',str(p/'coordinator.py')],cwd=p,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(p),role='CPU-only shared GPU admission owner',maximum_gpu_requests=4)
r.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','gpu-node1',PY+' -c '+shlex.quote(launch)],text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout);(ROOT/'coordinator_launch.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))

if __name__=='__main__':main()
