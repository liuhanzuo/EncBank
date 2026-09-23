"""Read hardware and use the account's normal self-service user-daemon startup."""
import json,os,subprocess,time,socket
from pathlib import Path
R=Path(__file__).resolve().parent
env=os.environ.copy();env['XDG_RUNTIME_DIR']='/run/user/'+str(os.getuid());env['DBUS_SESSION_BUS_ADDRESS']='unix:path='+env['XDG_RUNTIME_DIR']+'/bus'
def run(argv):
    p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=30)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
rows=[run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free','--format=csv']),
    run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
    run(['loginctl','show-user',str(os.getuid()),'-p','Linger','-p','State'])]
if '--enable-own-linger' in __import__('sys').argv:
    result=run(['loginctl','enable-linger',str(os.getuid())]);rows.append(result)
    if result['returncode']==0:
        for _ in range(10):
            if Path(env['XDG_RUNTIME_DIR']+'/bus').exists():break
            time.sleep(1)
    rows.append(run(['systemctl','--user','is-system-running']))
out=dict(host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),bus_exists=Path(env['XDG_RUNTIME_DIR']+'/bus').exists(),rows=rows)
(R/'jobs'/('node_probe_'+socket.gethostname()+'.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
