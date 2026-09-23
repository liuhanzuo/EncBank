"""CPU-only node readout and local mirroring for recovered C4 evaluations."""
import datetime,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/node_local_quality_20260919_1255'
READ=r'''
import json
from pathlib import Path
items=ITEMS
result=[]
for item in items:
 root=Path(item['stage']);folder=root/'quality'/('j%d_s42'%item['j'])
 record=dict(job=item['job'],j=item['j'],stage=str(root),exists=root.exists(),files={})
 for name in ['stage.json','start.json']:
  f=root/name
  if f.exists():record['files'][name]=json.loads(f.read_text())
 for name in ['progress.json','protocol.json','complete.json','parent_exit.json','correctness.json','cache_location.json']:
  f=folder/name
  if f.exists():record['files'][name]=json.loads(f.read_text())
 for name in ['child.stderr.log','child.stdout.log']:
  f=root/name
  if f.exists():
   with f.open('rb') as s:s.seek(max(0,f.stat().st_size-2400));record[name]=s.read().decode('utf-8','replace')
 f=folder/'predictions.jsonl'
 if f.exists():
  data=f.read_bytes();record['predictions']=data[:data.rfind(b'\n')+1].decode('utf-8')
 else:record['predictions']=''
 for ext in ['out','err']:
  f=Path('/tmp')/('qcm-c34-local-20021-'+item['job']+'.'+ext)
  if f.exists():record['bootstrap_'+ext]=f.read_text()[-2400:]
 result.append(record)
print(json.dumps(result))
'''
REMOTE=r'''
import datetime,json,subprocess,sys
p=json.load(sys.stdin)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=35).stdout
acct=run(['sacct','-j',','.join(t['job'] for t in p['tasks']),'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,End'])
queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'])
jobs=[l.split('|') for l in queue.splitlines() if l.split('|')[1].startswith(('qcm-c34-','qcm-q18-'))]
assert sum(int(j[3].split(':')[-1]) for j in jobs)<=3,jobs
records=[]
for node in sorted({t['node'] for t in p['tasks']}):
 tasks=[t for t in p['tasks'] if t['node']==node]
 active=next((t['job'] for t in tasks if any(j[0]==t['job'] and j[2]=='RUNNING' for j in jobs)),None)
 if active:cmd=['srun','--jobid='+active,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none']
 elif all(any(j[0]==t['job'] and j[2]=='PENDING' for j in jobs) for t in tasks):
  records.extend(dict(job=t['job'],j=t['j'],stage=t['stage'],exists=False,files={},predictions='',not_started=True) for t in tasks);continue
 else:cmd=['srun','--partition=gpu','--nodelist='+node,'--job-name=qcm-local-readout','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','--time=00:02:00']
 records.extend(json.loads(run(cmd+['/usr/bin/python3','-I','-B','-c',p['read_code'].replace('ITEMS',repr(tasks))])))
print(json.dumps(dict(at=datetime.datetime.utcnow().isoformat()+'Z',sacct=acct,queue=jobs,records=records,cpu_only=True)))
'''
def write(path,data):
    temp=path.with_suffix(path.suffix+'.mirror-tmp');temp.write_bytes(data);temp.replace(path)
def poll():
    recovery=json.loads((OUT/'recovery.json').read_text())
    p=dict(tasks=recovery['tasks'],read_code=READ)
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(REMOTE)],input=json.dumps(p),capture_output=True,encoding='utf-8',timeout=85)
    assert proc.returncode==0,proc.stderr[-2500:]
    state=json.loads(proc.stdout);failed=[];done=[]
    for rec in state['records']:
        raw=rec.pop('predictions').encode('utf-8');dest=OUT/rec['job'];dest.mkdir(exist_ok=True)
        quality=ROOT/'quality'/('j%d_s42'%rec['j']);quality.mkdir(parents=True,exist_ok=True)
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==len({(r['id'],r['arm']) for r in rows}) and all(r['status']=='ok' and r['j']==rec['j'] for r in rows)
        if raw:
            original=(ROOT/'delivery/storage_failure_20260919_1255'/('j%d'%rec['j'])/'predictions.jsonl').read_bytes()
            assert raw.startswith(original),'Saved scientific predictions changed'
            for folder in [dest,quality]:
                old=folder/'predictions.jsonl'
                if old.exists():assert raw.startswith(old.read_bytes()),'Mirrored answer prefix changed'
                write(old,raw)
        rec['mirrored_records']=len(rows)
        for name,value in rec['files'].items():
            data=(json.dumps(value,indent=2)+'\n').encode('utf-8');write(dest/name,data)
            if name not in ['stage.json','start.json']:write(quality/name,data)
        for name in ['child.stderr.log','child.stdout.log','bootstrap_out','bootstrap_err']:
            if name in rec:write(dest/name,rec[name].encode('utf-8'))
        acct=next((l.split('|') for l in state['sacct'].splitlines() if l.startswith(rec['job']+'|')),None)
        rec['slurm']=acct
        wait=rec['files'].get('parent_exit.json');complete=rec['files'].get('complete.json')
        if acct and (acct[1] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] or (wait and wait['returncode']!=0)):failed.append(rec['job'])
        if complete and wait and acct and acct[1:3]==['COMPLETED','0:0']:
            assert complete['records']==800 and complete['errors']==0 and len(rows)==800 and wait['actual_wait'] and wait['returncode']==0
            done.append(rec['j'])
    state['phase']='FAILED' if failed else ('COMPLETE' if sorted(done)==[6,12,18] else 'ACTIVE')
    state['failed_jobs']=failed;state['completed_depths']=done
    write(OUT/'monitor_state.json',(json.dumps(state,indent=2)+'\n').encode('utf-8'))
    return state
if __name__=='__main__':print(json.dumps(poll()))
