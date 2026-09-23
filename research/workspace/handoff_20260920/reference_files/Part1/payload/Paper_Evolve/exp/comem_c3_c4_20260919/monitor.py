import datetime,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import hashlib,json,subprocess
from pathlib import Path
p=Path('/srv/encbank/comem_c3_c4_20260919')
def read(x):
    f=p/x
    return json.loads(f.read_text()) if f.exists() else None
r=dict(queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%M|%R'],universal_newlines=True),
       owner=read('eval_owner_status.json'),owner_failure=read('eval_owner_failure.json'),training={},quality={})
for j in [6,12,18]:
    base='training/j%d_s42/'%j;m=read(base+'metadata.json');progress=read(base+'progress.json')
    r['training'][str(j)]=dict(progress=progress,complete=read(base+'complete.json'),wait=read(base+'training_parent_exit.json'),
        parameters=m['trainable_parameters'] if m else None,init=m['initial_adapter_sha256'] if m else None,
        parameter_layout=read(base+'trainable_parameters.json'))
    r['quality'][str(j)]=dict(progress=read('quality/j%d_s42/progress.json'%j),complete=read('quality/j%d_s42/complete.json'%j))
r['queue']='\n'.join(l for l in r['queue'].splitlines() if '|qcm-c34-' in l)
print(json.dumps(r))
'''
def main():
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],capture_output=True,encoding='utf-8',timeout=35)
    assert r.returncode==0,r.stderr[-1000:]
    remote=json.loads(r.stdout)
    if (ROOT/'delivery/node_local_quality_20260919_1255/recovery.json').exists():
        from monitor_node_quality import poll
        recovered=poll()
        remote['canonical_owner_stale']=remote['owner']
        remote['canonical_owner_failure']=remote['owner_failure']
        remote['node_local_recovery']=dict(phase=recovered['phase'],sacct=recovered['sacct'],completed_depths=recovered['completed_depths'],failed_jobs=recovered['failed_jobs'])
        remote['owner']=dict(phase=recovered['phase'],source='node-local real completion + actual child wait + Slurm receipts',at=recovered['at'])
        remote['owner_failure']=dict(jobs=recovered['failed_jobs']) if recovered['failed_jobs'] else None
        remote['quality']={str(x['j']):dict(records=x['mirrored_records'],progress=x['files'].get('progress.json'),complete=x['files'].get('complete.json'),wait=x['files'].get('parent_exit.json'),job=x['job'],stage=x['stage'],slurm=x['slurm']) for x in recovered['records']}
    ready=[x for x in remote['training'].values() if x['init']]
    if len(ready)==3:
        assert len({x['init'] for x in ready})==1 and len({x['parameters'] for x in ready})==1
        assert all(x['parameter_layout']==ready[0]['parameter_layout'] for x in ready)
        remote['same_initialization_and_trainable_parameters']=True
    for x in remote['training'].values():x.pop('parameter_layout',None)
    local={}
    for p in (ROOT/'serving/rep0/32768').glob('*'):
        if p.is_dir():local[p.name]={n:json.loads((p/(n+'.json')).read_text()) for n in ['progress','complete','parent_exit'] if (p/(n+'.json')).exists()}
    report=dict(at=datetime.datetime.now().astimezone().isoformat(),remote=remote,serving=local)
    (ROOT/'monitor.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
