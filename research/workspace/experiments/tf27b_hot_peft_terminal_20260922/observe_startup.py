"""Read only; run on the actual Slurm node, passing that node alias as argv[1]."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
T=Path('/srv/encbank/qencbank_align_codex_20260911/tf27b_hot_live_20260921')
entry=next(x for x in json.loads((T/'submissions.json').read_text()) if x['label']=='hot24_peft_merged')
R=Path(entry['root']);job=entry['job']
def read(path):return json.loads(path.read_text()) if path.exists() else None
def proc(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return dict(alive=False)
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    env=(p/'environ').read_bytes().split(b'\0')
    return dict(alive=stat[0]!='Z',state=stat[0],parent_pid=int(stat[1]),start_ticks=int(stat[19]),
        slurm_job_id=next((x.split(b'=',1)[1].decode() for x in env if x.startswith(b'SLURM_JOB_ID=')),None),
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
queue=subprocess.check_output(['squeue','-r','-h','-j',job,'-o','%i|%T|%N|%M|%l'],text=True)
out=dict(epoch=time.time(),root=str(R),job_id=job,actual_host=socket.gethostname(),node_alias=sys.argv[1],queue=queue,
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,Timelimit,NodeList'],text=True),records={},errors=[])
for n in ['worker/environment_qualification_parent_exit.json','worker/model_launch.json','worker/trials_launch.json',
          'worker/ready.json','worker/merged_startup_qualification.json','worker/failure.json',
          'worker/integrated_failure.json','peft_merge_receipt.json','worker/status.json']:
    value=read(R/n)
    if value is not None:out['records'][n]=value
requests=list((R/'mailbox').glob('*/*.request.json'));responses=list((R/'mailbox').glob('*/*.response.json'))
out.update(requests=len(requests),responses=len(responses),launched_tasks=len(list((R/'jobs').glob('launch-*--hot.json'))))
if queue:
    assert queue.strip().split('|')[2]==sys.argv[1], 'Actual-node process audit on wrong host'
    model=read(R/'worker/model_launch.json')
    if model:
        mp=proc(model['pid']);out['model_process']=mp
        if not mp['alive'] or mp['slurm_job_id']!=job:out['errors'].append('model identity')
        candidates=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if p.stat().st_uid!=os.getuid():continue
                cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if str(R/'launch_trial_group.py') in cmd and 'python' in cmd:
                    evidence=proc(p.name)
                    if evidence.get('slurm_job_id')==job:candidates.append(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        out['controller_pids']=candidates
        if len(candidates)!=1 or mp.get('parent_pid') not in candidates:out['errors'].append('unique model supervisor')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
        out['model_gpu_process']=[line for line in apps.splitlines() if line.split(',')[0].strip()==str(model['pid'])]
        if out['model_gpu_process']:
            uuid=out['model_gpu_process'][0].split(',')[1].strip()
            out['gpu']=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
out['source_sha_errors']=[n for n,d in read(R/'source_manifest.json').items() if hashlib.sha256((R/n).read_bytes()).hexdigest()!=d]
out['response_proofs']=[]
for path in responses[:3]:
    d=read(path);request=path.with_name(path.name.replace('.response.','.request.'))
    out['response_proofs'].append(dict(task=path.parent.name,status=d['status'],generated_tokens=d.get('generated_tokens'),
        request_sha_valid=d['request_sha256']==hashlib.sha256(request.read_bytes()).hexdigest(),
        response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),variant=d.get('variant'),lora_execution=d.get('lora_execution')))
if not read(R/'worker/ready.json'):
    for n in ['worker/environment_qualification.stderr.log','worker/model.stderr.log']:
        if (R/n).exists():out[n]=(R/n).read_text()[-2500:]
print(json.dumps(out))
