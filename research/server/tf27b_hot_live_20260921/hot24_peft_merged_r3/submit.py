"""Submit exactly one explicitly authorized new Hot24-merged benchmark group."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
T=R.parent
with (T/'hot24_peft_submission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert not (R/'submission.json').exists(),'Existing/uncertain submission; never duplicate'
    p=json.loads((R/'plan.json').read_text())
    assert json.loads((R/'peft_preflight.json').read_text())['passed']
    for name,sha in json.loads((R/'source_manifest.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
    records=json.loads((T/'submissions.json').read_text())
    assert all(x['root']!=str(R) for x in records)
    previous=[x for x in records if x['label']=='hot24_peft_merged']
    assert len(previous)==1 and previous[0]['job']=='122033'
    proof=json.loads((R/'environment_recovery.json').read_text())
    assert proof['model_calls']==0 and proof['qualification_container_closed']
    assert not list(Path(previous[0]['root']).joinpath('mailbox').glob('*/*.request.json'))
    queue=subprocess.check_output(['squeue','-r','-h','-u','liuhanzuo','-o','%i|%j|%T|%b|%N'],text=True)
    assert '121985|' not in queue and 'qencbank-kernel-' not in queue,'Wait for isolated speed pilot release'
    assert 'tf27b-tb-hot24-peft' not in queue and '122033|' not in queue
    formal=[x for x in queue.splitlines() if ('qencbank-tb-' in x or 'qencbank-agentmem-' in x) and 'gpu' in x.split('|')[3]]
    assert len(formal)<=4,formal
    cmd=['sbatch','--parsable','--job-name=tf27b-tb-hot24-peft','--partition=gpu',
        '--time=UNLIMITED','--no-requeue','--cpus-per-task=56','--mem=160G','--gres=gpu:nvidia_l20d:1',
        '--exclude=gpu-node1,gpu-node3,gpu-node6','--chdir='+str(R),
        '--output='+str(R/'logs/slurm-%j.out'),'--error='+str(R/'logs/slurm-%j.err'),
        '--wrap',p['engine_python']+' -B '+str(R/'launch_trial_group.py')]
    row=dict(label='hot24_peft_merged',arm='hot',variant=p['variant'],concurrency=24,hot_chunks=24,
        lora_execution='peft_merged_bf16',root=str(R),argv=cmd,epoch=time.time(),status='intent',
        authorization=p['authorization'],full89_jobs_unchanged=formal,added_gpu=1,
        supersedes_speed_pilot='121985 (user redirect, evidence retained)',previous_attempt='122033',previous_root=previous[0]['root'],automatic_retries=0)
    with (R/'submission.json').open('x') as f:json.dump(row,f,indent=2)
    submission_env=os.environ.copy();submission_env.update(json.loads((R/'submission_environment.json').read_text())['proxy_environment'])
    result=subprocess.run(cmd,capture_output=True,text=True,env=submission_env)
    row.update(status='submitted' if result.returncode==0 else 'submission_failed',exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    if result.returncode==0:row['job']=row['job_id']=result.stdout.strip().split(';')[0]
    (R/'submission.json').write_text(json.dumps(row,indent=2)+'\n')
    assert result.returncode==0,result.stderr
    records=json.loads((T/'submissions.json').read_text())
    assert all(x['root']!=str(R) for x in records)
    index=next(i for i,x in enumerate(records) if x['label']=='hot24_peft_merged')
    assert records[index]['job']=='122033'
    row['attempt_history']=[records[index]]
    records[index]=row
    tmp=T/('submissions.json.hot24peft.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(records,indent=2)+'\n');tmp.replace(T/'submissions.json')
    print(json.dumps(row))
