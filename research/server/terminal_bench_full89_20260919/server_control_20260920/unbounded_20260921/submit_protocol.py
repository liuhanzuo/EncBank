"""Qualify and submit exactly one immutable server job per arm under the 4-GPU cap."""
import argparse,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
from prepare_protocol import prepare,S,U

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def materialize(arm):
    selection=read(U/'selection.json');assert selection['status']=='PASS'
    name=arm+'_unbounded_20260921';h=S/name
    prepare(arm,name,selection['arms'][arm]['tasks'])
    p=read(h/'plan.json')
    p.update(protocol_selection_sha256=sha(U/'selection.json'),
        source_handoff_audits=[sha(U/'handoff_audit.json'),sha(U/'dense_handoff_audit.json')],
        host_admission='Across the three arms, reserve at most200GiB official task RAM on one node; also check fresh host and own allocation headroom. Eight tasks per arm maximum.',
        predecessor_jobs_closed=['112400','112403','114684','115237'],
        timing_boundary='Record setup, agent, verifier, transport, model and wall durations without task or per-call time cancellation.',
        authorization='User explicitly authorized no token/time ceiling, protocol replacement, truncated-task replay and completion of remaining tasks.',
        recovery_boundary='Fresh unlimited-time/EOS-native-context protocol. Previous results and invalid attempts remain immutable.',
        protocol_boundary='Server-only managed Apptainer; frozen model/adapter/sampling/task resources, generation/time policy changed by user request.')
    save(h/'plan.json',p)
    source=S/'unbounded_env_qualification_r3_20260921/container_qualification.json';q=read(source)
    assert q['status']=='PASS' and q['model_calls']==0 and len(q['checks'])==89
    for n,digest in q['backend_hashes'].items():assert sha(h/n)==digest,(arm,n)
    selected=[r for r in q['checks'] if r['task'] in p['tasks']]
    assert len(selected)==len(p['tasks']) and all(r['status']=='PASS' for r in selected)
    q.update(tasks=p['tasks'],checks=selected,plan_sha256=sha(h/'plan.json'),
        qualification_transfer=dict(source=str(source),sha256=sha(source),same_backend_bytes=True,
            same_frozen_task_image_resources_network=True,same_Harbor_setup_path=True,
            qualification_hostname='gpu-host',source_all89_model_calls=0))
    save(h/'container_qualification.json',q)
    save(h/'selection_provenance.json',dict(selection_path=str(U/'selection.json'),sha256=sha(U/'selection.json'),arm=arm,tasks=p['tasks']))
    manifest={f.name:sha(f) for f in h.iterdir() if f.is_file() and f.name!='server_source_manifest.json'}
    save(h/'server_source_manifest.json',manifest)
    return h
def preflight(h):
    p=read(h/'plan.json')
    args=[p['harbor_python'],str(h/'server_preflight.py')]
    if p['arm']=='encbank':args.append('--checkpoint')
    with (h/'submission_preflight.stdout.log').open('x') as out,(h/'submission_preflight.stderr.log').open('x') as err:
        result=subprocess.run(args,cwd=h,stdout=out,stderr=err)
    assert result.returncode==0,(str(h),read(h/'server_preflight.json'))
    print(json.dumps(dict(arm=h.name,preflight='PASS')),flush=True)
def submit(h):
    p=read(h/'plan.json');arm='dense' if p['arm']=='dense' else 'k'+str(p['top_k_chunks'])
    with (S.parent.parent/'terminal_bench_20260918/admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (h/'submission.json').exists()
        for n,digest in read(h/'server_source_manifest.json').items():assert sha(h/n)==digest,n
        q=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
        owned=[l for l in q.splitlines() if any(n in l for n in ['qencbank-tb-','qencbank-agentmem-']) and 'gpu' in l.split('|')[-1].lower()]
        assert len(owned)<4,owned
        assert all(any(a+'_unbounded_20260921'==x.name and str(read(x/'submission.json').get('job_id'))==l.split('|')[0] for a in ['dense','k12','k48'] for x in [S/(a+'_unbounded_20260921')] if (x/'submission.json').exists()) for l in owned),owned
        assert not any(p['job_name']==l.split('|')[1] for l in owned),owned
        old=subprocess.check_output(['squeue','-h','-j','112400,112403,114684,115237','-o','%i'],text=True);assert not old.strip(),old
        r=dict(status='intent',epoch=time.time(),arm=arm,root=str(h),tasks=len(p['tasks']),task_concurrency=8,gpu_requests=1,
            source_manifest_sha256=sha(h/'server_source_manifest.json'),selection_sha256=sha(U/'selection.json'),prior_owned_gpu_queue=owned,
            server_only=True,requested_time_limit='UNLIMITED',configured_generation_token_cap=None,request_timeout_seconds=None)
        save(h/'submission.json',r)
        child=subprocess.run(['sbatch','--parsable',str(h/'server.slurm')],capture_output=True,text=True)
        r.update(exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,actual_parent_wait=True,status='submitted' if child.returncode==0 else 'submission_failed')
        if child.returncode==0:r['job_id']=child.stdout.strip().split(';')[0]
        save(h/'submission.json',r);assert child.returncode==0,r
        print(json.dumps(r),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','preflight','submit']);parser.add_argument('arms',nargs='*',default=['dense','k12','k48']);args=parser.parse_args()
    for arm in args.arms:
        assert arm in ['dense','k12','k48']
        if args.operation=='prepare':materialize(arm)
        elif args.operation=='preflight':preflight(S/(arm+'_unbounded_20260921'))
        else:submit(S/(arm+'_unbounded_20260921'))
if __name__=='__main__':main()
