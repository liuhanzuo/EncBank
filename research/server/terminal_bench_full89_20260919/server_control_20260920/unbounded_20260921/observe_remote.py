"""Read-only experiment audit; writes nothing to scientific inputs or results."""
import collections,datetime,hashlib,json,subprocess,time
from pathlib import Path

U=Path(__file__).resolve().parent;S=U.parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seconds(start,end=None):
    if not start:return None
    a=datetime.datetime.fromisoformat(start.replace('Z','+00:00'))
    b=datetime.datetime.fromisoformat(end.replace('Z','+00:00')) if end else datetime.datetime.now(datetime.timezone.utc)
    return (b-a).total_seconds()
def main():
    selection=read(U/'selection.json');arms={};runs={}
    registry=read(U/'scale4_20260921/registry.json')
    specs=registry['runs'] if registry else [dict(id=a,arm=a,root=str(S/(a+'_unbounded_20260921')),tasks=selection['arms'][a]['tasks']) for a in ['dense','k12','k48']]
    containers=collections.defaultdict(list)
    roots={spec['root']:spec['id'] for spec in specs}
    managed=Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920/managed_instances')
    for spec_path in managed.glob('*/spec.json'):
        spec=read(spec_path);arm=roots.get(str(Path(spec.get('startup_lock','/missing')).parent))
        if arm is None:continue
        logs=next((source for source,target in spec.get('binds',[]) if target=='/logs/agent'),None)
        if not logs:continue
        service=read(spec_path.parent/'service.json') or {};closure=read(spec_path.parent/'closure.json')
        containers[arm].append(dict(task=Path(logs).parent.parent.name,name=spec['name'],root=str(spec_path.parent),
            owner_pid=spec['owner_pid'],owner_ticks=spec.get('owner_ticks'),memory_mb=spec['memory_mb'],
            role='qualification' if 'node_runtime_qualification' in Path(logs).parts else 'benchmark',
            memory_max=service.get('memory_max'),affinity=spec['affinity'],network_policy=spec['network_policy'],closure=closure))
    submissions={spec['id']:read(Path(spec['root'])/'submission.json') or dict(job_id=None,epoch=0,status='not_submitted') for spec in specs}
    jobs=','.join(str(s['job_id']) for s in submissions.values() if s.get('job_id'))
    queue=subprocess.check_output(['squeue','-r','-h','-j',jobs,'-o','%i|%T|%M|%l|%R'],text=True)
    accounting=subprocess.check_output(['sacct','-X','--noheader','-P','-j',jobs,'--format=JobID,State,ExitCode,Elapsed,Timelimit,NodeList'],text=True)
    for spec in specs:
        arm=spec['arm'];run_id=spec['id'];submission=submissions[run_id]
        h=Path(spec['root']);p=read(h/'plan.json');o=h/'execution';run=h/('run_'+p['arm']);box=Path(p['rpc_root'])/p['arm']
        assert set(spec['tasks']).issubset(p['tasks'])
        p['tasks']=spec['tasks']
        state=read(o/'status.json') or {};events=[]
        if (run/'events.jsonl').exists():
            for line in (run/'events.jsonl').read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:pass
        released={e['task_id'] for e in events if e.get('event')=='session_release'}
        tasks={};integrity_errors=[];total_tokens=0;by_task=collections.defaultdict(list)
        for f in box.glob('*.request.json'):
            req=read(f);by_task[req['task']].append((f,req))
        ownership_errors=sorted(set(by_task)-set(p['tasks']))
        ownership_errors+=sorted({f.stem for f in (o/'launches').glob('*.json')}-set(p['tasks']))
        for name in p['tasks']:
            reqs=by_task[name]
            row=dict(task=name,state='not_started',requests=len(reqs),responses=0,generated_tokens=0,model_seconds=0,truncated=0,H_check_failures=0,reply_proofs_ok=True,native_context_events=[])
            for f,req in reqs:
                response=f.with_name(f.name.replace('.request.','.response.'));broker=f.with_name(f.name.replace('.request.','.broker.'));r=read(response)
                if r is None:continue
                row['responses']+=1;row['generated_tokens']+=r.get('generated_tokens',0);row['model_seconds']+=r.get('request_seconds',0) or 0
                row['truncated']+=bool(r.get('hit_generation_cap') or r.get('hit_context_capacity') or r.get('status') in ['deadline','context_limit'])
                if r.get('status')=='context_limit':
                    actual_input=max(r.get('prompt_tokens',0),r.get('actual_input_tokens',0))
                    generated=r.get('generated_tokens',0)
                    row['native_context_events'].append(dict(request_id=req['request_id'],step=req['step'],
                        prompt_tokens=r.get('prompt_tokens'),actual_input_tokens=r.get('actual_input_tokens'),generated_tokens=generated,
                        native_context_tokens=p['context_tokens'],capacity_verified=actual_input+generated>=p['context_tokens'],
                        response_sha256=sha(response)))
                row['H_check_failures']+=bool(arm!='dense' and r.get('status')=='ok' and r.get('state_hashes_unchanged') is not True)
                b=read(broker)
                if b:
                    pr=b.get('proof',{})
                    if pr.get('sha256')!=sha(response) or pr.get('bytes')!=response.stat().st_size:
                        integrity_errors.append(req['request_id']);row['reply_proofs_ok']=False
                else:row['reply_proofs_ok']=False
            total_tokens+=row['generated_tokens']
            launch=read(o/'launches'/(name+'.json'));outcome=read(o/'task_outcomes'/(name+'.json'));receipt=read(o/'receipts'/(name+'.json'))
            row['parent_receipt']=receipt
            if launch:row.update(state='active',started_epoch=launch['epoch'],elapsed_seconds=time.time()-launch['epoch'])
            if outcome:
                row.update(state='closed_needs_audit',outcome=outcome)
                if receipt and launch:row.update(ended_epoch=receipt['epoch'],elapsed_seconds=receipt['epoch']-launch['epoch'])
                result_paths=[Path(x['path']) for x in outcome['results']]
                if len(result_paths)==1:
                    result=read(result_paths[0]);row.update(reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward'),exception_type=(result.get('exception_info') or {}).get('exception_type'),result_sha256=sha(result_paths[0]))
                    row['reported_reward']=row['reward']
                    row['phase_seconds']={k:seconds(v.get('started_at'),v.get('finished_at')) for k,v in result.items() if k in ['environment_setup','agent_setup','agent_execution','verifier','verifier_execution'] and isinstance(v,dict)}
                    row['trial_seconds']=seconds(result.get('started_at'),result.get('finished_at'))
                release_ok=arm=='dense' or all(req['task_id'] in released for _,req in reqs)
                row['sessions_released']=release_ok
                task_containers=[c for c in containers[run_id] if c['task']==name and c['role']=='benchmark']
                row['containers_closed']=bool(task_containers) and all(c.get('closure',{}).get('cgroup_empty') if c.get('closure') else False for c in task_containers)
                closure_verified=bool(receipt and receipt['actual_parent_wait'] and not receipt['transport_errors'] and row['reply_proofs_ok'] and row['responses']==row['requests'] and not row['H_check_failures'] and release_ok and row['containers_closed'])
                if outcome['valid_result'] and closure_verified and not row['truncated']:
                    row['state']='verified_complete'
                elif closure_verified and row.get('exception_type')=='ContextLengthExceededError' and row['native_context_events'] and all(e['capacity_verified'] for e in row['native_context_events']):
                    row.update(state='verified_context_failure',reported_reward=0.0,failure_reason='NATIVE_CONTEXT_CAPACITY',
                        adjudication='User policy 2026-09-21: count native-context exhaustion as failure; preserve original verifier/exception evidence.')
                elif not outcome['valid_result']:row['state']='incomplete'
            interrupted=read(o/'interrupted_parents'/(name+'.json'))
            if row['state']=='active' and interrupted and interrupted.get('actual_parent_wait'):
                row.update(state='infrastructure_interrupted',interruption_parent_receipt=interrupted,
                    classification='INFRASTRUCTURE_INTERRUPTION',automatic_retry=False)
                row.pop('elapsed_seconds',None)
                parent=read(h/'server_job_receipt.json') or {}
                if launch and parent.get('actual_parent_waits'):
                    row['ended_epoch_upper_bound']=parent['epoch']
                    row['elapsed_seconds_upper_bound']=parent['epoch']-launch['epoch']
                owned=[c for c in containers[run_id] if c['task']==name and c['role']=='benchmark']
                row['containers_closed']=bool(owned) and all((c.get('closure') or {}).get('cgroup_empty') for c in owned)
                row['sessions_released']=arm=='dense' or all(req['task_id'] in released for _,req in reqs)
            tasks[name]=row
        complete=[v for v in tasks.values() if v['state'] in ['verified_complete','verified_context_failure']]
        context_failed=[v for v in complete if v['state']=='verified_context_failure']
        new_passed=sum(v['reported_reward']==1 for v in complete);new_failed=sum(v['reported_reward']==0 for v in complete)
        runs[run_id]=dict(arm=arm,root=str(h),role=spec.get('role'),job_id=submission.get('job_id'),submission_status=submission.get('status'),planned=len(p['tasks']),retained=0,new_verified=len(complete),
            new_normal_completed=len(complete)-len(context_failed),new_context_failures=len(context_failed),new_passed=new_passed,new_failed=new_failed,
            cumulative_passed=new_passed,cumulative_failed=new_failed,
            cumulative_verified=len(complete),counts=dict(collections.Counter(v['state'] for v in tasks.values())),
            active=[name for name,row in tasks.items() if row['state']=='active'],controller_reported_active=list(state.get('active',{})),requests=sum(v['requests'] for v in tasks.values()),responses=sum(v['responses'] for v in tasks.values()),generated_tokens=total_tokens,
            worker_ready=read(run/'worker_ready.json'),worker_failure=read(run/'worker_failure.json'),memory_cap_failure=read(run/'memory_cap_failure.json'),
            owner_failure=read(o/'controller_failure.json'),process_receipt=read(run/'process_receipt.json'),worker_complete=read(run/'worker_complete.json'),
            server_parent_receipt=read(h/'server_job_receipt.json'),integrity_errors=integrity_errors,ownership_errors=ownership_errors,containers=containers[run_id],tasks=tasks,
            node_policy_amendment=read(h/'node_policy_amendment.json'),
            node_runtime_preflight=read(h/'node_runtime_preflight.json'),server_preflight=read(h/'server_preflight.json'),
            server_preflight_current=bool((read(h/'server_preflight.json') or {}).get('epoch',0)>=submission['epoch']))
    for arm in ['dense','k12','k48']:
        selected={rid:r for rid,r in runs.items() if r['arm']==arm};tasks={}
        for rid,r in selected.items():
            assert set(tasks).isdisjoint(r['tasks']), 'Duplicate canonical task ownership'
            tasks.update({name:dict(row,owner_run=rid,job_id=r['job_id']) for name,row in r['tasks'].items()})
        assert set(tasks)==set(selection['arms'][arm]['tasks']), 'Missing canonical task ownership'
        sums={k:sum(r[k] for r in selected.values()) for k in ['new_verified','new_normal_completed','new_context_failures','new_passed','new_failed','requests','responses','generated_tokens']}
        retained=selection['arms'][arm]
        failures={key:({rid:r[key] for rid,r in selected.items() if r[key]} or None) for key in ['worker_failure','owner_failure','memory_cap_failure']}
        ids=[r['job_id'] for r in selected.values() if r['job_id']]
        arms[arm]=dict(**sums,**failures,job_id=','.join(ids),job_ids=ids,run_ids=list(selected),planned=len(tasks),retained=retained['retained_count'],
            cumulative_verified=retained['retained_count']+sums['new_verified'],cumulative_passed=retained['retained_pass']+sums['new_passed'],cumulative_failed=retained['retained_fail']+sums['new_failed'],
            counts=dict(collections.Counter(row['state'] for row in tasks.values())),active=[name for name,row in tasks.items() if row['state']=='active'],tasks=tasks,
            integrity_errors=[dict(run=rid,error=e) for rid,r in selected.items() for e in r['integrity_errors']],
            ownership_errors=[dict(run=rid,task=e) for rid,r in selected.items() for e in r['ownership_errors']],
            containers=[c for r in selected.values() for c in r['containers']])
    print(json.dumps(dict(epoch=time.time(),queue=queue,accounting=accounting,arms=arms,runs=runs,
        scale4_registry_sha256=sha(U/'scale4_20260921/registry.json') if registry else None,
        dispatcher_status=read(U/'scale4_20260921/dispatcher_status.json'),dispatcher_failure=read(U/'scale4_20260921/dispatcher_failure.json'),
        scope='Canonical unique tasks across original runs and scale-out shards; retained history counted once, original evidence caveats preserved'),ensure_ascii=False))
if __name__=='__main__':main()
