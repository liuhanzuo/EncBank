"""Build one complete, disjoint 89-task partition per arm from verified evidence."""
import hashlib,json,time,tomllib,subprocess
from pathlib import Path

U=Path(__file__).resolve().parent;S=U.parent;B=S.parent
F=Path('/srv/encbank/Encbank_Migration_20260920/final_handoff_20260921')
R=Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proof(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def normalize(p):return p.replace('\\','/').replace('//','/').split('qencbank/')[-1]
def inspect_result(p):
    r=read(p);roll=(r.get('agent_result') or {}).get('rollout_details') or []
    calls=[]
    for d in roll:
        extra=d.get('extra') or {};n=len(extra.get('generated_tokens',[]))
        for i in range(n):
            e={k:v[i] for k,v in extra.items() if isinstance(v,list) and len(v)==n}
            calls.append({k:e.get(k) for k in ['request_id','step','status','generated_tokens','finish_reason','hit_generation_cap','state_hashes_unchanged']})
    exception=r.get('exception_info') or {}
    caps=[c for c in calls if c['hit_generation_cap'] or c['finish_reason']=='length' or c['generated_tokens']==32768]
    return dict(result=proof(p),calls=len(calls),truncated_calls=caps,
        statuses=sorted({str(c['status']) for c in calls}),exception_type=exception.get('exception_type'),
        exception_message=exception.get('exception_message'),reward=(r.get('verifier_result') or {}).get('rewards',{}).get('reward'),
        normal=bool(calls) and not exception and r.get('verifier_result') is not None,
        agent_execution=r.get('agent_execution'),all_H_checks=all(c['state_hashes_unchanged'] is True for c in calls) if calls and calls[0]['state_hashes_unchanged'] is not None else None)
def main():
    for name in ['handoff_audit.json','dense_handoff_audit.json']:assert read(U/name)['status']=='PASS'
    queue=subprocess.check_output(['squeue','-h','-j','112400,112403,114684,115237','-o','%i|%T'],text=True)
    assert not queue.strip(),queue
    assert read(F/'FINAL_READY.json')['local_controllers_stopped']
    complete=read(F/'closed_verification.json')['arms']
    tasks=sorted(p.name for p in (R/'tasks').iterdir() if (p/'task.toml').exists());assert len(tasks)==89
    report=dict(status='PASS',epoch=time.time(),authorization='2026-09-21 user: remove token/time ceilings; replace old controllers; retest truncated and finish remaining tasks',
        policy='Retain normal untruncated outcomes, positive or negative. New attempts preserve prior evidence and never replace historical rewards.',arms={})
    for arm,key in [('dense','dense'),('k12','top12'),('k48','top48')]:
        retained={};replay={};evidence={}
        for old in complete[key]['retained_normal']:
            name=old['task'];p=U/('verified_dense_handoff' if arm=='dense' else 'verified_handoff')/'repository'/normalize(old['path'])
            assert sha(p)==old['sha256'],str(p)
            row=inspect_result(p);assert row['normal'] and row['statuses']==['ok'],(name,row)
            evidence[name]=row
            if row['truncated_calls']:replay[name]='GENERATION_TOKEN_CAP'
            else:retained[name]=dict(row,source='old_final_handoff')
        if arm in ['dense','k12']:
            h=U/'hybrid_evidence'
            for name in ['regex-chess','vulnerable-secret']:
                results=list((h/'results'/arm/name).glob('*/result.json'));assert len(results)==1
                row=inspect_result(results[0]);evidence[name]=row
                if row['truncated_calls']:replay[name]='GENERATION_TOKEN_CAP'
                elif row['normal']:
                    receipt=h/'runs'/arm/'execution/receipts'/(name+'.json');assert read(receipt)['actual_parent_wait']
                    retained[name]=dict(row,source='local_Docker_supplement',parent_receipt=proof(receipt))
                else:replay[name]='USER_PROTOCOL_REPLACEMENT'
        if arm=='dense':
            h=S/'dense_parallel8_recovery_r2_20260921';assert read(h/'server_job_receipt.json')['actual_parent_waits']
            for p in (h/'execution/task_outcomes').glob('*.json'):
                outcome=read(p);name=outcome['task'];assert outcome['actual_parent_wait']
                row=inspect_result(Path(outcome['results'][0]['path']));evidence[name]=row
                box=Path(read(h/'plan.json')['rpc_root'])/'dense'
                reqs=[f for f in box.glob('*.request.json') if read(f)['task']==name]
                assert len(reqs)==outcome['model_requests']
                for f in reqs:
                    reply=f.with_name(f.name.replace('.request.','.response.'));broker=f.with_name(f.name.replace('.request.','.broker.'))
                    assert reply.exists() and broker.exists(),f
                    pr=read(broker)['proof'];assert sha(reply)==pr['sha256'] and reply.stat().st_size==pr['bytes']
                if row['truncated_calls']:replay[name]='GENERATION_TOKEN_CAP'
                elif outcome['valid_result'] and row['normal']:retained[name]=dict(row,source='Apptainer_recovery',parent_receipt=proof(p))
                else:replay[name]='USER_PROTOCOL_REPLACEMENT'
            for name in complete[key]['old_pending_adjudication']:
                base=F.parents[0]/'Part2/payload/qencbank/.runtime/terminal_bench_full89_20260919/results'
                matches=[]
                for root in base.glob('dense*'):
                    paths=list((root/name).glob('*/result.json'))+list(root.glob(name+'__*/result.json'))
                    for p in paths:
                        row=inspect_result(p)
                        if row['exception_type']=='AgentTimeoutError':matches.append(row)
                assert matches,('Missing actual timeout evidence',name)
                replay[name]='ACTUAL_AGENT_TIME_TRUNCATION';evidence[name]=matches
            for old in complete[key]['excluded']:
                assert old['classification']=='CONTEXT_CAPACITY_BOUNDARY'
                replay[old['task']]='PRIOR_NATIVE_CONTEXT_BOUNDARY_NEW_UNBOUNDED_GENERATION_PROTOCOL'
            replay['mteb-leaderboard']='ENVIRONMENT_NETWORK_CONTRACT_FAILURE_REPAIRED'
        else:
            old_rows={r['task']:r for r in read(F/'TASK_STATUS.json')['rows'] if r['arm']==key}
            for name,r in old_rows.items():
                if name not in retained and name not in replay:
                    replay[name]=r['status']
            if arm=='k12':
                for name in complete[key]['old_pending_adjudication']:
                    base=U/'verified_handoff/repository/.runtime/terminal_bench_full89_20260919/results'
                    matches=[]
                    for root in base.glob('encbank*'):
                        if not any(x in root.name for x in ['top12','refill_retest']):continue
                        for p in (root/name).glob('*/result.json'):
                            row=inspect_result(p)
                            if row['exception_type']=='AgentTimeoutError':matches.append(row)
                    assert matches,name
                    replay[name]='ACTUAL_AGENT_TIME_TRUNCATION';evidence[name]=matches
        for name in tasks:
            if name not in retained and name not in replay:replay[name]='NEVER_STARTED'
        assert not set(retained)&set(replay),(arm,set(retained)&set(replay))
        assert set(retained)|set(replay)==set(tasks)
        order=sorted(replay,key=lambda n:(tomllib.loads((R/'tasks'/n/'task.toml').read_text())['environment'].get('memory_mb',2048),n))
        report['arms'][arm]=dict(tasks=order,retained=retained,replay_reasons=replay,evidence=evidence,
            scheduled=len(order),retained_count=len(retained),retained_pass=sum(r['reward']==1 for r in retained.values()),retained_fail=sum(r['reward']==0 for r in retained.values()))
    p=U/'selection.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ['scheduled','retained_count','retained_pass','retained_fail']} for k,v in report['arms'].items()}))
if __name__=='__main__':main()
