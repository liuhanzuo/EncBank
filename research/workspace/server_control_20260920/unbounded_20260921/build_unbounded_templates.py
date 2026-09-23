"""Build unused server templates; never mutate submitted predecessors."""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent
OUT=HERE/'templates'

def change(p, old, new):
    s=p.read_text()
    assert old in s,(p.name,old[:100])
    p.write_text(s.replace(old,new))

POLICY='''"""No policy token/time ceiling; native position and memory bounds stay explicit."""
def output_allowance(context_tokens, actual_input_tokens, max_new_tokens=None):
    assert max_new_tokens is None, 'This protocol forbids an arbitrary output token cap'
    return max(0, context_tokens-actual_input_tokens)

def deadline_expired(request):
    assert request.get('deadline_epoch') is None, 'No elapsed-time cancellation in this protocol'
    return False

def native_memory_admissible(allocated, current_cache, projected_h_growth, rows, plan):
    worst_cache=rows*plan['context_tokens']*plan['kv_bytes_per_token']
    projected=allocated-current_cache+projected_h_growth+worst_cache+plan['decode_workspace_gib']*2**30
    return projected <= plan['native_allocator_cap_gib']*2**30, projected
'''

HARBOR_POLICY='''"""Run-scoped Harbor deadline policy, imported before this run constructs Trials."""
from harbor.trial.trial import Trial

def no_deadline(self, *args, **kwargs):
    return None

for method in ['_compute_agent_timeout_sec','_compute_verifier_timeout_sec',
               '_compute_agent_setup_timeout_sec','_compute_environment_build_timeout_sec',
               '_step_verifier_timeout_sec']:
    assert hasattr(Trial,method), method
    setattr(Trial,method,no_deadline)
'''

CPU_SLOTS='''"""CPU leases for this single server run; dead leases require explicit review."""
import fcntl,json,os
from pathlib import Path
from contextlib import contextmanager
H=Path(__file__).resolve().parent
@contextmanager
def locked():
    with (H/'container_cpu.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield
def acquire(name,count):
    with locked():
        f=H/'container_cpu_leases.json'
        leases=json.loads(f.read_text()) if f.exists() else {}
        assert name not in leases
        occupied={x for row in leases.values() for x in row['cpus']}
        free=sorted(set(os.sched_getaffinity(0))-occupied)
        assert len(free)>=count,'Insufficient allocated CPU slots for task'
        cpus=free[:count];leases[name]={'cpus':cpus,'owner_pid':os.getpid()}
        temp=f.with_suffix('.tmp');temp.write_text(json.dumps(leases));temp.replace(f)
        return cpus
def release(name):
    with locked():
        f=H/'container_cpu_leases.json'
        leases=json.loads(f.read_text()) if f.exists() else {}
        row=leases.pop(name,None)
        if row is not None: assert row['owner_pid']==os.getpid()
        temp=f.with_suffix('.tmp');temp.write_text(json.dumps(leases));temp.replace(f)
'''

def main():
    OUT.mkdir(exist_ok=True)
    for arm in ['dense','k12','k48']:
        dest=OUT/arm
        assert not dest.exists(), 'Never overwrite a built template; use a revision directory'
        shutil.copytree(HERE/arm,dest)
        # Common controller and corrected backend come from the qualified recovery.
        for n in ['server_owner.py','server_job.py','server_preflight.py','server_transport.py',
                  'apptainer_environment.py','apptainer_service.py','apptainer_executor.py']:
            shutil.copy2(HERE/'dense'/n,dest/n)
        (dest/'unbounded_policy.py').write_text(POLICY)
        (dest/'harbor_unbounded.py').write_text(HARBOR_POLICY)
        (dest/'cpu_slots.py').write_text(CPU_SLOTS)
        change(dest/'tb_agent_rpc.py','from harbor.agents.terminus_2.terminus_2 import Terminus2',
               'import harbor_unbounded\nfrom harbor.agents.terminus_2.terminus_2 import Terminus2')
        rpc=dest/'tb_agent_rpc.py'
        s=rpc.read_text()
        s='\n'.join(l for l in s.splitlines() if "raise TimeoutError('Server controller response watchdog" not in l)+'\n'
        rpc.write_text(s)
        change(dest/'live_mailbox.py',"remaining=d.pop('remaining_seconds');assert 0<=remaining<=12000\n            d.update(published_epoch=time.time());d['deadline_epoch']=d['published_epoch']+remaining",
               "remaining=d.pop('remaining_seconds');assert remaining is None\n            d.update(published_epoch=time.time(),deadline_epoch=None)")
        change(dest/'live_mailbox.py',"                if time.time()>req['deadline_epoch']+90:raise TimeoutError('Worker did not close expired generation')\n",'')
        change(dest/'server_transport.py',"request['remaining_seconds'] + 180",'None')
        change(dest/'server_owner.py',"max_wait_seconds=P['bootstrap_timeout_seconds'] + 300",'max_wait_seconds=None')
        change(dest/'startup_wait.py',"if clock()-began>max_wait_seconds:","if max_wait_seconds is not None and clock()-began>max_wait_seconds:")
        change(dest/'holder.py',"if not (R/'worker_ready.json').exists() and time.monotonic()-boot_started>P['bootstrap_timeout_seconds'] and not boot_failed:",
               "if P['bootstrap_timeout_seconds'] is not None and not (R/'worker_ready.json').exists() and time.monotonic()-boot_started>P['bootstrap_timeout_seconds'] and not boot_failed:")
        worker=dest/'agent_worker.py'
        s=worker.read_text()
        s='\n'.join(l for l in s.splitlines() if not (('if time.monotonic()>' in l) and ('probe_deadline' in l or '>deadline:' in l)))+'\n'
        worker.write_text(s)
        env=dest/'apptainer_environment.py'
        s=env.read_text();start=s.index('        available = sorted(os.sched_getaffinity(0))');end=s.index('        mode = self._network_policy',start)
        s=s[:start]+"        from cpu_slots import acquire\n        affinity = acquire(self._name,max(1,int(self._effective_cpus or 1)))\n"+s[end:]
        s=s.replace('result = await self.exec(bootstrap,timeout_sec=300)','result = await self.exec(bootstrap,timeout_sec=None)')
        s=s.replace("        (self._root/'closure.json').write_text", "        from cpu_slots import release\n        release(self._name)\n        (self._root/'closure.json').write_text")
        env.write_text(s)
        if arm=='dense':
            change(worker,'from capacity_admission import reservation,admissible','from capacity_admission import reservation,admissible\nfrom unbounded_policy import output_allowance,deadline_expired')
            change(worker,"time.time()>=req['deadline_epoch']",'deadline_expired(req)')
            change(worker,"min(P['max_new_tokens'],P['context_tokens']-len(ids))","output_allowance(P['context_tokens'],len(ids),P['max_new_tokens'])")
            change(worker,"if last.finished and reply['status']=='error':reply['status']='ok'","if last.finished and reply['status']=='error':reply['status']='context_limit' if out.finish_reason=='length' else 'ok'\n                reply['hit_context_capacity']=out.finish_reason=='length'\n                reply['configured_generation_token_cap']=None")
            change(worker,"            if not active and time.monotonic()-last_activity>14400:raise TimeoutError('No requests for four hours; owner needs inspection')\n",'')
            change(dest/'capacity_admission.py','return prompt_tokens + min(max_new_tokens, context_tokens-prompt_tokens)',
                   "assert max_new_tokens is None\n    return context_tokens")
        else:
            service=dest/'service_loop.py'
            change(service,'from session import Session,digest,request_seed','from session import Session,digest,request_seed,native_ids\nfrom unbounded_policy import output_allowance,deadline_expired,native_memory_admissible')
            change(service,"if time.time()>=q['deadline_epoch']:return 'deadline'","if deadline_expired(q):return 'deadline'")
            s=service.read_text();start=s.index("        if len(full)>=P['context_tokens']:");end=s.index('        selection_start=',start)
            s=s[:start]+s[end:]
            s=s.replace("        seed=request_seed(P,q['task'],q['step']);", "        if max(len(query),len(pack))>=P['context_tokens']:\n            reply.update(status='context_limit',text='',prompt_tokens=len(pack),generated_tokens=0,actual_input_tokens=max(len(query),len(pack)))\n            dump(B/(rid+'.response.json'),reply);seen.add(path.name);return None\n        seed=request_seed(P,q['task'],q['step']);")
            s=s.replace("limit=min(P['max_new_tokens'],P['context_tokens']-len(full))", "limit=output_allowance(P['context_tokens'],max(len(query),len(pack)),P['max_new_tokens'])")
            s=s.replace("hit_generation_cap=len(g)==row['limit'] and (not g or g[-1] not in stop)","hit_generation_cap=False,configured_generation_token_cap=None,hit_context_capacity=len(g)==row['limit'] and (not g or g[-1] not in stop)")
            s=s.replace("if token in stop or len(row['generated'])>=row['limit']:status='ok'", "if token in stop:status='ok'\n                        elif len(row['generated'])>=row['limit']:status='context_limit'")
            s=s.replace("                if time.monotonic()-last>14400:raise TimeoutError('No owner requests for four hours')\n",'')
            # Admission reserves a complete native-length KV cache for every active generation.
            marker='    def prepare(path):'
            helper='''    def may_admit(path,rows,current_cache):
        q=json.loads(path.read_text())
        if cancelled(q):return True
        full=native_ids(tok,q['messages'])
        old=sessions.get(q['task_id'])
        old_h=sum(t.numel()*t.element_size() for t in old.bank.values()) if old else 0
        growth=max(0,len(full)*reader.config.hidden_size*2-old_h)
        allowed,projected=native_memory_admissible(torch.cuda.memory_allocated(),current_cache,growth,rows,P)
        if not allowed:
            event('memory_admission_wait',request_id=q['request_id'],projected_bytes=projected,active_rows=rows-1)
            if rows==1:raise MemoryError('Native memory capacity cannot fit one full-context generation with retained H; no token truncation or quality zero')
        return allowed
'''
            s=s.replace(marker,helper+marker)
            s=s.replace("            for path in chosen:\n                row=prepare(path)","            for path in chosen:\n                cached=sum(cache_bytes(r['lower'])+cache_bytes(r['upper']) for r in active)\n                if not may_admit(path,len(active)+1,cached):break\n                row=prepare(path)")
            s=s.replace("                    for path in waiting[:P['decode_batch_size']-len(active)]:\n                        row=prepare(path)","                    for path in waiting[:P['decode_batch_size']-len(active)]:\n                        cached=cache_bytes(lower)+cache_bytes(upper)+sum(cache_bytes(r['lower'])+cache_bytes(r['upper']) for r in extra)\n                        if not may_admit(path,len(active)+len(extra)+1,cached):break\n                        row=prepare(path)")
            service.write_text(s)
        print(arm,'template ready')

if __name__=='__main__': main()
