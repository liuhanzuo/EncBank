"""Prepare exact authorized six-job plan and independent bootstrap; no launch."""
from pathlib import Path
import json
B=Path(__file__).resolve().parent
main=json.loads((B/'full_plan.json').read_text())
jobs=[j for j in main['jobs'] if j['benchmark']=='babilong' and j['arm'] in ('pub','pub_sink','cbos') and j['cells'][0]['task'] in ('qa2','qa3') and j['cells'][0]['length']=='8k']
assert len(jobs)==6 and sum(j['expected_n'] for j in jobs)==600
plan={**json.loads((B/'babi16_priority_full_plan.json').read_text()),'jobs':jobs,'state_dir':'/data/liuhanzuo/comem_v2_20260908/outputs/queue_babi8_priority/full',
      'notes':['Six exact canonical native BABILong jobs qa2/qa3@8k only; no protocol/input/output change.',
               'Start only after nonQwen formal1750 complete, all its historical PIDs exited, and idle GPU1 admission.',
               'Per-job unchanged native receipts, main pending/frontier distance>=40, exclusive empty GPU1; stop after six jobs.']}
(B/'babi8_priority_full_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(B/'babi16_priority_bootstrap.py').read_text().replace('babi16','babi8').replace('BABI16','BABI8').replace("cell['length']!='16k'","cell['length']!='8k'")
s=s.replace("'exit; next GPU1 priority reserved for root-coordinated nonQwen smoke if ready'","'exit after these six canonical jobs; no automatic next workload'")
helper='''def verify_predecessor():
    from native_priority_receipt import completed
    prior=read(R/'outputs/bootstrap_babi16_priority/status.json')
    if prior.get('status')!='completed' or prior.get('completed_jobs')!=6 or prior.get('completed_predictions')!=600 or len(prior.get('jobs',{}))!=6:raise RuntimeError('BABI16 predecessor incomplete')
    pids=[prior.get('pid'),prior.get('queue_pid')]
    for item in prior['jobs'].values():
        if item.get('status')!='complete' or item.get('exit_code')!=0:raise RuntimeError('BABI16 job incomplete')
        pids.append(item.get('model_pid'));q=read(Path(item['queue_state']));pids.append(q.get('queue_pid'))
        pids.extend(x.get('pid') for x in q['jobs'].values() if not x.get('external_dependency'))
    p=read(B/'babi16_priority_full_plan.json')
    if len(p['jobs'])!=6 or not all(completed(j) for j in p['jobs']):raise RuntimeError('BABI16 native receipts invalid')
    bootstrap=R/'outputs/bootstrap_non_qwen_formal';formal=R/'outputs/non_qwen_smol_20260909/formal'
    state=read(bootstrap/'status.json');marker=read(formal/'non_qwen_FORMAL_COMPLETE.json')
    if state.get('status')!='completed' or state.get('child_exit_code')!=0 or state.get('completed_rows')!=1750:raise RuntimeError('NonQwen formal bootstrap incomplete')
    expected={'status':'complete','formal_rows':1750,'expected_rows':1750,'verified_smoke_reused':70,'formal_generations':1680,'completed_cells':20,'excluded_native_references':4}
    if any(marker.get(k)!=v for k,v in expected.items()):raise RuntimeError('NonQwen formal receipt incomplete')
    if len(marker.get('cells',[]))!=20 or any(not c.get('complete') or c['n']!=c['expected_n'] for c in marker['cells']):raise RuntimeError('NonQwen formal task cells incomplete')
    if len(list((formal/'records').glob('*.json')))!=1750:raise RuntimeError('NonQwen formal record count incomplete')
    # Historical PIDs below were observed in the actual formal launch receipt.
    pids += [3166138,3166145,state.get('pid'),state.get('child_pid')]
    for path in list((bootstrap/'history').glob('status*.json'))+list((formal/'attempts').glob('*/status.json')):
        obj=read(path);pids += [obj.get('pid'),obj.get('child_pid')]
    live=[p for p in pids if alive(p)]
    if live:raise RuntimeError('Predecessor processes have not naturally exited: '+str(live))
    return {'babi16_native_receipts':6,'babi16_predictions':600,'non_qwen_rows':1750,'non_qwen_cells':20,'smoke_reused':70,'formal_generations':1680,'non_qwen_child_exit_code':0,'historical_pids':sorted(set(p for p in pids if p)),'all_recorded_processes_exited':True}

'''
assert 'def main():' in s
s=s.replace('def main():',helper+'def main():',1)
s=s.replace("state['predecessors_complete_and_exited']=True;save()","state['latest_predecessor']=verify_predecessor();state['predecessors_complete_and_exited']=True;save()")
# Recheck process ownership immediately before each finite one-job dispatch.
s=s.replace("if allowed and idle:break","if allowed and idle:\n                    state['latest_predecessor']=verify_predecessor()\n                    again,reading=gpu_idle(1,512);state['pre_dispatch_gpu_check']=reading;save()\n                    if again:break")
(B/'babi8_priority_bootstrap.py').write_text(s)
print(json.dumps({'jobs':[j['id'] for j in jobs],'exact_canonical_jobs':True,'new_predictions':600}))
