"""Prepare six authorized original qa1/qa5@16k canonical jobs; no launch."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent
name='babi_q15_16k_priority';oldname='babi_q15_32k_priority'
main=json.loads((B/'full_plan.json').read_text())
jobs=[j for j in main['jobs'] if j['benchmark']=='babilong' and j['arm'] in ('pub','pub_sink','cbos') and j['cells'][0]['task'] in ('qa1','qa5') and j['cells'][0]['length']=='16k']
assert len(jobs)==6 and sum(j['expected_n'] for j in jobs)==600
plan={**json.loads((B/(oldname+'_full_plan.json')).read_text()),'jobs':jobs,'state_dir':f'/data/liuhanzuo/comem_v2_20260908/outputs/queue_{name}/full',
      'notes':['Six exact original canonical BABILong qa1/qa5@16k basic-method jobs only; no changed inputs/config/outputs.',
               'Start only after qa1/qa5@32k six native receipts/600 and all historical predecessor PIDs naturally exit; GPU1 empty.',
               'Per-job exact native receipts, main pending/frontier>=40, canonical pre-model-load reuse. Stop after six; no replacement or next batch.']}
(B/(name+'_full_plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
s=(B/(oldname+'_bootstrap.py')).read_text().replace(oldname,name).replace('babi-q15-32k-canonical','babi-q15-16k-canonical').replace('BABI qa1/qa5 32k','BABI qa1/qa5 16k').replace("cell['length']!='32k'","cell['length']!='16k'")
addition='''    previous32=read(R/'outputs/bootstrap_babi_q15_32k_priority/status.json')
    if previous32.get('status')!='completed' or previous32.get('completed_jobs')!=6 or previous32.get('completed_predictions')!=600 or len(previous32.get('jobs',{}))!=6:raise RuntimeError('qa1/qa5@32k predecessor incomplete')
    pids += [previous32.get('pid'),previous32.get('queue_pid')]
    for item in previous32['jobs'].values():
        if item.get('status')!='complete' or item.get('exit_code')!=0:raise RuntimeError('qa1/qa5@32k predecessor job incomplete')
        pids.append(item.get('model_pid'));q32=read(Path(item['queue_state']));pids.append(q32.get('queue_pid'))
        pids.extend(x.get('pid') for x in q32['jobs'].values() if not x.get('external_dependency'))
    p32=read(B/'babi_q15_32k_priority_full_plan.json')
    if len(p32['jobs'])!=6 or not all(completed(j) for j in p32['jobs']):raise RuntimeError('qa1/qa5@32k native receipts invalid')
'''
needle="    bootstrap=R/'outputs/bootstrap_non_qwen_formal';formal=R/'outputs/non_qwen_smol_20260909/formal'"
assert needle in s;s=s.replace(needle,addition+needle)
s=s.replace("return {'babi8_native_receipts':6", "return {'q15_32k_native_receipts':6,'q15_32k_predictions':600,'babi8_native_receipts':6")
assert '_priority_priority' not in s and f"B/'{name}_full_plan.json'" in s
(B/(name+'_bootstrap.py')).write_text(s)
pre=(B/(oldname+'_preflight_0731.py')).read_text().replace(oldname,name).replace('20260909_0731','20260909_0831')
(B/(name+'_preflight_0831.py')).write_text(pre)
health=(B/('health_'+oldname+'_0731_remote.py')).read_text().replace(oldname,name).replace('20260909_0731','20260909_0831')
(B/('health_'+name+'_0831_remote.py')).write_text(health)
print(json.dumps({'jobs':[j['id'] for j in jobs],'exact_canonical_jobs':True,'predictions':600}))
