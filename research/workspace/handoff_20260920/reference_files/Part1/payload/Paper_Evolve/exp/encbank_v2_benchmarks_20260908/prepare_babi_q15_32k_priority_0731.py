"""Prepare six authorized original qa1/qa5@32k canonical jobs; no launch."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent
main=json.loads((B/'full_plan.json').read_text())
jobs=[j for j in main['jobs'] if j['benchmark']=='babilong' and j['arm'] in ('pub','pub_sink','cbos') and j['cells'][0]['task'] in ('qa1','qa5') and j['cells'][0]['length']=='32k']
assert len(jobs)==6 and sum(j['expected_n'] for j in jobs)==600
name='babi_q15_32k_priority'
plan={**json.loads((B/'babi8_priority_full_plan.json').read_text()),'jobs':jobs,'state_dir':f'/data/liuhanzuo/encbank_v2_20260908/outputs/queue_{name}/full',
      'notes':['Six exact original canonical BABILong qa1/qa5@32k basic-method jobs only; no changed inputs/config/outputs.',
               'Start only after BABI8 six native receipts/600 and all historical predecessor PIDs naturally exit; GPU1 empty.',
               'Per-job exact native receipts, main pending/frontier>=40, canonical pre-model-load reuse. Stop after six; no replacement or next batch.']}
(B/(name+'_full_plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
s=(B/'babi8_priority_bootstrap.py').read_text().replace('babi8_priority',name).replace('babi8-canonical','babi-q15-32k-canonical').replace('BABI8','BABI qa1/qa5 32k')
s=s.replace("('qa2','qa3')","('qa1','qa5')").replace("cell['length']!='8k'","cell['length']!='32k'")
addition='''    previous8=read(R/'outputs/bootstrap_babi8_priority/status.json')
    if previous8.get('status')!='completed' or previous8.get('completed_jobs')!=6 or previous8.get('completed_predictions')!=600 or len(previous8.get('jobs',{}))!=6:raise RuntimeError('BABI8 predecessor incomplete')
    pids += [previous8.get('pid'),previous8.get('queue_pid')]
    for item in previous8['jobs'].values():
        if item.get('status')!='complete' or item.get('exit_code')!=0:raise RuntimeError('BABI8 predecessor job incomplete')
        pids.append(item.get('model_pid'));q8=read(Path(item['queue_state']));pids.append(q8.get('queue_pid'))
        pids.extend(x.get('pid') for x in q8['jobs'].values() if not x.get('external_dependency'))
    p8=read(B/'babi8_priority_full_plan.json')
    if len(p8['jobs'])!=6 or not all(completed(j) for j in p8['jobs']):raise RuntimeError('BABI8 native receipts invalid')
'''
needle="    bootstrap=R/'outputs/bootstrap_non_qwen_formal';formal=R/'outputs/non_qwen_smol_20260909/formal'"
assert needle in s;s=s.replace(needle,addition+needle)
s=s.replace("return {'babi16_native_receipts':6", "return {'babi8_native_receipts':6,'babi8_predictions':600,'babi16_native_receipts':6")
(B/(name+'_bootstrap.py')).write_text(s)
assert '_priority_priority' not in s and f"B/'{name}_full_plan.json'" in s
pre=(B/'babi8_priority_preflight_0631.py').read_text().replace('babi8_priority',name).replace('20260909_0631','20260909_0731')
pre=pre.replace("    with (out/'bootstrap.log').open('x') as log:", "    attempt=1\n    logpath=out/'bootstrap.log'\n    while logpath.exists():\n        attempt+=1;logpath=out/f'bootstrap_attempt{attempt}.log'\n    with logpath.open('x') as log:")
pre=pre.replace("report.update(launched=True,bootstrap_pid=child.pid", "report.update(launched=True,attempt=attempt,bootstrap_log=str(logpath),bootstrap_pid=child.pid")
pre=pre.replace("(R/'outputs/diagnostics'/name).write_text(json.dumps(report,indent=2)+'\\n');print(json.dumps(report,indent=2))", "dest=R/'outputs/diagnostics'/name\nif args.launch and dest.exists():\n    history=dest.with_name(dest.stem+'_attempt1.json')\n    if not history.exists():history.write_text(dest.read_text())\ndest.write_text(json.dumps(report,indent=2)+'\\n');print(json.dumps(report,indent=2))")
(B/(name+'_preflight_0731.py')).write_text(pre)
health=(B/'health_babi8_priority_0631_remote.py').read_text().replace('babi8_priority',name).replace('20260909_0631','20260909_0731')
(B/('health_'+name+'_0731_remote.py')).write_text(health)
print(json.dumps({'jobs':[j['id'] for j in jobs],'exact_canonical_jobs':True,'predictions':600}))
