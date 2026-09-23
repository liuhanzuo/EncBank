"""Read-only experiment observer. Writes only separate monitoring artifacts."""
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path('/srv/encbank/client/encbank_local_20260921')
OUTPUT = pathlib.Path('/srv/encbank/workspace/local_docker_20260921/monitoring')
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
REMOTE = '/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921'
PYTHON = '/srv/encbank/qencbank_runtime_20260911/python312/bin/python'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() if value else None


def span(value):
    if not value or not value.get('started_at'):
        return None
    return (timestamp(value.get('finished_at')) or time.time()) - timestamp(value['started_at'])


def process_alive(pid):
    return bool(pid) and pathlib.Path('/proc', str(pid)).exists()


def atomic(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def inspect_task(family, task, now):
    run = ROOT / 'runs' / family
    execution = run / 'execution'
    launch = read(execution / 'launches' / (task + '.json'))
    outcome = read(execution / 'task_outcomes' / (task + '.json'))
    receipt = read(execution / 'receipts' / (task + '.json'))
    row = dict(family=family, task=task, state='pending', elapsed_seconds=None,
               model_requests=0, replies=0, generated_tokens=0, transport_seconds=0,
               latest_request_epoch=None, latest_reply_epoch=None, issues=[])
    if launch:
        row.update(state='running', started_epoch=launch['epoch'],
                   elapsed_seconds=now-launch['epoch'], parent_alive=process_alive(launch.get('pid')))
    box = ROOT / 'rpc' / family / ('dense' if family == 'dense' else 'encbank')
    steps = []
    for request in box.glob('*.request.json'):
        q = read(request)
        if q['task'] != task:
            continue
        row['model_requests'] += 1
        row['latest_request_epoch'] = max(row['latest_request_epoch'] or 0, q['client_created_epoch'])
        rid = q['request_id']
        response = box / (rid + '.response.json')
        error = read(box / (rid + '.error.json'))
        if error:
            row['issues'].append({'request_id': rid, 'transport_error': str(error.get('error'))[:1000]})
        if not response.exists():
            continue
        raw = response.read_bytes()
        r = json.loads(raw)
        row['replies'] += 1
        row['latest_reply_epoch'] = max(row['latest_reply_epoch'] or 0, response.stat().st_mtime)
        row['generated_tokens'] += r.get('generated_tokens', 0)
        broker = read(box / (rid + '.broker.json'))
        if not broker:
            row['issues'].append({'request_id': rid, 'issue': 'reply exists, broker proof not yet written'})
            continue
        proof = broker['proof']
        assert hashlib.sha256(raw).hexdigest() == proof['sha256'], rid
        assert len(raw) == proof['bytes'], rid
        assert all(q[k] == r[k] for k in ['request_id', 'task', 'task_id', 'step']), rid
        assert r.get('generated_tokens', 0) == len(r.get('generated_ids', [])), rid
        row['transport_seconds'] += broker['seconds']
        steps.append(q['step'])
        if r['status'] != 'ok':
            row['issues'].append({'request_id': rid, 'response_status': r['status']})
    row['waiting_requests'] = row['model_requests'] - row['replies']
    row['seconds_since_reply'] = now-row['latest_reply_epoch'] if row['latest_reply_epoch'] else None
    results = list((ROOT / 'results' / family / task).glob('*/result.json'))
    if len(results) > 1:
        row['issues'].append({'issue': 'multiple results require reconciliation'})
    if results:
        raw = results[0].read_bytes()
        result = json.loads(raw)
        row.update(result_sha256=hashlib.sha256(raw).hexdigest(),
                   started_at=result.get('started_at'), finished_at=result.get('finished_at'),
                   elapsed_seconds=span(result),
                   phase_seconds={k: span(result.get(k)) for k in
                                  ['environment_setup', 'agent_setup', 'agent_execution', 'verifier']},
                   rewards=(result.get('verifier_result') or {}).get('rewards'),
                   exception_type=(result.get('exception_info') or {}).get('exception_type'))
    if outcome:
        valid = outcome['valid_result'] and bool(receipt) and receipt['actual_parent_wait']
        valid = valid and receipt['exit_code'] == 0 and not receipt['transport_errors']
        valid = valid and row['model_requests'] > 0 and row['model_requests'] == row['replies']
        valid = valid and sorted(steps) == list(range(row['model_requests'])) and not row['issues']
        row['state'] = 'completed' if valid else 'closed_incomplete'
        row['parent_wait'] = receipt
    return row


def remote_snapshot():
    script = '''import json,pathlib,subprocess,sys,time,datetime
p=pathlib.Path(REMOTE_ROOT)
def read(f): return json.loads(f.read_text()) if f.exists() else None
out={'batch':read(p/'status.json'),'slurm':subprocess.check_output(['sacct','-X','-j','112400,112403,114684','--format=JobID,State,Elapsed,Timelimit,Start,End','-P'],text=True),'arms':{}}
for family in ['dense','k12','k48']:
 r=p/family/('run_dense' if family=='dense' else 'run_encbank')
 row={n:read(r/n) for n in ['process_start.json','worker_ready.json','worker_failure.json','memory_cap_failure.json','bootstrap_timeout.json','process_receipt.json','worker_complete.json'] if (r/n).exists()}
 for n in ['owned_nvml.jsonl','events.jsonl']:
  f=r/n
  if f.exists():
   with f.open('rb') as stream:
    stream.seek(0,2);length=stream.tell();stream.seek(max(0,length-65536));lines=stream.read().splitlines()
   try: row[n]=[json.loads(x) for x in lines[-3:]]
   except ValueError: row[n]=[]
 out['arms'][family]=row
for directory,family,series_key in [('dense_parallel4_20260921','dense-p4','dense_parallel4'),('dense_parallel8_recovery_r2_20260921','dense-p8-recovery','dense_parallel8_recovery')]:
 n=p.parent/directory
 if (n/'plan.json').exists():
  plan=read(n/'plan.json');sub=read(n/'submission.json');exe=n/'execution'
  extra={'submission':sub,'qualification':read(n/'qualification_submission.json'),'controller':read(exe/'status.json'),'failure':read(exe/'controller_failure.json'),'tasks':[]}
  worker=n/'run_dense'
  extra['worker']={k:read(worker/k) for k in ['process_start.json','worker_ready.json','worker_failure.json','memory_cap_failure.json','bootstrap_timeout.json','process_receipt.json','worker_complete.json'] if (worker/k).exists()}
  for key in ['owned_nvml.jsonl','events.jsonl']:
   f=worker/key
   if f.exists():
    with f.open('rb') as stream:
     stream.seek(0,2);length=stream.tell();stream.seek(max(0,length-65536));lines=stream.read().splitlines()
    try:extra['worker'][key]=[json.loads(x) for x in lines[-3:]]
    except ValueError:extra['worker'][key]=[]
  for task in plan['tasks']:
   launch=read(exe/'launches'/(task+'.json'));outcome=read(exe/'task_outcomes'/(task+'.json'))
   item={'family':family,'task':task,'state':'pending','elapsed_seconds':None,'model_requests':0,'replies':0,'series':'server_apptainer','phase_seconds':{}}
   if launch:item.update(state='running',elapsed_seconds=time.time()-launch['epoch'])
   for f in pathlib.Path(plan['rpc_root']).glob('dense/*.request.json'):
    req=read(f)
    if req['task']==task:
     item['model_requests']+=1;item['replies']+=f.with_name(req['request_id']+'.response.json').exists()
   if outcome:item.update(state='completed_pending_audit' if outcome['valid_result'] else 'closed_incomplete',outcome=outcome)
   results=list((pathlib.Path(plan['results_root'])/task).glob('*/result.json'))
   if results:
    result=read(results[0])
    def span(v):
     if not v or not v.get('started_at'):return None
     a=datetime.datetime.fromisoformat(v['started_at'].replace('Z','+00:00')).timestamp()
     b=datetime.datetime.fromisoformat(v['finished_at'].replace('Z','+00:00')).timestamp() if v.get('finished_at') else time.time()
     return b-a
    item.update(elapsed_seconds=span(result),phase_seconds={k:span(result.get(k)) for k in ['environment_setup','agent_setup','agent_execution','verifier']},rewards=(result.get('verifier_result') or {}).get('rewards'))
   extra['tasks'].append(item)
  if sub and sub.get('job_id'):
   extra['slurm']=subprocess.check_output(['sacct','-X','-j',sub['job_id'],'--format=JobID,State,Elapsed,Timelimit,Start,End','-P'],text=True)
  out[series_key]=extra
print(json.dumps(out))
'''.replace('REMOTE_ROOT', repr(REMOTE))
    p = subprocess.run([SSH, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'gpu-node1', PYTHON, '-'], input=script, capture_output=True, text=True, timeout=45)
    if p.returncode:
        return {'read_error': p.stderr[-2000:]}
    return json.loads(p.stdout)


def main():
    now = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = read(OUTPUT / 'latest.json')
    report = dict(observed_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                  epoch=now, local_batch=read(ROOT/'status.json'),
                  controller_start=read(ROOT/'owner_start.json'), controllers={}, tasks=[])
    for family in ['dense', 'k12', 'k48']:
        e = ROOT / 'runs' / family / 'execution'
        report['controllers'][family] = {n: read(e/n) for n in
            ['status.json', 'controller_failure.json', 'owner_complete.json'] if (e/n).exists()}
        for task in ['regex-chess', 'vulnerable-secret']:
            report['tasks'].append(inspect_task(family, task, now))
    report['remote'] = remote_snapshot()
    report['tasks'].extend(report['remote'].get('dense_parallel4', {}).get('tasks', []))
    report['tasks'].extend(report['remote'].get('dense_parallel8_recovery', {}).get('tasks', []))
    docker = subprocess.run(['docker', 'ps', '--format', '{{.Names}}|{{.Status}}'],
                            capture_output=True, text=True, timeout=20)
    report['docker'] = {'exit_code': docker.returncode, 'containers': docker.stdout.splitlines()}
    report['completed'] = sum(r['state']=='completed' for r in report['tasks'])
    report['closed_incomplete'] = sum(r['state']=='closed_incomplete' for r in report['tasks'])
    report['completed_pending_audit'] = sum(r['state']=='completed_pending_audit' for r in report['tasks'])
    report['changes'] = []
    if previous:
        old = {(r['family'], r['task']): r for r in previous['tasks']}
        for row in report['tasks']:
            before = old.get((row['family'], row['task']), {})
            if row['state'] != before.get('state'):
                report['changes'].append({'family': row['family'], 'task': row['task'],
                                           'before': before.get('state'), 'after': row['state']})
            row['new_replies_since_previous_observation'] = row['replies']-before.get('replies', 0)
    text = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    atomic(OUTPUT/'latest.json', text)
    atomic(OUTPUT/(dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'), text)
    lines = ['# Encbank 补测耗时与进度', '', '观测时间：'+report['observed_at'], '',
             '| 方法 | 题目 | 状态 | 总耗时/分钟 | Agent耗时/分钟 | 模型回复/请求 |',
             '|---|---|---|---:|---:|---:|']
    for row in report['tasks']:
        total = '-' if row['elapsed_seconds'] is None else f"{row['elapsed_seconds']/60:.2f}"
        agent = row.get('phase_seconds', {}).get('agent_execution')
        agent = '-' if agent is None else f'{agent/60:.2f}'
        lines.append(f"| {row['family']} | {row['task']} | {row['state']} | {total} | {agent} | {row['replies']}/{row['model_requests']} |")
    lines += ['', '耗时仅用于统计，不据此终止任务。运行中耗时为截至本次观测的时长。',
              '总耗时含环境、agent和评分；模型传输等待另存JSON，不能与总耗时相加。',
              '旧实验累计数需另查最新交接，不能以这里的六次补测替代。', '',
              '```text', report['remote'].get('slurm', str(report['remote'])), '```', '']
    atomic(OUTPUT/'progress.md', '\n'.join(lines))
    print(json.dumps({k: report[k] for k in ['observed_at','completed','closed_incomplete','changes']}, ensure_ascii=False))
    print('\n'.join(lines[4:12]))


if __name__ == '__main__':
    main()
