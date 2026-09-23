import datetime,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;T=H.parent/'hot_buffer_terminal_20260921'
def load(p):return json.loads(p.read_text()) if p.exists() else None
jobs=load(T/'submissions.json') or [];rows=[]
queue=subprocess.run(['squeue','-h','-j',','.join(j['job'] for j in jobs),'-o','%i|%T|%N|%R'],capture_output=True,text=True).stdout
for j in jobs:
    R=Path(j['root']);task=j['task'];arm=j['arm'];pair=R/'pairs'/task
    for name,expected in load(R/'source_manifest.json').items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected
    receipt=load(R/'results'/(task+'--'+arm)/'execution_receipt.json');wait=load(R/'jobs'/('wait-'+task+'--'+arm+'.json'))
    rs=[];qs=sorted((R/'mailbox'/task).glob(arm+'*.request.json'))
    for p in qs:
        r=load(p.with_name(p.name.replace('.request.','.response.')))
        if r:assert r['request_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();rs.append(r)
    valid=bool(receipt and wait and wait['exit_code']==0 and wait['actual_parent_wait'] and not receipt['exception']
        and receipt['closure']['cgroup_empty'] and receipt['verifier'] is not None and rs and len(rs)==len(qs) and all(r['status']=='ok' for r in rs))
    sums={k:sum(r.get(k,0) for r in rs) for k in ['generated_tokens','request_seconds','prefill_seconds','decode_seconds','promotion_seconds','hit_chunks','requested_chunks','rebuilt_chunks','promoted_hits']}
    row=dict(task=task,arm=arm,job=j['job'],status=load(pair/'status.json'),requests=len(qs),responses=len(rs),valid=valid,
        rewards=receipt['verifier'] if valid else None,wall_seconds=receipt['elapsed_seconds'] if receipt else None,
        complete=load(R/'jobs/integrated_complete.json'),**sums)
    for name in ['worker_failure.json','integrated_failure.json']:
        failure=load(pair/name)
        if failure:row[name]=failure.get('error','')[-1000:]
    rows.append(row)
snapshot=dict(at=datetime.datetime.now().astimezone().isoformat(),queue=queue,valid_trials=sum(r['valid'] for r in rows),expected_trials=len(jobs),rows=rows)
(H/'terminal_snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
print(queue.strip())
for r in rows:print(json.dumps(r))
