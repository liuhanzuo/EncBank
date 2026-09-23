"""Judge only newly completed LoCoMo generations; reuse exact authorized decisions."""
import datetime, fcntl, gzip, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/encbank/encbank_new_backbones_formal_20260915')
sys.path.append(str(OLD))
from astra_judge_batch import identity, run_batch, write_json
from codex_judge_linux import call
from locomo_judge_reference import PROMPT

OUT=ROOT/'judge_gpt6_astra'

def retryable(result):
    messages=' '.join(e.get('message',e.get('error',{}).get('message','')) for e in result.get('errors',[])).lower()
    if any(w in messages for w in ('401','403','forbidden','unauthorized','quota')):return False
    return (result.get('returncode') is None and not result.get('turn_completed')) or any(w in messages for w in (
        'overloaded','502','503','504','timed out','stream closed before response.completed','stream disconnected before completion'))

def classify(stimulus,out):
    # The previous client timed out without a completed answer. Count ALL saved
    # attempts for this stimulus, including those before the inspected restart.
    entry=Path(out).parent
    previous=[]
    for f in sorted(entry.glob('attempt-*/**/result.json')):
        previous.append(json.loads(f.read_text()))
    # glob('**') includes the attempt directory itself on supported Python.
    for r in previous:
        if r['ok']:return r
    for r in previous:
        if not retryable(r):return r
    if len(previous)>=3:
        # Only a recorded successful service probe may reopen specifically
        # inspected, exhausted transient failures. One extra request each.
        recovery_path=OUT/'transport_recovery.json'
        recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
        grant=recovery.get('extra_attempts_by_stimulus',{}).get(entry.name)
        if not grant:return previous[-1]
        assert grant['maximum_additional_attempts']==1 and grant['previous_attempts']==len(previous)
        target=entry/('service-recovery-'+recovery['id'])
        if (target/'result.json').exists():return json.loads((target/'result.json').read_text())
        assert not (target/'dispatch.json').exists(),'Uncertain recovery request; inspect before another call'
        write_json(target/'dispatch.json',dict(at=time.time(),recovery_id=recovery['id'],maximum_attempts=1))
        return call(PROMPT.format(**stimulus),target,timeout=180)
    for attempt in range(len(previous),3):
        target=Path(out) if attempt==len(previous) else Path(out)/f'transport-retry-{attempt}'
        result=call(PROMPT.format(**stimulus),target,timeout=180)
        if result['ok'] or not retryable(result) or attempt==2:return result
        write_json(Path(out)/'transport_retries.json',dict(total_attempts=attempt+1,maximum_total_attempts=3,
             timeout_seconds=180,reason='No completed judgment; bounded transient transport retry'))
        time.sleep(10 if attempt==0 else 30)
    raise AssertionError('Unreachable')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'watch.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('MIDCACHE_JUDGE_API_KEY')
    protocol=json.loads((OLD/'judge_protocol.json').read_text())
    assert protocol['protocol_id']=='midcache-locomo-gpt6-astra-v1' and protocol['model']=='gpt-6-astra'
    oldstatus=json.loads((OLD/'judge_gpt6_astra/formal_new_models/status.json').read_text())
    assert oldstatus['available_complete'] and not oldstatus['errors']
    # Preserve calibration/transport identity; no repeat control calls or key on disk.
    protocol.update(status='READY',calibration_passed=True)
    write_json(OUT/'protocol.json',protocol)
    questions={};last_set=set()
    plan=json.loads((ROOT/'plan.json').read_text())
    while not (OUT/'STOP').exists():
        status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        finished=[]
        for task in plan['tasks']:
            if not task['id'].startswith(('quant-full-','large-final-')):continue
            if status.get('tasks',{}).get(task['id'],{}).get('state')=='COMPLETED':finished.append(task)
        ids={t['id'] for t in finished}
        if ids!=last_set:
            inputs=[];rows=[]
            for task in finished:
                cfg=task['model'];shard=int(task['id'].rsplit('s',1)[1])
                key=(cfg,shard)
                if key not in questions:
                    with gzip.open(OLD/'samples'/cfg/f'shard{shard}.jsonl.gz','rt',encoding='utf-8') as f:
                        questions[key]={r['id']:r for r in map(json.loads,f) if r['benchmark']=='locomo'}
                records=[]
                pred=ROOT/task['complete'];pred=pred.parent/'predictions.jsonl'
                for line in pred.open(encoding='utf-8'):
                    r=json.loads(line)
                    if r['benchmark']!='locomo':continue
                    q=questions[key][r['id']]
                    records.append(dict(cohort=cfg,arm=r['arm'],id=r['id'],shard=shard,
                        category=q['extra']['category'],status=r['status'],question=q['question'],answers=q['answers'],pred=r['text']))
                assert records
                target=OUT/'inputs'/(task['id']+'.jsonl');target.parent.mkdir(exist_ok=True)
                content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
                if target.exists():assert target.read_text(encoding='utf-8')==content
                else:target.write_text(content,encoding='utf-8')
                inputs.append(target);rows.extend(records)
            reused=0
            for r in rows:
                digest=identity(r,protocol)
                dst=OUT/'cache'/digest/'decision.json'
                src=OLD/'judge_gpt6_astra/formal_new_models/cache'/digest/'decision.json'
                if dst.exists() or not src.exists():continue
                vote=json.loads(src.read_text())
                assert vote['stimulus_digest']==digest and vote['protocol_id']==protocol['protocol_id'] and vote['model']==protocol['model']
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
                write_json(dst.parent/'reuse_origin.json',dict(source=str(src),exact_stimulus_match=True))
                reused+=1
            if inputs:
                write_json(OUT/'watch_status.json',dict(phase='JUDGING',available=len(rows),reused_original_cache=reused,pid=os.getpid(),at=time.time()))
                report=run_batch(inputs,OUT,protocol,workers=2,classifier=classify)
                assert not report['errors'],'Judge transport failed; retain all valid decisions and inspect before restart'
                assert report['available_complete']
                last_set=ids
        complete=len(last_set)==16
        write_json(OUT/'watch_status.json',dict(phase='COMPLETE' if complete else 'WAITING_GENERATIONS',
             verified_generation_tasks=len(last_set),target_generation_tasks=16,pid=os.getpid(),at=time.time()))
        if complete:
            subprocess.run([sys.executable,'-B',str(ROOT/'summarize.py')],cwd=ROOT,check=True)
            break
        if status.get('phase')=='NEEDS_ATTENTION':break
        time.sleep(60)

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'watch_failure.json',dict(error=traceback.format_exc(),at=time.time()))
        raise
