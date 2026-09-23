"""Concurrent independent judgments, with exact-input caching across methods."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
from locomo_judge_reference import PROMPT,_REFUSAL_RE

ROOT=Path(__file__).resolve().parent
MODEL='gpt-6-astra'

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def stimulus(r):
    return dict(question=r['question'],gold=' OR '.join(map(str,r['answers'])),pred=r['pred'])

def identity(r,protocol):
    value=dict(protocol=protocol['protocol_id'],category=int(r['category']),status=r['status'],**stimulus(r))
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def classify(stim,out):
    from codex_judge_client import call
    return call(PROMPT.format(**stim),out)

def run_batch(inputs,out,protocol,workers=8,classifier=classify):
    assert protocol['model']==MODEL and protocol['status']=='READY' and protocol['calibration_passed']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/'judge_decisions.jsonl';seen={}
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);key=(r['cohort'],r['arm'],r['id'])
            assert key not in seen and r['protocol_id']==protocol['protocol_id'] and r['model']==MODEL
            seen[key]=r
    rows={};groups={};unchanged=0
    for p in inputs:
        with Path(p).open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);key=(r['cohort'],r['arm'],r['id'])
                assert key not in rows and int(r['category']) in (1,2,3,4,5)
                assert r['status'] in ('ok','OOM')
                assert isinstance(r['pred'],str) if r['status']=='ok' else r['pred'] is None
                rows[key]=r;digest=identity(r,protocol)
                if key in seen:
                    assert seen[key]['stimulus_digest']==digest;unchanged+=1
                else:groups.setdefault(digest,[]).append(r)
    assert set(seen).issubset(rows)
    def one(item):
        digest,group=item;r=group[0];cat=int(r['category'])
        entry=out/'cache'/digest;receipt=entry/'decision.json'
        if receipt.exists():
            vote=json.loads(receipt.read_text())
            assert vote['protocol_id']==protocol['protocol_id'] and vote['stimulus_digest']==digest
            return digest,vote,None
        if r['status']=='OOM':
            vote=dict(judge_correct=None,judge_raw='UNSCORED_OOM',usage=None)
        elif cat==5:
            vote=dict(judge_correct=int(not r['pred'].strip() or bool(_REFUSAL_RE.search(r['pred']))),
                judge_raw='LOCAL_ABSTENTION_RULE',usage=None)
        else:
            result=classifier(stimulus(r),entry/f'attempt-{time.time_ns()}')
            label=result['answer'].strip()
            if not result['ok'] or label not in ('CORRECT','WRONG'):
                return digest,None,dict(stimulus_digest=digest,status='JUDGE_FAILED_UNSCORED',result=result)
            vote=dict(judge_correct=int(label=='CORRECT'),judge_raw=label,usage=result.get('usage'))
        vote.update(model=MODEL,protocol_id=protocol['protocol_id'],stimulus_digest=digest)
        write_json(receipt,vote)
        return digest,vote,None
    errors=[];items=iter(groups.items());finished_groups=0
    with dest.open('a',encoding='utf-8') as output, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        def enqueue():
            try:item=next(items)
            except StopIteration:return False
            active[pool.submit(one,item)]=item[0];return True
        for _ in range(workers):
            if not enqueue():break
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                del active[future]
                digest,vote,error=future.result()
                if error:errors.append(error)
                else:
                    finished_groups+=1
                    for r in groups[digest]:
                        key=(r['cohort'],r['arm'],r['id'])
                        row={k:r[k] for k in ('cohort','arm','id','shard','category','status')}
                        row.update(vote,reasoning_effort='low',seed=None)
                        output.write(json.dumps(row,ensure_ascii=False)+'\n');seen[key]=row
                    output.flush()
                if not errors:enqueue()
                write_json(out/'progress.json',dict(decisions=len(seen),available=len(rows),
                    groups_completed=finished_groups,errors=len(errors),at=time.time()))
    complete=set(seen)==set(rows) and not errors
    summary={}
    for cohort in sorted({r['cohort'] for r in rows.values()}):
        summary[cohort]={}
        for arm in sorted({r['arm'] for r in rows.values()}):
            records=[r for r in seen.values() if r['cohort']==cohort and r['arm']==arm]
            valid=[r['judge_correct'] for r in records if r['judge_correct'] is not None]
            full=len(records)==1986 and len(valid)==1986
            summary[cohort][arm]=dict(records=len(records),scored=len(valid),expected=1986,
                score=100*sum(valid)/1986 if full else None,complete=full)
    report=dict(available_complete=complete,decisions=len(seen),available_records=len(rows),
        oom=sum(r['judge_correct'] is None for r in seen.values()),errors=errors,
        protocol_id=protocol['protocol_id'],model=MODEL,summary=summary,
        caching='Identical question, gold, candidate, category and protocol receive one shared blinded judgment.')
    write_json(out/'status.json',report)
    return report
