"""Export verified saved answers, with the exact original question and gold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=('Qwen3.5-9B','Qwen3.8-27B')
def main():
    data=json.loads((ROOT/'data/locomo10.json').read_text())
    if isinstance(data,dict):data=list(data.values())
    questions=[]
    for ci,d in enumerate(data):
        if not isinstance(d.get('conversation',{}),dict):continue
        for qi,qa in enumerate(d.get('qa',[])):
            q=(qa.get('question','') or '').strip()
            if not q:continue
            ans=qa.get('answer')
            if ans is None:ans=qa.get('adversarial_answer','')
            questions.append(dict(question=q,answers=[ans if isinstance(ans,str) else str(ans)],
                category=qa.get('category',-1),source_id=f'conv{ci}_qa{qi}'))
    assert len(questions)==1986
    out=ROOT/'judge_inputs';out.mkdir(exist_ok=True)
    manifest=[]
    for model in MODELS:
        for s in range(4):
            shard=ROOT/'results_locomo'/model/f'shard{s}'
            if not (shard/'verified_summary.json').exists():continue
            v=json.loads((shard/'verified_summary.json').read_text())
            assert v['verified'] and v['scope_id']=='locomo-resumed-astra-20260916'
            dest=out/f'{model}_shard{s}.jsonl'
            if not dest.exists():
                rows=[];seen=set()
                with (shard/'predictions.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        r=json.loads(line);assert r['benchmark']=='locomo'
                        ix=int(r['id'].rsplit(':',1)[1]);q=questions[ix]
                        assert q['answers']==r['answers'] and q['category']==r['extra']['category']
                        assert q['source_id']==r['extra']['source_id'] and r['index']==ix
                        key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
                        rows.append(dict(cohort=model,arm=r['arm'],id=r['id'],shard=s,
                            question=q['question'],answers=q['answers'],category=q['category'],
                            pred=r['text'],status=r['status']))
                assert len(rows)==v['records']
                tmp=dest.with_suffix('.tmp')
                tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
                tmp.replace(dest)
            manifest.append(dict(file=dest.name,cohort=model,shard=s,records=v['records'],oom=v['oom_records']))
    result=dict(verified_shards=len(manifest),expected_shards=8,files=manifest)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
