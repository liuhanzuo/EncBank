"""CPU-only saved-output rescore; incomplete cohorts never receive final scores."""
import collections, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.extend(['/srv/encbank/comem_infra_recheck_20260912/deps',
 '/srv/encbank/comem_followups_20260913/COMem','/srv/encbank/comem_frozen_j12_20260912'])
from common import MODELS, dump
from binding import OLD, old_predictions, sha256
from scoring import scores

EXPECTED=dict(ruler=1500,longeval=500,longbench=1150,babilong=2100,locomo=1986)

def main():
    status=json.loads((ROOT/'status.json').read_text())
    plan=json.loads((ROOT/'plan.json').read_text())
    records={};judge={}
    for p in (OLD/'judge_gpt6_astra/formal_new_models/judge_decisions.jsonl',ROOT/'judge_gpt6_astra/judge_decisions.jsonl'):
        if not p.exists():continue
        for line in p.open(encoding='utf-8'):
            r=json.loads(line);key=(r['cohort'],r['arm'],r['id'])
            assert key not in judge;judge[key]=r
    for cfg in MODELS:
        gate=ROOT/'results/quant-cell'/cfg['name']/'cell/complete.json'
        if gate.exists() and json.loads(gate.read_text()).get('h16_parity_passed'):
            frozen=json.loads((ROOT/'verified_inputs.json').read_text())['models'][cfg['name']]
            for p,h in frozen['old_prediction_files'].items():assert sha256(OLD/p)==h
            for r in old_predictions(cfg).values():
                r=dict(r,arm='cache_h16',origin='reused old H16 after exact wrapper parity')
                records[(cfg['name'],r['arm'],r['id'])]=r
    for task in plan['tasks']:
        if task['kind']!='evaluation' or status['tasks'].get(task['id'],{}).get('state')!='COMPLETED':continue
        p=(ROOT/task['complete']).parent/'predictions.jsonl'
        for line in p.open(encoding='utf-8'):
            r=json.loads(line)
            if r['arm']=='cache_h16':continue # parity validation, not duplicate samples
            key=(task['model'],r['arm'],r['id']);assert key not in records;records[key]=r
    buckets=collections.defaultdict(list);rescored=0
    for (model,arm,uid),r in records.items():
        r=dict(r)
        if r['status']=='ok' and r['benchmark']!='locomo':
            res=scores(r,r['text'])['score'];assert abs(res-r['score'])<1e-10,(uid,res,r['score']);rescored+=1
        if r['benchmark']=='locomo':
            original_arm='cache_lora' if arm=='cache_h16' else arm
            d=judge.get((model,original_arm,uid));r['score']=d['judge_correct'] if d else None
        buckets[(model,arm,r['benchmark'])].append(r)
    table={}
    for (model,arm,b),rows in sorted(buckets.items()):
        valid=[r for r in rows if r['status']=='ok' and r['score'] is not None]
        cell_scores=collections.defaultdict(list)
        for r in valid:cell_scores[(r['task'],r['length'])].append(r['score'])
        ready=len(rows)==EXPECTED[b] and len(valid)==EXPECTED[b]
        if b=='locomo':value=100*sum(r['score'] for r in valid)/EXPECTED[b] if ready else None
        else:value=100*sum(sum(v)/len(v) for v in cell_scores.values())/len(cell_scores) if ready else None
        report=dict(expected=EXPECTED[b],generated=len(rows),scored=len(valid),complete=ready,score=value,
            statuses=dict(collections.Counter(r['status'] for r in rows)),
            cap_without_stop=sum(r.get('reached_cap_without_stop',False) for r in rows),
            cells={str(k):dict(n=len(v),mean=100*sum(v)/len(v)) for k,v in cell_scores.items()})
        if b=='locomo':
            for label,categories,n in [('C1-4',{1,2,3,4},1540),('C5',{5},446)]:
                part=[r for r in valid if int(r['extra']['category']) in categories]
                report[label]=dict(expected=n,scored=len(part),score=100*sum(r['score'] for r in part)/n if len(part)==n else None)
        table.setdefault(model,{}).setdefault(arm,{})[b]=report
    dump(ROOT/'delivery/summary.json',dict(table=table,cpu_rescored=rescored,
        precision_only='rank32/4000 H16/H8/H4',training_extension='rank128/8000 H16 plus midpoint4000 LongEval',
        timing='No fixed-workload latency measured',protocol='unchanged prompt-pack fixtures'))
    lines=['# 新模型量化和加强训练结果','',
           '未完成或评分缺失显示“待完成”，不按已完成子集计算正式分数。H16 / H8 / H4 使用原 rank32/4000 adapter；r128 另列。','',
           '| 模型 | 方法 | RULER | LongEval | LongBench | BABILong | LoCoMo全量 |','|---|---|---:|---:|---:|---:|---:|']
    for model,arms in table.items():
        for arm,data in arms.items():
            vals=[f"{data[b]['score']:.2f}" if b in data and data[b]['score'] is not None else '待完成' for b in EXPECTED]
            lines.append('| '+model+' | '+arm+' | '+' | '.join(vals)+' |')
    (ROOT/'delivery/RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(cpu_rescored=rescored,methods=sum(map(len,table.values())))))

if __name__=='__main__':main()
