"""Full-support model/benchmark macros; incomplete/OOM/Judge-pending cells stay null."""
import collections, json
from common import ROOT, MODELS, ARMS, SHARDS, dump
from evaluation_scope import read_scope

scope=read_scope()
report={'models':{},'not_supported_on_these_backbones':['InfLLM','MemoryLLM'],
        'generation_verified_complete':True,'all_metrics_complete':True,
        'scope':scope,'deferred_benchmarks':scope['deferred']}
for cfg in MODELS:
    values=collections.defaultdict(list);support=collections.Counter();verified_shards=0
    for s in range(SHARDS):
        out=ROOT/'results'/cfg['name']/f'shard{s}'
        v=out/'verified_summary.json'
        if not v.exists():continue
        checked=json.loads(v.read_text())
        assert checked['verified'] and checked['scope_id']==scope['scope_id'];verified_shards+=1
        judges={}
        if (out/'judge_decisions.jsonl').exists():
            for line in (out/'judge_decisions.jsonl').read_text(encoding='utf-8').splitlines():
                r=json.loads(line);key=(r['id'],r['arm']);assert key not in judges
                assert r['judge_correct'] in (0,1);judges[key]=r['judge_correct']
        with (out/'predictions.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);b=r['benchmark'];arm=r['arm']
                assert b in scope['benchmarks']
                cell=(r['task'],r['length']) if b!='locomo' else ('all','all')
                score=judges.get((r['id'],arm)) if b=='locomo' else r['score']
                if r['status']!='ok':score=None
                values[(arm,b,cell)].append(score);support[(arm,b)]+=1
    report['generation_verified_complete'] &= verified_shards==SHARDS
    table={}
    for arm in ARMS:
        table[arm]={}
        for b,n,expected_cells in [('ruler',1500,15),('longeval',500,5),('longbench',1150,6),('babilong',2100,21),('locomo',1986,1)]:
            if b not in scope['benchmarks']:continue
            cells={c:v for (a,bb,c),v in values.items() if a==arm and bb==b}
            valid=verified_shards==SHARDS and support[(arm,b)]==n and len(cells)==expected_cells and all(vs and all(v is not None for v in vs) for vs in cells.values())
            mean=sum(sum(vs)/len(vs) for vs in cells.values())/expected_cells*100 if valid else None
            table[arm][b]=dict(score=mean,records=support[(arm,b)],expected=n,complete=valid)
            report['all_metrics_complete'] &= valid
    report['models'][cfg['name']]=dict(j=cfg['j'],verified_shards=verified_shards,table=table)
dump(ROOT/'summary.json',report)
print(json.dumps(report,indent=2))
