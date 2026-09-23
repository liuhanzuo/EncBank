"""CPU-only completed LongEval/InfiniteBench/RULER128 readback audit."""
import ast
import json
from pathlib import Path
import re
import prepare_infinitebench as ib

B=Path(__file__).resolve().parent
O=B/'results/remote/outputs/benchmarks_v2/full'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def near(a,b): assert abs(a-b)<1e-10,(a,b)
def native(p):
    done=read(p/'COMPLETED.json')
    assert done['status']=='completed'
    return p/('attempts/'+done['output_dir'].split('/attempts/')[1]),done
report={'longeval':[],'infinitebench':[],'ruler128':[],'warnings':[]}
configs={}; records={}
for length in ('4k','8k','16k','32k','64k','128k'):
    for arm in ('fix_all','j0'):
        p=O/'longeval'/arm/length; n,done=native(p); d=read(n/f'longeval_{length}.json'); rs=d['records']; c=read(p/'run_config.json'); opts=c['driver_options']
        assert len(rs)==50 and [r['sample_index'] for r in rs]==list(range(50))
        assert opts['seed']==1234 and opts['max_new_tokens']==16 and opts['topk']==12 and opts['chunk_size']==512 and opts['selector']=='bm25'
        assert opts['lengths']==[length] and opts['num_samples']==50 and opts['baseline']=='none'
        assert c['scoring']=='unchanged legacy driver; do not relabel as official'
        for r in rs:
            assert isinstance(r['output'],str) and r['output']!='[OOM]'
            m=re.search(r'\d{4,}',r['output']); pred=m.group(0) if m else ''
            assert pred==r['pred'] and r['correct']==(pred==r['expected'])
        correct=sum(r['correct'] for r in rs); near(correct/50,d['summary']['accuracy'])
        assert d['summary']['correct']==correct and d['summary']['total']==50
        assert done['new_generations']+done['reused_generations']==50
        configs[(length,arm)]=c;records[(length,arm)]=rs
        report['longeval'].append({'task':'lines','length':length,'arm':arm,'n':50,'score_percent':100*correct/50,'metric':'legacy exact first >=4-digit-number accuracy','source':str(n.relative_to(B)/f'longeval_{length}.json')})
    assert {k:v for k,v in configs[(length,'fix_all')].items() if k!='arm'}=={k:v for k,v in configs[(length,'j0')].items() if k!='arm'}
    assert [{k:r[k] for k in ('sample_index','label','expected','n_lines')} for r in records[(length,'fix_all')]]==[{k:r[k] for k in ('sample_index','label','expected','n_lines')} for r in records[(length,'j0')]]

task='longbook_qa_eng'
source=[{k:v for k,v in r.items() if k!='context'} for r in ib.iter_task(task)]
assert len(source)==351
for arm in ('fix_all','j0'):
    rsall=[];paths=[]
    for shard in range(4):
        p=O/'infinitebench'/arm/f'{task}_s{shard}of4';n,done=native(p);c=read(p/'run_config.json'); opts=c['driver_options']; rs=lines(n/f'{task}_{shard}.jsonl')
        assert [r['index'] for r in rs]==list(range(shard,351,4))
        assert done['new_generations']+done['reused_generations']==len(rs)
        assert opts['prompt_style']=='yarn-mistral' and opts['max_new_tokens'] is None and ib.MAX_NEW_TOKENS[task]==40
        assert opts['topk']==12 and opts['chunk_size']==512 and opts['selector']=='bm25' and opts['baseline']=='none'
        assert opts['num_shards']==4 and opts['shard_index']==shard and opts['max_samples']==-1
        assert c['scoring']=='official helper'
        for r in rs:
            s=source[r['index']]
            assert r['id']==s['id'] and r['answers']==ib.get_answer(s,task)
            assert r['status']=='ok' and isinstance(r['pred'],str) and r['pred']!='[OOM]'
            assert r['truncation']==r['pack']['truncation']=='none' and not r['pack']['chat_template']
            assert r['prompt_style']=='yarn-mistral'
            near(ib.score_prediction(r['pred'],s,task),r['score'])
        scores=read(n/'scores.json')[task]; assert scores['n']==len(rs);near(scores['score'],sum(r['score'] for r in rs)/len(rs))
        configs[('ib',arm,shard)]=c;rsall.extend(rs);paths.append(str(n.relative_to(B)/f'{task}_{shard}.jsonl'))
    rsall.sort(key=lambda r:r['index']);assert [r['index'] for r in rsall]==list(range(351)) and len({r['id'] for r in rsall})==351
    records[('ib',arm)]=rsall
    mean=100*sum(r['score'] for r in rsall)/351
    report['infinitebench'].append({'task':task,'arm':arm,'n':351,'score_percent':mean,'display':f'{mean:.2f}','metric':'official answer F1','sources':paths,'generation_cap':40,'prompt_style':'yarn-mistral; no chat template'})
for shard in range(4):
    assert {k:v for k,v in configs[('ib','fix_all',shard)].items() if k!='arm'}=={k:v for k,v in configs[('ib','j0',shard)].items() if k!='arm'}
assert all(a['id']==b['id'] and a['answers']==b['answers'] and a['pack']==b['pack'] for a,b in zip(records[('ib','fix_all')],records[('ib','j0')]))

for arm in ('fix_all','j0'):
    p=O/'ruler'/arm/'niah_single_2_128k.json'
    if not p.is_file():continue
    d=read(p);rs=d['rows']
    if len(rs)!=50 or 'niah_single_2/128k' not in d['summary']:continue
    assert [r['i'] for r in rs]==list(range(50))
    for r in rs:
        near(sum(a.lower() in r[f'{arm}_out'].lower() for a in r['answers'])/len(r['answers']),r[f'{arm}_recall'])
    score=100*sum(r[f'{arm}_recall'] for r in rs)/50; near(score,d['summary']['niah_single_2/128k'][arm])
    state=read(B/'results/remote/outputs/queue_v2/full/state.json')['jobs'][f'ruler__{arm}__niah_single_2_128k']
    assert state['status']=='completed' and state['exit_code']==0
    records[('r128',arm)]=rs
    report['ruler128'].append({'task':'niah_single_2','length':'128k','arm':arm,'n':50,'score_percent':score,'source':str(p.relative_to(B))})
if len(report['ruler128'])==2:
    assert [{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in records[('r128','fix_all')]]==[{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in records[('r128','j0')]]
report['warnings'].append('LongEval is the repository legacy synthetic lines generator and scorer, not an official benchmark reproduction. It saves per-ID labels/answers/line counts but not full pack metadata.')
report['warnings'].append('InfiniteBench En.QA scores are low; matched pack F1 means alone do not establish a significant or general advantage. Generation cap 40 and plain yarn-mistral prompt differ from LongBench.')
(B/'heartbeat_additional_20260908_2050_writeback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:[{a:b for a,b in c.items() if a not in ('source','sources')} for c in v] for k,v in report.items() if k!='warnings'},indent=2))
