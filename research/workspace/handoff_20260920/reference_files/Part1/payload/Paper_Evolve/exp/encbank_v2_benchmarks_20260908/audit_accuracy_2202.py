"""Light CPU audit of matched RULER outputs and newly complete LongBench only."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import string
from datetime import datetime

B=Path(__file__).resolve().parent
O=B/'results/remote/outputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def near(a,b):assert abs(a-b)<1e-10,(a,b)
def pure(path,names,env):
    tree=ast.parse(path.read_text(encoding='utf-8'));tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in tree.body}==names
    exec(compile(tree,str(path),'exec'),env);return env

old=read(B/'heartbeat_longbench_20260908_2050_writeback.json')
progress={k:read(B/'results'/f'{k}_progress.json') for k in ('main','ruler_matched16k')}
manifest=read(B/'data/ruler_matched16k/manifest.json')
plan=read(B/'ruler_matched16k_full_plan.json')
state=read(O/'queue_ruler_matched16k/full/state.json')
report={'timestamp':datetime.now().astimezone().isoformat(),'progress':{k:{a:d[a] for a in ('completed_jobs','planned_jobs','observed_predictions','all_complete')} for k,d in progress.items()},'ruler':[],'ruler_pairs':[],'longbench_new':[],'qa_comparisons':[],'limitations':[]}
identities=('task','length','i','answers','n_tokens')
evidence=('input_ids_sha256','selected_context_ids_sha256','query_chunk_sha256','selected_context_indices')
scores={};data={}
for task in ('niah_multikey_1','niah_single_2','variable_tracking'):
    fp=B/'data/ruler_matched16k'/f'{task}_16k.jsonl';fixture=rows(fp)
    assert len(fixture)==50 and [r['i'] for r in fixture]==list(range(50))
    assert hashlib.sha256(fp.read_bytes()).hexdigest()==manifest['tasks'][task]['sha256']
    for r in fixture:
        assert r['n_tokens']==len(r['input_ids'])
        chunks=[r['input_ids'][i:i+512] for i in range(0,len(r['input_ids']),512)]
        assert r['context_chunk_lengths']==[len(c) for c in chunks[:-1]] and r['query_chunk_ids']==chunks[-1]
        assert len(set(r['selected_context_indices']))==len(r['selected_context_indices'])==12
        assert all(0<=i<len(chunks)-1 for i in r['selected_context_indices'])
        selected=[t for i in r['selected_context_indices'] for t in chunks[i]]
        assert digest(r['input_ids'])==r['input_ids_sha256'] and digest(selected)==r['selected_context_ids_sha256'] and digest(chunks[-1])==r['query_chunk_sha256']
        assert r['selector']==('iter_bm25' if task=='variable_tracking' else 'bm25')
        assert r['max_new_tokens']==(60 if task=='variable_tracking' else 48)
    for family,arm in (('ruler_matched16k','fix_all'),('ruler_matched16k','j0'),('trained_pub','pub'),('cacheblend16','cacheblend16')):
        p=O/family/'full/ruler'/arm/f'{task}_16k.json';d=read(p);rs=d['rows'];opts=d['args']
        assert len(rs)==50 and [r['i'] for r in rs]==list(range(50))
        assert all({k:r[k] for k in identities}=={k:f[k] for k in identities} for r,f in zip(rs,fixture))
        assert opts['j']==12 and opts['n']==50 and opts['topk']==12 and opts['seed']==42 and opts['selector']=='auto' and opts['iter_hop_topk']==4 and opts['max_new_tokens']==48
        assert opts['tasks']==task and opts['lengths']=='16k' and opts['arms']==arm
        assert (opts['adapter'].endswith('/8b_j12_pub_4k/final') if family=='trained_pub' else opts['adapter']=='') and not opts['adapter_pt']
        fractional=[]
        for r in rs:
            prediction=r[f'{arm}_out'];assert isinstance(prediction,str) and prediction!='[OOM]'
            score=Fraction(sum(x.lower() in prediction.lower() for x in r['answers']),len(r['answers']))
            near(float(score),r[f'{arm}_recall']);fractional.append(score)
        mean=float(100*sum(fractional)/50);near(mean,d['summary'][f'{task}/16k'][arm]);scores[(task,arm)]=fractional
        if family=='ruler_matched16k':
            receipt=d['matched_input_fixture'];assert receipt['samples']==receipt['selected_pack_checks']==50 and receipt['smoke_only'] is False
            assert receipt['sha256']==manifest['tasks'][task]['sha256'] and receipt['path'].endswith(fp.name)
            assert all(all(r[k]==f[k] for k in evidence) for r,f in zip(rs,fixture))
            job=state['jobs'][f'ruler__{arm}__{task}_16k'];assert job['status']=='completed' and job['exit_code']==0
            planned=next(j for j in plan['jobs'] if j['id']==f'ruler__{arm}__{task}_16k')
            assert '--smoke-only' not in planned['argv']
            assert planned['argv'][planned['argv'].index('--expected-fixture-sha256')+1]==receipt['sha256']
            raw=O/'ruler_matched16k/full/ruler'/arm/f'{task}_16k_attempts'/Path(receipt['raw_attempt_output']).name
            rawrows=read(raw)['rows'];assert [{k:v for k,v in r.items() if k not in evidence} for r in rs]==rawrows
            pc=next(c for c in progress['ruler_matched16k']['cells'] if c['arm']==arm and c['cell']==f'{task}/16k')
            assert pc['complete'] and pc['n']==pc['expected_n']==50;near(mean,pc['score_percent'])
        report['ruler'].append({'task':task,'length':'16k','arm':arm,'family':family,'n':50,'score_percent':mean,'source':str(p.relative_to(B)),'draw':'remote_310_siphash24_fixed16k','source_id_fields_match_fixture':True,'runtime_pack_checks':50 if family=='ruler_matched16k' else None,'pack_evidence':'runtime checked against fixed input' if family=='ruler_matched16k' else 'reconstructed original selector; baseline lacks persisted indices'})
    for other in ('j0','pub','cacheblend16'):
        diff=[a-b for a,b in zip(scores[(task,'fix_all')],scores[(task,other)])]
        value=float(100*sum(diff)/50)
        report['ruler_pairs'].append({'task':task,'contrast':f'V2 minus {other}','n':50,'delta_points':value,'better':sum(x>0 for x in diff),'tie':sum(x==0 for x in diff),'worse':sum(x<0 for x in diff),'descriptive_only':True})

lb=pure(B/'protocol_sources/longbench/metrics.py',{'normalize_answer','f1_score','qa_f1_score'},{'re':re,'string':string,'Counter':Counter})['qa_f1_score']
caps=read(B/'protocol_sources/longbench/config/dataset2maxlen.json')
previous={(c['arm'],c['task']) for c in old['longbench'] if c['family']=='benchmarks_v2'}
new=[c for c in progress['main']['cells'] if c['benchmark']=='longbench' and c['complete'] and (c['arm'],c['cell']) not in previous]
for cell in new:
    arm,task=cell['arm'],cell['cell'];src=rows(B/'data/longbench'/f'{task}.jsonl');n=150 if task=='multifieldqa_en' else 200
    assert len(src)==n;allrows=[];paths=[]
    for shard in range(2):
        p=O/'benchmarks_v2/full/longbench'/arm/f'{task}_s{shard}of2';c=read(p/'run_config.json');done=read(p/'COMPLETED.json');rs=rows(p/'predictions.jsonl');summary=read(p/'scores.json')
        expected=[{'index':i,'id':str(src[i].get('_id',src[i].get('id',i))),'task':task} for i in range(shard,n,2)]
        assert c['expected']==expected and [{k:r[k] for k in ('index','id','task')} for r in rs]==expected
        assert done['status']=='completed' and done['n']==len(rs)==n//2 and done['files'][0]['records']==len(rs)
        assert done['new_generations']+done['reused_generations']==len(rs)
        opt=c['options'];assert opt['seed']==42 and opt['max_samples']==-1 and opt['num_shards']==2 and opt['shard_index']==shard
        assert opt['topk']==12 and opt['chunk_size']==512 and opt['selector']=='bm25' and not opt['adapter']
        assert c['protocol']['template']=='tokenizer.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)'
        for r in rs:
            s=src[r['index']];assert r['answers']==s['answers'] and r['question']==s['input']
            assert isinstance(r['pred'],str) and r['pred']!='[OOM]' and r['status']=='ok' and r['max_new_tokens']==caps[task]
            assert r['pack']['truncation']=='none' and r['scored_prediction']==r['pred']
            near(max(lb(r['pred'],str(a)) for a in s['answers']),r['score'])
        mean=100*sum(r['score'] for r in rs)/len(rs);near(mean,summary['tasks'][task]['score']);assert done['scores']==summary
        data[(task,arm,shard)]=(c,rs);allrows.extend(rs);paths.append(str(p.relative_to(B)))
    allrows.sort(key=lambda r:r['index']);assert [r['index'] for r in allrows]==list(range(n)) and len({r['id'] for r in allrows})==n
    mean=100*sum(r['score'] for r in allrows)/n;near(mean,cell['score_percent'])
    report['longbench_new'].append({'task':task,'arm':arm,'n':n,'score_percent':mean,'display':f'{mean:.2f}','sources':paths,'generation_cap':caps[task]})

for task in sorted({c['task'] for c in report['longbench_new']}):
    for arm in [c['arm'] for c in report['longbench_new'] if c['task']==task]:
        for family,other in (('trained_pub','pub'),('cacheblend16','cacheblend16'),('benchmarks_v2','j0')):
            if other==arm:continue
            if family=='benchmarks_v2' and (task,other,0) not in data:continue
            for shard in range(2):
                c,rs=data[(task,arm,shard)];p=O/family/'full/longbench'/other/f'{task}_s{shard}of2'
                oc=read(p/'run_config.json');orr=rows(p/'predictions.jsonl');assert read(p/'COMPLETED.json')['status']=='completed'
                assert len(rs)==len(orr) and all(all(r[k]==o[k] for k in ('index','id','question','answers','pack','max_new_tokens')) for r,o in zip(rs,orr))
                assert {k:v for k,v in c['options'].items() if k not in ('arm','out','adapter')}=={k:v for k,v in oc['options'].items() if k not in ('arm','out','adapter')}
                assert {k:v for k,v in c['protocol'].items() if k!='cacheblend'}=={k:v for k,v in oc['protocol'].items() if k!='cacheblend'}
                fa={x['path']:x for x in c['files']};fb={x['path']:x for x in oc['files']}
                assert all(fa[k]==fb[k] for k in fa.keys()&fb.keys())
            ref=next(x for x in report['longbench_new'] if x['task']==task and x['arm']==arm)
            if family=='benchmarks_v2':othermean=next(x['score_percent'] for x in report['longbench_new'] if x['task']==task and x['arm']==other)
            else:othermean=next(x['score_percent'] for x in old['longbench'] if x['task']==task and x['family']==family)
            report['qa_comparisons'].append({'task':task,'arm':arm,'other':other,'n':ref['n'],'delta_points':ref['score_percent']-othermean,'same_source_and_pack':True,'shared_protocol_and_source_descriptors_equal':True})
report['limitations']=['The new matched 16k draw is distinct from the historical Python3.13 draw; keep the old grid and paired analyses separate.','Runtime pack equality is directly verified for new V2/j0. The two baseline packs are reconstructed from the original deterministic selector and shared sample fields, because their original outputs did not save pack indices.','Scores and paired mean differences are descriptive; no significance or equivalence intervals were computed.','Only newly complete LongBench cells were rescored; earlier baseline scores were reused from the completed 20:50 audit after source/pack/protocol alignment.']
(B/'heartbeat_accuracy_20260908_2202_writeback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('ruler','limitations')},ensure_ascii=False,indent=2))
