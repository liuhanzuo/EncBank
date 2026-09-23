"""CPU-only readback audit of completed QA and 16k strong-baseline outputs."""
import ast
from collections import Counter
import datetime
import json
from pathlib import Path
import re
import string

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
OUT = B / 'results/remote/outputs'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def near(a, b):
    assert abs(a-b) < 1e-10, (a, b)
def functions(path, names, ns):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in tree.body} == names
    exec(compile(tree, str(path), 'exec'), ns)
    return ns

scorer = functions(B/'protocol_sources/longbench/metrics.py',
    {'normalize_answer','f1_score','qa_f1_score'}, {'re':re,'string':string,'Counter':Counter})['qa_f1_score']
ruler_scorer = functions(ROOT/'COMem/eval/ruler.py', {'_string_match_all_one'}, {'re':re})['_string_match_all_one']
caps = read(B/'protocol_sources/longbench/config/dataset2maxlen.json')
tasks = ['narrativeqa','qasper','hotpotqa','2wikimqa','musique','multifieldqa_en']
families = [('trained_pub','pub'),('cacheblend16','cacheblend16'),('benchmarks_v2','fix_all'),('benchmarks_v2','j0')]
progress = {name:read(B/'results'/f'{name}_progress.json') for name in ('main','trained_pub','cacheblend16')}
audit = {'timestamp':datetime.datetime.now().astimezone().isoformat(), 'snapshots':{}, 'longbench':[], 'ruler':[], 'comparisons':[], 'warnings':[]}
records, configs = {}, {}
for name,d in progress.items():
    audit['snapshots'][name] = {k:d[k] for k in ('planned_jobs','completed_jobs','observed_predictions','reused_predictions','all_complete')}

for task in tasks:
    source = rows(B/'data/longbench'/f'{task}.jsonl')
    n = 150 if task=='multifieldqa_en' else 200
    assert len(source)==n
    for family, arm in families:
        dirs = [OUT/family/'full/longbench'/arm/f'{task}_s{s}of2' for s in range(2)]
        if not all((p/'COMPLETED.json').is_file() for p in dirs):
            continue
        all_rows=[]
        for shard,p in enumerate(dirs):
            c,done,score = (read(p/f) for f in ('run_config.json','COMPLETED.json','scores.json'))
            rs=rows(p/'predictions.jsonl')
            expected = [{'index':i,'id':str(source[i].get('_id',source[i].get('id',i))),'task':task} for i in range(n) if i%2==shard]
            assert c['expected']==expected
            assert [{k:r[k] for k in ('index','id','task')} for r in rs]==expected
            assert done['status']=='completed' and done['n']==len(rs)==n//2
            assert done['new_generations']+done['reused_generations']==len(rs)
            assert len(done['files'])==1 and done['files'][0]['records']==len(rs)
            o=c['options']
            assert o['seed']==42 and o['j']==12 and o['dtype']=='bfloat16' and o['attn_impl']=='sdpa'
            assert o['topk']==12 and o['chunk_size']==512 and o['selector']=='bm25' and o['max_samples']==-1
            assert o['tasks']==[task] and o['num_shards']==2 and o['shard_index']==shard and o['arm']==arm
            assert c['protocol']['source_context']=='whole source; no truncation'
            assert c['protocol']['template']=='tokenizer.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)'
            if family=='trained_pub':
                assert o['adapter']=='/data/liuhanzuo/comem_v2_20260908/outputs/8b_j12_pub_4k/final'
                assert done['reader']['class']=='CoMem' and done['reader']['write_sink'] is False
                files={Path(f['path']).name:f for f in c['files'] if '/8b_j12_pub_4k/final/' in f['path']}
                assert {k:v['size'] for k,v in files.items()}=={'adapter.pt':232891470,'adapter_model.safetensors':232829136,'adapter_config.json':617}
            if arm=='cacheblend16':
                assert not o['adapter']
                assert c['protocol']['cacheblend']['bootstrap_full_layers']==2
                assert c['protocol']['cacheblend']['selection']=='layer-1 V summed squared contextual deviation; floor(.16 * context_tokens)'
                assert c['protocol']['cacheblend']['chunk_write_sink'] is False
                assert any(f['path'].endswith('/cacheblend_contextual.py') for f in c['files'])
            for r in rs:
                s=source[r['index']]
                assert r['question']==s['input'] and r['answers']==s['answers']
                assert r['status']=='ok' and isinstance(r['pred'],str) and r['pred']!='[OOM]'
                assert r['max_new_tokens']==caps[task]
                assert r['pack']['truncation']=='none'
                assert r['pack']['context_query_boundary']=='explicit; independently tokenized; no padding'
                assert len(set(r['pack']['selected_indices']))==len(r['pack']['selected_indices'])<=12
                expected_score=max(scorer(r['pred'],str(a)) for a in s['answers'])
                near(expected_score,r['score'])
                assert r['scored_prediction']==r['pred']
            mean=100*sum(r['score'] for r in rs)/len(rs)
            near(mean,score['tasks'][task]['score'])
            assert score['tasks'][task]['n']==len(rs)
            assert done['scores']==score
            configs[(family,arm,task,shard)]=c
            all_rows.extend(rs)
        all_rows.sort(key=lambda r:r['index'])
        assert [r['index'] for r in all_rows]==list(range(n)) and len({r['id'] for r in all_rows})==n
        mean=100*sum(r['score'] for r in all_rows)/n
        prog=progress[family if family!='benchmarks_v2' else 'main']
        pc=next(c for c in prog['cells'] if c['benchmark']=='longbench' and c['arm']==arm and c['cell']==task)
        assert pc['complete'] and pc['n']==pc['expected_n']==n
        near(mean,pc['score_percent'])
        records[(family,arm,task)]=all_rows
        audit['longbench'].append({'family':family,'arm':arm,'task':task,'n':n,'score_percent':mean,'display':f'{mean:.2f}',
            'sources':[str(p.relative_to(B)) for p in dirs], 'max_new_tokens':caps[task]})

for task in tasks:
    eligible=[(f,a) for f,a in families if (f,a,task) in records]
    ref=eligible[0]
    for f,a in eligible[1:]:
        rr=records[(*ref,task)]; other=records[(f,a,task)]
        assert all(x['pack']==y['pack'] for x,y in zip(rr,other)), (task,ref,f,a,'pack mismatch')
        for shard in range(2):
            ca=configs[(*ref,task,shard)]; cb=configs[(f,a,task,shard)]
            for k in set(ca['options'])|set(cb['options']):
                if k not in ('arm','out','adapter'):
                    assert ca['options'].get(k)==cb['options'].get(k),(task,k)
            assert {k:v for k,v in ca['protocol'].items() if k!='cacheblend'}=={k:v for k,v in cb['protocol'].items() if k!='cacheblend'}
            fa={x['path']:x for x in ca['files']}; fb={x['path']:x for x in cb['files']}
            for path in fa.keys()&fb.keys():
                assert fa[path]==fb[path], ('source descriptor mismatch',path)
        audit['comparisons'].append({'task':task,'reference':list(ref),'other':[f,a],'n':len(rr),'all_pack_fields_equal':True,'shared_configuration_equal':True,'shared_source_descriptors_equal':True})

for family,arm in families[:2]:
    state=read(OUT/f'queue_{family}'/'full/state.json')
    for task in ('niah_multikey_1','niah_single_2','variable_tracking'):
        p=OUT/family/'full/ruler'/arm/f'{task}_16k.json'; d=read(p); rs=d['rows']; o=d['args']
        assert len(rs)==o['n']==50 and sorted(r['i'] for r in rs)==list(range(50))
        assert len({r['i'] for r in rs})==50
        assert d['arms']==[arm] and o['arms']==arm and o['seed']==42 and o['j']==12 and o['topk']==12
        assert o['selector']=='auto' and o['iter_hop_topk']==4 and o['max_new_tokens']==48
        assert o['tasks']==task and o['lengths']=='16k' and not o['adapter_pt']
        assert (o['adapter'].endswith('/8b_j12_pub_4k/final') if family=='trained_pub' else not o['adapter'])
        job=state['jobs'][f'ruler__{arm}__{task}_16k']
        assert job['status']=='completed' and job['exit_code']==0
        log=(OUT/f'queue_{family}'/'full'/Path(job['log']).name).read_text(encoding='utf-8')
        assert 'PYTHONHASHSEED=0' in log and f'[S15] {task}/16k n=50:' in log
        if arm=='cacheblend16':
            assert 'CacheBlend-style Qwen3 port: layer-1 V ranking, floor(.16*n_ctx), full two-layer bootstrap.' in log
        for r in rs:
            assert r['task']==task and r['length']=='16k' and isinstance(r[f'{arm}_out'],str) and r[f'{arm}_out']!='[OOM]'
            near(ruler_scorer(r[f'{arm}_out'],r['answers']),r[f'{arm}_recall'])
        mean=100*sum(r[f'{arm}_recall'] for r in rs)/50
        near(mean,d['summary'][f'{task}/16k'][arm])
        prior=ROOT/'exp/results'/('s15c_ruler_vt_16k.json' if task=='variable_tracking' else 's15_ruler_j12_16k.json')
        old=[r for r in read(prior)['rows'] if r['task']==task and r['length']=='16k']
        assert len(old)==50
        same_source=[{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in rs]==[{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in old]
        if family=='cacheblend16':
            peer=read(OUT/'trained_pub/full/ruler/pub'/f'{task}_16k.json')['rows']
            assert [{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in rs]==[{k:r[k] for k in ('task','length','i','answers','n_tokens')} for r in peer]
        pc=next(c for c in progress[family]['cells'] if c['benchmark']=='ruler' and c['cell']==f'{task}/16k')
        near(mean,pc['score_percent'])
        audit['ruler'].append({'family':family,'arm':arm,'task':task,'length':'16k','n':50,'score_percent':mean,'display':f'{mean:.2f}',
            'source':str(p.relative_to(B)),'original_source_match':str(prior),'source_fields_equal':same_source,'queue_completed_exit0':True,
            'writeback_scope':'independent draw only; not paired with historical V2/j0' if not same_source else 'paired with historical V2/j0'})

for family in ('trained_pub','cacheblend16'):
    assert sum(c['n'] for kind in ('longbench','ruler') for c in audit[kind] if c['family']==family)==1300
audit['warnings'].append('Both new RULER baseline draws match each other in all 150 saved sample identity fields, but ALL 50 samples of EACH task differ from the historical V2/j0 draw despite seed42/hashseed0. They cannot be called paired with the paper V2/j0. RULER also does not store per-ID selected-pack indices.')
audit['warnings'].append('QA metadata reader.class is the underlying CoMem before the ExplicitPackReader/ContextualCacheBlend wrapper; arm, recorded sources, protocol and factory branch establish the selective variant. Per-prediction selected positions are not saved.')
audit['deltas']=[]
by={(c['family'],c['arm'],c['task']):c for c in audit['longbench']}
for task in tasks:
    v=by.get(('benchmarks_v2','fix_all',task))
    if not v: continue
    for family,arm in (('benchmarks_v2','j0'),('trained_pub','pub'),('cacheblend16','cacheblend16')):
        o=by.get((family,arm,task))
        if o:
            delta=v['score_percent']-o['score_percent']
            audit['deltas'].append({'task':task,'contrast':f'V2 minus {family}/{arm}','delta_points':delta,'display':f'{delta:+.2f}'})
(B/'heartbeat_longbench_20260908_2050_writeback.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'snapshot':audit['snapshots'],'longbench':[{k:v for k,v in c.items() if k not in ('sources',)} for c in audit['longbench']], 'ruler':[{k:v for k,v in c.items() if k in ('arm','task','score_percent','n')} for c in audit['ruler']], 'deltas':audit['deltas'], 'matched_pairs':len(audit['comparisons'])},ensure_ascii=False,indent=2))
