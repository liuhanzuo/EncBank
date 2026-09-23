"""Independent official rescoring and exact cache/fixture check for full oracle groups."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,hashlib,sqlite3
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908';sys.path.insert(0,str(B))
import official_qa_driver as qa
from summarize_oracle_support import check_source,check_options,natural_generation_key
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-10,(a,b)
fixtures=rows(B/'data/oracle_support/inputs.jsonl');contexts=read(B/'data/oracle_support/contexts.json');manifest=read(B/'data/oracle_support/manifest.json')
plans=read(B/'oracle_support_full_plan.json');pairing=read(R/'outputs/diagnostics/oracle_support_paired.json')
fmap={f['id']:f for f in fixtures};groups=defaultdict(list)
for f in fixtures:groups[f['task']].append(f)
raw={};cons={};checked=0
for job in plans['jobs']:
    path=Path(job['output'])
    if not (path/'COMPLETED.json').exists():continue
    marker=read(path/'COMPLETED.json');config=read(path/'run_config.json');data=rows(path/'predictions.jsonl');check_options(config,job['arm'])
    expected={(c['key'],i) for c in job['cells'] for i in c['indices']}
    assert marker['status']=='completed' and marker['n']==len(data)==len(expected)
    assert {(r['task'],r['index']) for r in data}==expected
    db=cons.setdefault(str(path),sqlite3.connect(f'file:{path / "generations.sqlite3"}?mode=ro',uri=True))
    for row in data:
        f=fmap[row['id']];check_source(row,f,oracle=True)
        value,decoded=qa.score_prediction(row['pred'],f['sample']);near(value,row['score']);assert decoded==row['scored_prediction']
        context=contexts[f['context_key']]['context_ids'];tokens=context+f['query_ids']
        content={'tokens':[tokens],'generation':{'context_token_count':len(context),'selected_indices':f['oracle_pack']['selected_indices'],'chunk_size':512,'max_new_tokens':f['sample']['max_new_tokens']}}
        key=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        cached=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
        assert cached and json.loads(cached[0])==row['pred'] and cached[1]==len(tokens)
        assert (job['arm'],row['id']) not in raw;raw[(job['arm'],row['id'])]=row;checked+=1
natural={};natural_connections={}
for benchmark in ('locomo','longbench'):
    for arm in ('fix_all','j0','fix_none'):
        folders=list((R/'outputs/benchmarks_v2/full'/benchmark/arm).glob('all_s*of10' if benchmark=='locomo' else 'hotpotqa_s*of2'))
        for folder in folders:
            if not (folder/'COMPLETED.json').exists():continue
            config=read(folder/'run_config.json');check_options(config,arm,benchmark)
            data=rows(folder/'predictions.jsonl');assert len(data)==len(config['expected'])
            assert [{'index':r['index'],'id':r['id'],'task':r['task']} for r in data]==config['expected']
            natural_connections[str(folder)]=sqlite3.connect(f'file:{folder / "generations.sqlite3"}?mode=ro',uri=True)
            for row in data:
                if row['id'] in fmap:natural[(arm,row['id'])]=(row,str(folder))
cells=[];valid={};natural_checked=0
for task,items in sorted(groups.items()):
    for arm in ('fix_all','j0','fix_none'):
        oracle_values=[];pairs=[];generated=derived=visible=0
        for f in items:
            row=raw.get((arm,f['id']));nat=natural.get((arm,f['id']))
            if nat:
                nr,folder=nat;check_source(nr,f)
                value,decoded=qa.score_prediction(nr['pred'],f['sample']);near(value,nr['score']);assert decoded==nr['scored_prediction']
                key,token_n=natural_generation_key(f,contexts);cached=natural_connections[folder].execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
                assert cached and json.loads(cached[0])==nr['pred'] and cached[1]==token_n;natural_checked+=1
            if row is not None:score=row['score'];generated+=1
            elif nat and f['natural_pack']==f['oracle_pack']:score=nat[0]['score'];derived+=1
            else:continue
            oracle_values.append(score);valid[(task,arm,f['id'])]=score
            if nat:pairs.append((nat[0]['score'],score));visible+=int(f['natural_all_required_visible'])
        n=len(items);complete=len(oracle_values)==n;paired=len(pairs)==n
        cell={'task':task,'arm':arm,'n':n,'oracle_observed_n':len(oracle_values),'oracle_complete':complete,'paired_observed_n':len(pairs),'paired_complete':paired,
              'oracle_score_percent':100*sum(oracle_values)/n if complete else None,
              'natural_score_percent':100*sum(x for x,y in pairs)/n if paired else None,
              'oracle_minus_natural_pp':100*sum(y-x for x,y in pairs)/n if paired else None,
              'new_oracle_generations':generated,'identical_pack_derived':derived,
              'natural_all_required_visible_n':manifest['tasks'][task]['natural_all_required_visible_n'],
              'read_pack_token_delta_counts':manifest['tasks'][task]['read_pack_token_delta_counts']}
        cells.append(cell)
        old=next(c for c in pairing['cells'] if c['task']==task and c['arm']==arm)
        if old['paired_complete']:
            assert paired;near(cell['oracle_score_percent'],old['oracle_f1_percent']);near(cell['natural_score_percent'],old['natural_f1_percent']);near(cell['oracle_minus_natural_pp'],old['oracle_minus_natural_pp'])
visibility=[]
for task,items in groups.items():
    diff=[valid[(task,'fix_all',f['id'])]-valid[(task,'fix_none',f['id'])] for f in items if (task,'fix_all',f['id']) in valid and (task,'fix_none',f['id']) in valid]
    visibility.append({'task':task,'n':len(items),'observed_n':len(diff),'complete':len(diff)==len(items),'v2_minus_fix_none_pp':100*sum(diff)/len(items) if len(diff)==len(items) else None})
for c in list(cons.values())+list(natural_connections.values()):c.close()
result={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only; official rescore and read-only SQLite token/input/pack/prediction checks; no model or timing',
        'cells':cells,'oracle_visibility':visibility,'official_generated_scores_checked':checked,'natural_exact_cache_checks':natural_checked,
        'selection':{k:manifest[k] for k in ('source_rows','eligible_inputs','selection_status_counts')},
        'budget':'Same number of selected chunks (top12 or all if fewer); required support chunks inserted, remaining slots filled by original BM25 rank; chronological order. Token budget can differ when boundary chunks differ; delta histogram retained.',
        'boundary':'Annotated support must align and fit the chunk budget. This selected eligible subset excludes unknown alignment, missing annotations, adversarial questions and over-budget evidence. It cannot estimate overall retrieval recall; partial groups stay null. No natural fix_none LoCoMo control was planned.',
        'source':str(B/'data/oracle_support/inputs.jsonl')}
(R/'outputs/diagnostics/heartbeat_accuracy_20260909_0005_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('cells','oracle_visibility','official_generated_scores_checked','natural_exact_cache_checks')},indent=2))
