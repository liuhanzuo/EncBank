"""New full Qasper fix_none and new full oracle single-hop fix_none only; CPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';sys.path.insert(0,str(B))
import official_qa_driver as qa
from transformers import AutoTokenizer
from summarize_oracle_support import check_source,check_options
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-10,(a,b)
def key_for(context,query,selected,cap):
    content={'tokens':[context+query],'generation':{'context_token_count':len(context),'selected_indices':selected,'chunk_size':512,'max_new_tokens':cap}}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only; new completed groups only; no model/timing',
        'writeback':[],'comparisons':[],'oracle':[]}
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
qasper={};sources=[];matched_options=None;options=None
for shard in range(2):
    run=R/'outputs/benchmarks_v2/full/longbench/fix_none'/f'qasper_s{shard}of2'
    marker=read(run/'COMPLETED.json');config=read(run/'run_config.json');data=rows(run/'predictions.jsonl');opts=config['options'];check_options(config,'fix_none','longbench')
    assert marker['status']=='completed' and marker['n']==len(data)==100
    assert marker['reader']['class']=='EncbankLower' and marker['reader']['effective_j']==marker['reader']['requested_j']==12 and marker['reader']['write_sink'] is True
    assert opts['tasks']==['qasper'] and opts['num_shards']==2 and opts['shard_index']==shard and opts['max_samples']==-1
    samples=list(qa.longbench_samples(SimpleNamespace(**opts)))
    assert [{'index':s['index'],'id':s['id'],'task':s['task']} for s in samples]==config['expected']
    db=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
    for row,sample in zip(data,samples):
        assert all(row.get(k)==v for k,v in sample.items() if k!='marked_prompt') and row['status']=='ok'
        value,decoded=qa.score_prediction(row['pred'],sample);near(value,row['score']);assert decoded==row['scored_prediction']
        ids,nctx,selected,pack=qa.tokenize_pack(tok,sample,512,'bm25',12);assert pack==row['pack']
        key=key_for(ids[0,:nctx].tolist(),ids[0,nctx:].tolist(),selected,sample['max_new_tokens'])
        got=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
        assert got and json.loads(got[0])==row['pred'] and got[1]==ids.numel()
        assert row['index'] not in qasper;qasper[row['index']]=row
    db.close();sources.append(str(run));options=opts
assert len(qasper)==200 and sorted(qasper)==list(range(200))
mean=100*sum(qasper[i]['score'] for i in range(200))/200
report['writeback'].append({'benchmark':'longbench','task':'qasper','arm':'fix_none','n':200,'expected_n':200,'score_percent':mean,'display':f'{mean:.2f}',
    'complete':True,'sources':sources,'official_rescores':200,'exact_source_pack_cache_checks':200,'generation_cap':128})
for arm in ('fix_all','j0'):
    reference={}
    for shard in range(2):
        run=R/'outputs/benchmarks_v2/full/longbench'/arm/f'qasper_s{shard}of2';config=read(run/'run_config.json');assert read(run/'COMPLETED.json')['n']==100
        assert {k:v for k,v in config['options'].items() if k not in ('arm','out','shard_index')}=={k:v for k,v in options.items() if k not in ('arm','out','shard_index')}
        for r in rows(run/'predictions.jsonl'):reference[r['index']]=r
    assert len(reference)==200
    for i in range(200):assert all(qasper[i][k]==reference[i][k] for k in qasper[i] if k not in ('pred','scored_prediction','score'))
    delta=100*sum(reference[i]['score']-qasper[i]['score'] for i in range(200))/200
    report['comparisons'].append({'benchmark':'longbench','task':'qasper','n':200,'left_arm':arm,'right_arm':'fix_none','difference_pp':delta,
        'left_score_previously_verified':100*sum(reference[i]['score'] for i in range(200))/200,'paired_source_pack':True})
dest=R/'outputs/diagnostics/heartbeat_accuracy_20260909_0027_new.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'qasper':report['writeback'],'comparisons':report['comparisons']}),flush=True)

# Only the newly completed oracle category4/fix_none group is rescored. V2 references
# were fully independently checked in the 00:05 report and are aligned by fixture ID.
fixtures=[f for f in rows(B/'data/oracle_support/inputs.jsonl') if f['task']=='locomo_category_4/oracle'];fmap={f['id']:f for f in fixtures};assert len(fmap)==840
contexts=read(B/'data/oracle_support/contexts.json');plan=read(B/'oracle_support_full_plan.json')
values={};oracle_sources=[]
for job in plan['jobs']:
    if job['arm']!='fix_none' or not any(c['key']=='locomo_category_4/oracle' for c in job['cells']):continue
    run=Path(job['output']);marker=read(run/'COMPLETED.json');config=read(run/'run_config.json');data=rows(run/'predictions.jsonl');check_options(config,'fix_none')
    expected={(c['key'],i) for c in job['cells'] for i in c['indices']};assert marker['status']=='completed' and marker['n']==len(data)==len(expected)
    assert {(r['task'],r['index']) for r in data}==expected
    db=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
    for row in data:
        f=fmap[row['id']];check_source(row,f,oracle=True);value,decoded=qa.score_prediction(row['pred'],f['sample']);near(value,row['score']);assert decoded==row['scored_prediction']
        context=contexts[f['context_key']]['context_ids'];key=key_for(context,f['query_ids'],f['oracle_pack']['selected_indices'],f['sample']['max_new_tokens'])
        got=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone();assert got and json.loads(got[0])==row['pred'] and got[1]==len(context)+len(f['query_ids'])
        assert f['id'] not in values;values[f['id']]=row['score']
    db.close();oracle_sources.append(str(run))
assert set(values)==set(fmap)
v2={}
for folder in (R/'outputs/oracle_support/full/fix_all').glob('locomo_category_4_oracle_part*'):
    assert (folder/'COMPLETED.json').exists()
    for row in rows(folder/'predictions.jsonl'):check_source(row,fmap[row['id']],oracle=True);v2[row['id']]=row['score']
for folder in (R/'outputs/benchmarks_v2/full/locomo/fix_all').glob('all_s*of10'):
    assert (folder/'COMPLETED.json').exists()
    for row in rows(folder/'predictions.jsonl'):
        f=fmap.get(row['id'])
        if f and f['natural_pack']==f['oracle_pack']:check_source(row,f);v2[row['id']]=row['score']
assert set(v2)==set(fmap)
score=100*sum(values[f['id']] for f in fixtures)/840;v2score=100*sum(v2[f['id']] for f in fixtures)/840
prior=read(R/'outputs/diagnostics/heartbeat_accuracy_20260909_0005_oracle.json');old=next(c for c in prior['cells'] if c['task']=='locomo_category_4/oracle' and c['arm']=='fix_all');near(v2score,old['oracle_score_percent'])
delta=100*sum(v2[f['id']]-values[f['id']] for f in fixtures)/840
report['oracle'].append({'task':'locomo_category_4/oracle','arm':'fix_none','n':840,'expected_n':840,'oracle_complete':True,'score_percent':score,'display':f'{score:.2f}',
    'oracle_v2_score_previously_verified':v2score,'v2_minus_fix_none_pp':delta,'natural_control_planned':False,'natural_score_percent':None,'oracle_minus_natural_pp':None,
    'official_rescores':840,'exact_fixture_pack_cache_checks':840,'sources':oracle_sources,'generation_cap':50})
report['protocol']={'model':'Qwen3-8B BF16, j12; fix_none lower_layers=[] with same sink/upper reader',
    'qasper':'official LongBench Qasper template, Qwen user chat thinking disabled, greedy cap128, full source/no truncation, BM25 top12/chunk512; paired with existing V2/j0 input packs',
    'oracle':'Same prepared eligible 840-example LoCoMo category4 subset and exact forced pack; no natural fix_none control planned. V2 reference reused from prior independent full-group audit; no global benchmark inference.'}
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'oracle_new':report['oracle']}),flush=True)
