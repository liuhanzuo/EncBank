"""Only newly complete MFQA, natural Hotpot fix_none, and its n8 oracle pair."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/comem_v2_20260908'); B=R/'workspace/exp/comem_v2_benchmarks_20260908'; sys.path.insert(0,str(B))
import official_qa_driver as qa
from transformers import AutoTokenizer
from summarize_oracle_support import check_source,check_options,natural_generation_key
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-10,(a,b)
def key_for(context,query,selected,cap):
    content={'tokens':[context+query],'generation':{'context_token_count':len(context),'selected_indices':selected,'chunk_size':512,'max_new_tokens':cap}}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only, newly complete groups; no models or timing','writeback':[],'comparisons':[],'oracle':[]}
dest=R/'outputs/diagnostics/heartbeat_accuracy_20260909_0129_qa.json'
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
audited={}; configs={}
for task,arms,n in [('multifieldqa_en',('fix_all','j0'),150),('hotpotqa',('fix_none',),200)]:
  for arm in arms:
    values={}; sources=[]; caps=set()
    for shard in range(2):
      run=R/'outputs/benchmarks_v2/full/longbench'/arm/f'{task}_s{shard}of2'
      marker=read(run/'COMPLETED.json');config=read(run/'run_config.json');data=rows(run/'predictions.jsonl');opts=config['options'];check_options(config,arm,'longbench')
      assert marker['status']=='completed' and marker['n']==len(data)==n//2
      assert opts['tasks']==[task] and opts['num_shards']==2 and opts['shard_index']==shard and opts['max_samples']==-1
      assert marker['reader']['requested_j']==12
      if arm=='j0':assert marker['reader']['class']=='CoMem' and marker['reader']['effective_j']==0 and marker['reader']['write_sink'] is False
      else:assert marker['reader']['class']=='CoMemLower' and marker['reader']['effective_j']==12 and marker['reader']['write_sink'] is True
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
        assert row['index'] not in values;values[row['index']]=row;caps.add(sample['max_new_tokens'])
      db.close();sources.append(str(run));configs[(task,arm)]=config
    assert len(values)==n and sorted(values)==list(range(n));assert len(caps)==1
    mean=100*sum(values[i]['score'] for i in range(n))/n;audited[(task,arm)]=values
    report['writeback'].append({'benchmark':'longbench','task':task,'arm':arm,'n':n,'expected_n':n,'score_percent':mean,'display':f'{mean:.2f}','complete':True,'sources':sources,'official_rescores':n,'exact_source_pack_cache_checks':n,'generation_cap':list(caps)[0]})
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['writeback'][-1]),flush=True)

for task,left,right,n in [('multifieldqa_en','fix_all','j0',150),('hotpotqa','fix_all','fix_none',200),('hotpotqa','j0','fix_none',200)]:
  for arm in (left,right):
    if (task,arm) in audited:continue
    reference={}
    for shard in range(2):
      run=R/'outputs/benchmarks_v2/full/longbench'/arm/f'{task}_s{shard}of2';config=read(run/'run_config.json');assert read(run/'COMPLETED.json')['n']==n//2
      for row in rows(run/'predictions.jsonl'):reference[row['index']]=row
      configs[(task,arm)]=config
    assert len(reference)==n;audited[(task,arm)]=reference
  lv,rv=audited[(task,left)],audited[(task,right)]
  trim=lambda c:{k:v for k,v in c['options'].items() if k not in ('arm','out','shard_index')}
  assert trim(configs[(task,left)])==trim(configs[(task,right)])
  for i in range(n):assert all(lv[i][k]==rv[i][k] for k in lv[i] if k not in ('pred','scored_prediction','score'))
  report['comparisons'].append({'benchmark':'longbench','task':task,'n':n,'left_arm':left,'right_arm':right,'difference_pp':100*sum(lv[i]['score']-rv[i]['score'] for i in range(n))/n,'left_score_percent':100*sum(lv[i]['score'] for i in range(n))/n,'right_score_percent':100*sum(rv[i]['score'] for i in range(n))/n,'paired_source_pack':True})
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'comparisons':report['comparisons']}),flush=True)

# Audit only the newly available n8 fix_none natural/oracle pair. V2/j0 were checked before.
fixtures=[f for f in rows(B/'data/oracle_support/inputs.jsonl') if f['task']=='hotpotqa/oracle'];assert len(fixtures)==8
fmap={f['id']:f for f in fixtures};contexts=read(B/'data/oracle_support/contexts.json');plan=read(B/'oracle_support_full_plan.json')
raw={};sources=[]
for job in plan['jobs']:
  if job['arm']!='fix_none' or not any(c['key']=='hotpotqa/oracle' for c in job['cells']):continue
  run=Path(job['output']);config=read(run/'run_config.json');marker=read(run/'COMPLETED.json');data=rows(run/'predictions.jsonl');check_options(config,'fix_none')
  assert marker['status']=='completed' and marker['n']==len(data)==job['expected_n']
  expected={(c['key'],i) for c in job['cells'] for i in c['indices']};assert {(r['task'],r['index']) for r in data}==expected
  db=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
  for row in data:
    f=fmap[row['id']];check_source(row,f,oracle=True);value,decoded=qa.score_prediction(row['pred'],f['sample']);near(value,row['score']);assert decoded==row['scored_prediction']
    context=contexts[f['context_key']]['context_ids'];key=key_for(context,f['query_ids'],f['oracle_pack']['selected_indices'],f['sample']['max_new_tokens'])
    got=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone();assert got and json.loads(got[0])==row['pred'] and got[1]==len(context)+len(f['query_ids'])
    assert f['id'] not in raw;raw[f['id']]=row
  db.close();sources.append(str(run))
natural={r['id']:r for r in audited[('hotpotqa','fix_none')].values()};paired=[];generated=derived=0
for f in fixtures:
  nr=natural[f['id']];check_source(nr,f)
  # The full task audit already reconstructed exact tokens, pack and SQLite entries.
  if f['id'] in raw:orr=raw[f['id']];generated+=1
  else:assert f['natural_pack']==f['oracle_pack'];orr=nr;derived+=1
  paired.append({'id':f['id'],'natural_score':nr['score'],'oracle_score':orr['score'],'identical_pack':f['natural_pack']==f['oracle_pack'],'natural_all_required_visible':f['natural_all_required_visible'],'read_pack_tokens_natural':f['natural_pack']['read_pack_tokens'],'read_pack_tokens_oracle':f['oracle_pack']['read_pack_tokens']})
assert generated==1 and derived==7 and len(paired)==8
manifest=read(B/'data/oracle_support/manifest.json');pairing=read(R/'outputs/diagnostics/oracle_support_paired.json')
cell={'task':'hotpotqa/oracle','arm':'fix_none','n':8,'oracle_complete':True,'paired_complete':True,'oracle_score_percent':100*sum(p['oracle_score'] for p in paired)/8,'natural_score_percent':100*sum(p['natural_score'] for p in paired)/8,'oracle_minus_natural_pp':100*sum(p['oracle_score']-p['natural_score'] for p in paired)/8,'new_oracle_generations':generated,'identical_pack_derived':derived,'natural_all_required_visible_n':sum(p['natural_all_required_visible'] for p in paired),'read_pack_token_delta_counts':manifest['tasks']['hotpotqa/oracle']['read_pack_token_delta_counts'],'exact_new_oracle_cache_checks':generated,'natural_source_pack_cache_checks_within_full_task':8,'sources':sources,'pairs':paired}
ref=next(c for c in pairing['cells'] if c['task']=='hotpotqa/oracle' and c['arm']=='fix_none');assert ref['paired_complete'];near(cell['oracle_score_percent'],ref['oracle_f1_percent']);near(cell['natural_score_percent'],ref['natural_f1_percent']);near(cell['oracle_minus_natural_pp'],ref['oracle_minus_natural_pp'])
report['oracle'].append(cell)
report['protocol']={'model':'Qwen3-8B BF16; j12; no LoRA; fix_none lower_layers=[] with same sink/upper reader','prompt':'Official LongBench template; one-user Qwen chat, thinking disabled; greedy decoding; full source context, no truncation','retrieval':'BM25 top12 (or all when shorter), chunks512; chronological selected-pack order; same source/full-token/query/pack and generation settings within each compared task','metric':'Official LongBench max-over-answers normalized token F1; per-sample recomputed; reported 0..100','source_files':[str(B/'data/longbench/multifieldqa_en.jsonl'),str(B/'data/longbench/hotpotqa.jsonl')],'scoring_entry':str(B/'official_qa_driver.py'),'oracle_boundary':'Hotpot n8 eligible annotation-aligned subset from 200; 192 unresolved alignment are excluded, so not population recall. Same chunk budget; required chunks inserted then original BM25 rank fill, chronological order. Seven identical natural/oracle packs reuse validated natural generation; one different-pack generation. Missing/unplanned natural LoCoMo fix_none remains unknown, not zero.'}
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'oracle':report['oracle']}),flush=True)
