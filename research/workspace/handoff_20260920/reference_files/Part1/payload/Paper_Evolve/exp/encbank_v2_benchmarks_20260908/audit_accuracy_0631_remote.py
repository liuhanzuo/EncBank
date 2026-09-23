"""New complete LongBench stock-baseline cells only; remote CPU verification."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';sys.path.insert(0,str(B))
import official_qa_driver as qa
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer
from summarize_oracle_support import check_options
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-10,(a,b)
def key_for(ids,nctx,selected,cap):
    c={'tokens':ids.tolist(),'generation':{'context_token_count':nctx,'selected_indices':selected,'chunk_size':512,'max_new_tokens':cap}}
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only; no model inference or timing','new_complete_cells':[],'comparisons':[],'unknown':[]}
dest=R/'outputs/diagnostics/heartbeat_accuracy_20260909_0631_qa.json';tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
snapshot=read(R/'outputs/diagnostics/heartbeat_accuracy_20260909_0631_snapshot.json')
assert {j['id'] for j in snapshot['completed_jobs'] if j['benchmark']=='longbench' and j['arm'] in ('pub_sink','cbos')}=={'longbench__pub_sink__musique_s0of2','longbench__pub_sink__musique_s1of2','longbench__cbos__musique_s0of2','longbench__cbos__musique_s1of2'}
report['snapshot_timestamp']=snapshot['timestamp']
audited={};configs={}
for task,arms,n in [('musique',('pub_sink','cbos'),200)]:
  for arm in arms:
    values={};sources=[];caps=set()
    for shard in range(2):
      run=R/'outputs/benchmarks_v2/full/longbench'/arm/f'{task}_s{shard}of2'
      marker=read(run/'COMPLETED.json');config=read(run/'run_config.json');data=rows(run/'predictions.jsonl');opts=config['options'];check_options(config,arm,'longbench')
      assert marker['status']=='completed' and marker['n']==len(data)==n//2
      assert opts['tasks']==[task] and opts['num_shards']==2 and opts['shard_index']==shard and opts['max_samples']==-1
      info=marker['reader'];assert info['requested_j']==12 and info['arm']==arm
      if arm=='cbos':assert info['class']=='EncbankLower' and info['effective_j']==36 and info['write_sink'] is True
      else:assert info['class']=='Encbank' and info['effective_j']==12 and info['write_sink']==(arm=='pub_sink')
      samples=list(qa.longbench_samples(SimpleNamespace(**opts)));assert [{'index':s['index'],'id':s['id'],'task':s['task']} for s in samples]==config['expected']
      db=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
      for row,sample in zip(data,samples):
        assert all(row.get(k)==v for k,v in sample.items() if k!='marked_prompt') and row['status']=='ok'
        value,decoded=qa.score_prediction(row['pred'],sample);near(value,row['score']);assert decoded==row['scored_prediction']
        ids,nctx,selected,pack=qa.tokenize_pack(tok,sample,512,'bm25',12);assert pack==row['pack']
        got=db.execute('select prediction,n_tokens from generations where key=?',(key_for(ids,nctx,selected,sample['max_new_tokens']),)).fetchone()
        assert got and json.loads(got[0])==row['pred'] and got[1]==ids.numel()
        assert row['index'] not in values;values[row['index']]=row;caps.add(sample['max_new_tokens'])
      db.close();sources.append(str(run));configs[(task,arm)]=config
    assert len(values)==n and sorted(values)==list(range(n)) and len(caps)==1
    mean=100*sum(values[i]['score'] for i in range(n))/n;audited[(task,arm)]=values
    report['new_complete_cells'].append({'benchmark':'longbench','task':task,'arm':arm,'n':n,'expected_n':n,'score_percent':mean,'display':f'{mean:.2f}','complete':True,'sources':sources,'official_rescores':n,'exact_source_pack_cache_checks':n,'generation_cap':list(caps)[0],'reader':{k:info[k] for k in ('class','arm','requested_j','effective_j','write_sink')}})
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['new_complete_cells'][-1]),flush=True)

for task,arms,n in [('musique',('pub_sink','cbos'),200)]:
  references=['fix_all','j0','pub']+(['fix_none'] if task=='qasper' else [])
  for arm in references:
    values={}
    for shard in range(2):
      run=R/'outputs/benchmarks_v2/full/longbench'/arm/f'{task}_s{shard}of2';config=read(run/'run_config.json');marker=read(run/'COMPLETED.json');assert marker['status']=='completed' and marker['n']==n//2
      for row in rows(run/'predictions.jsonl'):assert row['index'] not in values;values[row['index']]=row
      configs[(task,arm)]=config
    assert len(values)==n and sorted(values)==list(range(n));audited[(task,arm)]=values
  pairs=[(ref,arm) for arm in arms for ref in references]
  for left,right in pairs:
    lv,rv=audited[(task,left)],audited[(task,right)]
    trim=lambda c:{k:v for k,v in c['options'].items() if k not in ('arm','out','shard_index')}
    assert trim(configs[(task,left)])==trim(configs[(task,right)])
    for i in range(n):assert all(lv[i][k]==rv[i][k] for k in lv[i] if k not in ('pred','scored_prediction','score'))
    report['comparisons'].append({'benchmark':'longbench','task':task,'n':n,'left_arm':left,'right_arm':right,'difference_pp':100*sum(lv[i]['score']-rv[i]['score'] for i in range(n))/n,'left_score_percent':100*sum(lv[i]['score'] for i in range(n))/n,'right_score_percent':100*sum(rv[i]['score'] for i in range(n))/n,'paired_source_pack':True,'reference_audit':'V2/j0 (no new fix_none task) already independently verified on earlier heartbeat; only realigned here'})
report['unknown']=[]
report['runtime']={'torch_cpu_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'cuda_initialized':torch.cuda.is_initialized(),'model_loaded':False}
assert not torch.cuda.is_initialized()
report['protocol']={'model':'Qwen3-8B BF16, SDPA, seed42, no adapter; j12 for pub/pub_sink; cbos effectivej36 isolated full KV, not CacheBlend','prompt':'Official LongBench dataset prompt in Qwen one-user chat, thinking disabled, greedy generation; no truncation; full source context and query','retrieval':'BM25 top12 or all shorter contexts, chunk512, selected pack in source order; matched full tokens/query/pack/cap within each task','generation_caps':{'musique':32},'metric':'Official LongBench qa_f1_score, max over accepted answers, 0..100','official_metric_source':str(B/'protocol_sources/longbench/metrics.py'),'source_data':str(B/'data/longbench'),'driver':str(B/'official_qa_driver.py')}
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'comparisons':report['comparisons'],'unknown':report['unknown']}),flush=True)

