"""Only newly complete 2Wiki and LoCoMo; CPU official scoring/input/pack checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import json, hashlib, sqlite3, sys
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from functools import lru_cache
from datetime import datetime, timezone
R=Path('/data/liuhanzuo/encbank_v2_20260908'); B=R/'workspace/exp/encbank_v2_benchmarks_20260908'
sys.path.insert(0,str(B))
import official_qa_driver as driver
from transformers import AutoTokenizer

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-10,(a,b)
class CachedTokenizer:
    def __init__(self):self.tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
    def __getattr__(self,k):return getattr(self.tok,k)
    @lru_cache(maxsize=256)
    def encode(self,text,add_special_tokens=False):return self.tok.encode(text,add_special_tokens=add_special_tokens)

tok=CachedTokenizer()
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only; tokenizer/official scoring; no model',
        'writeback':[],'comparisons':[],'protocols':{},'source_and_pack_checks':{},'natural_rows':{}}
dest=R/'outputs/diagnostics/heartbeat_accuracy_20260909_0005_qa.json'
for benchmark,task,n,shards in [('longbench','2wikimqa',200,2),('locomo','all',1986,10)]:
    byarm={}; expected_by_arm={}; dbs={}; values={}; sources={}; reference_options=None; reference_protocol=None
    for arm in ('fix_all','j0'):
        merged=[]; expected=[]; sources[arm]=[]; perid={}
        for shard in range(shards):
            run=R/'outputs/benchmarks_v2/full'/benchmark/arm/f'{task}_s{shard}of{shards}'
            config=read(run/'run_config.json'); receipt=read(run/'COMPLETED.json'); data=rows(run/'predictions.jsonl')
            assert receipt['status']=='completed' and receipt['n']==len(data)
            assert receipt['reader']['class']==('EncbankLower' if arm=='fix_all' else 'Encbank')
            assert receipt['reader']['requested_j']==12 and receipt['reader']['effective_j']==(12 if arm=='fix_all' else 0)
            opts=config['options']; args=SimpleNamespace(**opts)
            assert opts['seed']==42 and opts['selector']=='bm25' and opts['topk']==12 and opts['chunk_size']==512
            assert opts['dtype']=='bfloat16' and opts['attn_impl']=='sdpa' and not opts['adapter']
            assert opts['num_shards']==shards and opts['shard_index']==shard and opts['max_samples']==-1
            canonical={k:v for k,v in opts.items() if k not in ('arm','out','shard_index')}
            if reference_options is None:reference_options,reference_protocol=canonical,config['protocol']
            assert canonical==reference_options and config['protocol']==reference_protocol
            samples=list(driver.longbench_samples(args) if benchmark=='longbench' else driver.locomo_samples(args))
            assert len(samples)==len(data)==len(config['expected'])
            assert [{'index':s['index'],'id':s['id'],'task':s['task']} for s in samples]==config['expected']
            dbs[(arm,shard)]=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
            for sample,row in zip(samples,data):
                assert all(row.get(k)==v for k,v in sample.items() if k!='marked_prompt')
                assert row['status']=='ok' and row['id'] not in perid
                score,decoded=driver.score_prediction(row['pred'],sample);near(score,row['score']);assert decoded==row['scored_prediction']
                perid[row['id']]=(row,shard)
            merged+=data; expected+=samples; sources[arm].append(str(run))
        merged.sort(key=lambda x:x['index']);expected.sort(key=lambda x:x['index'])
        assert len(merged)==n and [r['index'] for r in merged]==list(range(n))
        assert len({r['id'] for r in merged})==n
        byarm[arm]=perid;expected_by_arm[arm]=expected
        values[arm]=defaultdict(list)
        for row in merged:values[arm][row['task']].append(row['score'])
    assert [s['id'] for s in expected_by_arm['fix_all']]==[s['id'] for s in expected_by_arm['j0']]
    for count,sample in enumerate(expected_by_arm['fix_all'],1):
        left=byarm['fix_all'][sample['id']][0];right=byarm['j0'][sample['id']][0]
        assert all(left[k]==right[k] for k in left if k not in ('pred','scored_prediction','score'))
        ids,context_count,selected,pack=driver.tokenize_pack(tok,sample,512,'bm25',12)
        assert left['pack']==right['pack']==pack
        generation={'context_token_count':context_count,'selected_indices':selected,'chunk_size':512,'max_new_tokens':sample['max_new_tokens']}
        key=hashlib.sha256(json.dumps({'tokens':ids.tolist(),'generation':generation},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for arm in ('fix_all','j0'):
            row,shard=byarm[arm][sample['id']]
            got=dbs[(arm,shard)].execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
            assert got and json.loads(got[0])==row['pred'] and got[1]==ids.numel()
        if count%200==0:print(f'{benchmark}: {count}/{n} exact source/pack/cache pairs',flush=True)
    for con in dbs.values():con.close()
    for task_key in sorted(values['fix_all']):
        for arm in ('fix_all','j0'):
            v=values[arm][task_key];mean=100*sum(v)/len(v)
            report['writeback'].append({'benchmark':benchmark,'task':task_key,'arm':arm,'n':len(v),'expected_n':len(v),
                'score_percent':mean,'display':f'{mean:.2f}','metric':'adversarial accuracy' if task_key=='category_5' else 'official F1',
                'complete':True,'source_and_pack_verified':True,'exact_generation_cache_verified':True,'sources':sources[arm],
                'generation_cap':expected_by_arm[arm][0]['max_new_tokens']})
        a,b=values['fix_all'][task_key],values['j0'][task_key];assert len(a)==len(b)
        delta=100*sum(x-y for x,y in zip(a,b))/len(a)
        report['comparisons'].append({'benchmark':benchmark,'task':task_key,'n':len(a),'v2_minus_j0_pp':delta})
    report['protocols'][benchmark]={'shared_options':reference_options,'protocol':reference_protocol,
        'max_new_tokens':sorted({s['max_new_tokens'] for s in expected_by_arm['fix_all']}),
        'metric_scope':'LoCoMo official answerable F1 by category; explicit option-decoding adaptation for adversarial accuracy; never pool five categories' if benchmark=='locomo' else 'official LongBench max-over-reference QA F1'}
    report['source_and_pack_checks'][benchmark]={'samples_per_arm':n,'arms':2,'official_rescores':2*n,'reconstructed_packs':n,'exact_cache_checks':2*n,'shards_per_arm':shards}
    # Retain compact natural scores for an independent oracle paired-score check.
    report['natural_rows'][benchmark]={arm:{key:{'score':row['score'],'task':row['task']} for key,(row,_) in data.items()} for arm,data in byarm.items()}
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed_benchmark':benchmark,'cells':[r for r in report['writeback'] if r['benchmark']==benchmark],'comparisons':[r for r in report['comparisons'] if r['benchmark']==benchmark]}),flush=True)
