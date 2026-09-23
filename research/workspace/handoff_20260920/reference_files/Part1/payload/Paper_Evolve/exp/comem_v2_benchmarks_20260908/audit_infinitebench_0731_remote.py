"""Two newly complete InfiniteBench En.MC baselines; CPU, no model loading."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
sys.path[:0]=[str(B),str(R/'workspace/COMem'),str(R/'workspace/exp')]
import prepare_infinitebench as ib
from benchmark_pack import BOUNDARY,tokenize_explicit_prompt
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def near(a,b):assert abs(a-b)<1e-9,(a,b)
task='longbook_choice_eng';n=229;targets=('pub','pub_sink');references=('fix_all','j0')
snapshot=read(B/'heartbeat_extended_20260909_0731_snapshot.json')
eligible=set(snapshot['eligible_native_markers'])
assert all(str(O/'infinitebench'/arm/f'{task}_s{shard}of4'/'COMPLETED.json') in eligible for arm in targets for shard in range(4))
records={};dbs={};sources={};common=None
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'new pub/pub_sink English long-book MC only; V2/j0 metadata and packs used as previously scored reference, not rescored',
   'writeback':[],'comparisons':[],'execution':'remote CPU, CUDA hidden, tokenizer only, no model construction'}
for arm in targets+references:
    records[arm]={};sources[arm]=[]
    for shard in range(4):
        run=O/'infinitebench'/arm/f'{task}_s{shard}of4';m=read(run/'COMPLETED.json');c=read(run/'run_config.json');o=c['driver_options']
        assert m['status']=='completed';out=Path(m['output_dir']);rr=rows(out/f'{task}_{shard}.jsonl')
        expected=list(range(shard,n,4));assert [r['index'] for r in rr]==expected
        assert m['new_generations']+m['reused_generations']==len(expected)
        assert c['arm']==arm and c['scoring']=='official helper'
        assert o['model_path']==str(R/'models/Qwen3-8B') and o['dtype']=='bfloat16' and o['attn_impl']=='sdpa'
        assert o['resume_j']==12 and o['topk']==12 and o['chunk_size']==512 and o['selector']=='bm25' and o['sink_tokens']=='bos'
        assert o['tasks']==[task] and o['num_shards']==4 and o['shard_index']==shard and o['max_samples']==-1
        assert o['prompt_style']=='yarn-mistral' and o['max_new_tokens'] is None and o['baseline']=='none' and o['lora_adapter']==''
        shared={k:v for k,v in o.items() if k!='shard_index'}
        if common is None:common=shared
        assert common==shared
        if arm in targets:
            ri=m['reader'];assert ri['arm']==arm and ri['effective_j']==(36 if arm=='cbos' else 12)
            assert ri['class']==('CoMemLower' if arm=='cbos' else 'CoMem') and ri['write_sink']==(arm!='pub')
            assert ri['model_config']['model_type']=='qwen3' and ri['model_config']['num_hidden_layers']==36
            assert ri['kernel_policy']=='s15 repeat_kv: use_gqa_in_sdpa=False (all arms)'
            score=read(out/'scores.json')[task];assert score['n']==len(rr)
            near(score['score'],sum(r['score'] for r in rr)/len(rr))
            dbs[arm,shard]=sqlite3.connect(f'file:{run/"generations.sqlite3"}?mode=ro',uri=True)
        for r in rr:assert r['index'] not in records[arm];records[arm][r['index']]=r
        sources[arm].append(str(run))
    assert sorted(records[arm])==list(range(n)) and len({r['id'] for r in records[arm].values()})==n
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
ranges={k:[] for k in ('input_tokens','context_tokens','query_tokens','read_pack_tokens')}
scores={arm:[] for arm in targets};keys=[]
for i,src in enumerate(ib.iter_task(task)):
    prompt=ib.format_prompt(dict(src,context=src['context']+BOUNDARY),task,prompt_style='yarn-mistral')
    ids,nctx,selected,pack=tokenize_explicit_prompt(tok,prompt,str(src.get('input') or src.get('question') or ''),512,'bm25',12)
    key=hashlib.sha256(json.dumps({'tokens':ids.tolist(),'generation':{'context_token_count':nctx,'selected_indices':selected,'chunk_size':512,'max_new_tokens':40}},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    keys.append(key)
    for arm in targets+references:
        r=records[arm][i]
        assert r['id']==src['id'] and r['task']==task and r['answers']==ib.get_answer(src,task) and r['status']=='ok'
        assert r['input_tokens']==ids.numel() and r['pack']==pack and r['prompt_style']=='yarn-mistral' and r['truncation']=='none'
        if arm in targets:
            value=float(ib.score_prediction(r['pred'],src,task));near(value,r['score']);assert 0<=value<=1
            hit=dbs[arm,i%4].execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
            assert hit and json.loads(hit[0])==r['pred'] and hit[1]==ids.numel();scores[arm].append(value)
    for k in ranges:ranges[k].append(pack[k])
    if (i+1)%25==0:print(f'IB MC two-arm official/source/token/pack/cache checks {i+1}/{n}',flush=True)
assert len(keys)==n
for db in dbs.values():db.close()
means={}
for arm in targets:
    assert len(scores[arm])==n;mean=100*sum(scores[arm])/n;means[arm]=mean
    d['writeback'].append({'benchmark':'infinitebench','task':task,'arm':arm,'complete':True,'n':n,'expected_n':n,'metric':'official option accuracy','correct':int(sum(scores[arm])),'score_percent':mean,'display':f'{mean:.2f}',
        'official_rescores':n,'exact_source_token_pack_own_generation_cache_checks':n,'same_source_tokens_and_ordered_pack_with_previously_verified_v2_j0':True,'sources':sources[arm]})
# Read saved reference scores solely for descriptive side-by-side placement; no
# rerun of their scorer and no new claim about their already-checked cache entries.
for arm in references:means[arm]=100*sum(records[arm][i]['score'] for i in range(n))/n
for arm in targets:d['comparisons'].append({'task':task,'new_arm':arm,'score_percent':means[arm],'v2_previously_verified_score_percent':means['fix_all'],
    'full_recompute_previously_verified_score_percent':means['j0'],'v2_minus_new_arm_pp':means['fix_all']-means[arm],'paired_n':n,'significance_test_performed':False,
    'reference_scorer_and_cache_not_rerun':True})
d['protocol']={'model':'stock Qwen3-8B BF16, SDPA, same repeat_kv policy; j12, isolated full KV effective j36',
    'template':'official vendored yarn-mistral bare template, no chat wrapper, explicit independently tokenized context/query boundary',
    'retrieval':'BM25 top12, chunks512, no source truncation/padding, identical ordered selected packs',
    'generation':'greedy cap40, first-token EOS suppressed, subsequent natural EOS; no fixed-length serving comparison',
    'scoring':'vendored official English long-book multiple-choice option accuracy (including official answer-text extraction); score_percent=100*correct/229',
    'coverage':'One complete English long-book MC task for two newly complete baseline arms, not full InfiniteBench; isolated-KV MC remains partial/unknown',
    'token_ranges':{k:{'min':min(v),'max':max(v)} for k,v in ranges.items()}}
d['checks']={'official_rescores':458,'new_source_token_pack_exact_input_own_cache_checks':458,'reference_scorer_calls':0,'reference_cache_lookups':0,
    'same_source_and_pack_checks_for_reference':458,'models_constructed':0,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
    'input_keys_sha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest()}
d['snapshot_timestamp_utc']=snapshot['timestamp_utc']
d['snapshot_file']=str(B/'heartbeat_extended_20260909_0731_snapshot.json')
d['checks'].update(torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),cuda_initialized=torch.cuda.is_initialized())
assert all(x in (0.0,1.0) for vals in scores.values() for x in vals)
assert not torch.cuda.is_initialized()
d['finished_at_utc']=datetime.now(timezone.utc).isoformat()
dest=B/'heartbeat_infinitebench_20260909_0731_new.json';dest.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'scores_percent':means,'writeback':d['writeback'],'checks':d['checks'],'output':str(dest)}),flush=True)

