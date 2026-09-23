"""Only newly full pub/BOS LoCoMo arms; remote CPU full official/input/cache audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import json,hashlib,sqlite3,sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter,defaultdict
from functools import lru_cache
from datetime import datetime,timezone
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(16)
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
sys.path.insert(0,str(B))
import official_qa_driver as driver
from transformers import AutoTokenizer
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-9,(a,b)
snapshot=read(B/'heartbeat_extended_20260909_1032_snapshot.json')
eligible=set(snapshot['eligible_native_markers']);state=snapshot['state']
targets=('pub','pub_sink');refs=('fix_all','j0');expected_counts={1:282,2:321,3:96,4:841,5:446}
prior=read(R/'outputs/diagnostics/heartbeat_accuracy_20260909_0005_qa.json')
old_scores=prior['natural_rows']['locomo']
class CachedTokenizer:
    def __init__(self):self.tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
    def __getattr__(self,k):return getattr(self.tok,k)
    @lru_cache(maxsize=256)
    def encode(self,text,add_special_tokens=False):return self.tok.encode(text,add_special_tokens=add_special_tokens)
tok=CachedTokenizer();byarm={};dbs={};samples_byarm={};sources={};shared_options=None;shared_protocol=None
checks=Counter();input_keys=hashlib.sha256();ranges=defaultdict(list)
for arm in targets+refs:
    byarm[arm]={};samples_byarm[arm]={};sources[arm]=[]
    for shard in range(10):
        run=O/'locomo'/arm/f'all_s{shard}of10'
        if arm in targets:
            assert str(run/'COMPLETED.json') in eligible
            job=state['jobs'][f'locomo__{arm}__all_s{shard}of10']
            assert job['status']=='completed' and job['exit_code']==0
            log=Path(job['log']).read_text(errors='replace')
            assert job['gpu'] in state['gpus'] and job['pid']>0 and job['finished_at']>=job['started_at']
            assert '[locomo/'+arm+']' in log
            checks['completed_remote_job_records']+=1
            checks['native_log_admission_lines_present']+=int('[remote admission]' in log)
        cfg=read(run/'run_config.json');marker=read(run/'COMPLETED.json');rr=rows(run/'predictions.jsonl');summary=read(run/'scores.json')
        opts=cfg['options'];args=SimpleNamespace(**opts)
        assert marker['status']=='completed' and marker['n']==len(rr)==len(cfg['expected'])
        assert marker['new_generations']+marker['reused_generations']==len(rr)
        assert opts['benchmark']=='locomo' and opts['arm']==arm and opts['model']==str(R/'models/Qwen3-8B') and opts['j']==12
        assert opts['seed']==42 and opts['selector']=='bm25' and opts['topk']==12 and opts['chunk_size']==512
        assert opts['dtype']=='bfloat16' and opts['attn_impl']=='sdpa' and not opts['adapter']
        assert opts['num_shards']==10 and opts['shard_index']==shard and opts['max_samples']==-1 and opts['categories']==list(range(1,6))
        info=marker['reader'];assert info['class']==('EncbankLower' if arm=='fix_all' else 'Encbank')
        assert info['requested_j']==12 and info['effective_j']==(0 if arm=='j0' else 12)
        if arm in targets:assert info['write_sink']==(arm=='pub_sink')
        assert info['kernel_policy']=='s15 repeat_kv: use_gqa_in_sdpa=False (all arms)'
        mc=info['model_config'];assert mc['num_hidden_layers']==36 and mc['num_key_value_heads']==8 and mc['num_attention_heads']==32 and mc['dtype']=='bfloat16'
        canonical={k:v for k,v in opts.items() if k not in ('arm','out','shard_index')}
        if shared_options is None:shared_options=canonical;shared_protocol=cfg['protocol']
        assert canonical==shared_options and cfg['protocol']==shared_protocol
        # Rebuild source identities and stable category5 option mappings from the official data.
        samples=list(driver.locomo_samples(args));assert len(samples)==len(rr)
        assert [{'index':s['index'],'id':s['id'],'task':s['task']} for s in samples]==cfg['expected']
        assert [s['index'] for s in samples]==list(range(shard,1986,10))
        if arm in targets:dbs[arm,shard]=sqlite3.connect(f'file:{run/"generations.sqlite3"}?mode=ro',uri=True)
        shard_values=defaultdict(list)
        for sample,row in zip(samples,rr):
            assert all(row.get(k)==v for k,v in sample.items() if k!='marked_prompt')
            assert row['status']=='ok' and row['id'] not in byarm[arm]
            if arm in targets:
                score,decoded=driver.score_prediction(row['pred'],sample)
                near(score,row['score']);assert decoded==row['scored_prediction'];checks['official_rescores']+=1
            else:
                saved=old_scores[arm][row['id']];assert saved['task']==row['task'];near(saved['score'],row['score'])
                checks['old_verified_score_identity_checks']+=1
            shard_values[row['task']].append(row['score'])
            byarm[arm][row['id']]=(row,shard);samples_byarm[arm][row['id']]=sample
        if arm in targets:
            for task,v in shard_values.items():
                assert summary['tasks'][task]['n']==len(v);near(summary['tasks'][task]['score'],100*sum(v)/len(v))
        sources[arm].append(str(run))
    assert len(byarm[arm])==1986 and sorted(x[0]['index'] for x in byarm[arm].values())==list(range(1986))
    assert Counter(x[0]['category'] for x in byarm[arm].values())==expected_counts
    print(f'LoCoMo {arm}: source identities and {"new official scores" if arm in targets else "old verified score identities"} 1986/1986',flush=True)
ordered=sorted(samples_byarm['pub'].values(),key=lambda x:x['index'])
for count,sample in enumerate(ordered,1):
    keyid=sample['id']
    assert all(samples_byarm[arm][keyid]==sample for arm in targets+refs)
    ids,nctx,selected,pack=driver.tokenize_pack(tok,sample,512,'bm25',12)
    assert sample['max_new_tokens']==50 and pack['read_pack_tokens']+50<=40960
    generation={'context_token_count':nctx,'selected_indices':selected,'chunk_size':512,'max_new_tokens':50}
    key=hashlib.sha256(json.dumps({'tokens':ids.tolist(),'generation':generation},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    input_keys.update(f'{sample["index"]}:{key}\n'.encode());checks['reconstructed_source_token_ordered_packs']+=1
    for name in ('input_tokens','context_tokens','query_tokens','read_pack_tokens'):ranges[name].append(pack[name])
    for arm in targets+refs:
        row,shard=byarm[arm][keyid];assert row['pack']==pack
        if arm in targets:
            got=dbs[arm,shard].execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
            assert got and json.loads(got[0])==row['pred'] and got[1]==ids.numel()
            checks['new_exact_source_token_pack_own_generation_cache_checks']+=1
        else:checks['old_reference_source_metadata_and_ordered_pack_checks']+=1
    if count%200==0:print(f'LoCoMo source/token/ordered-pack/own-cache {count}/1986 x2 new arms',flush=True)
for db in dbs.values():db.close()
cells=[];comparisons=[];paired=[];reference_cells=[]
for cat,n in expected_counts.items():
    task=f'category_{cat}'
    perarm={arm:[x[0] for x in sorted(byarm[arm].values(),key=lambda x:x[0]['index']) if x[0]['category']==cat] for arm in targets+refs}
    scores={arm:100*sum(r['score'] for r in rr)/n for arm,rr in perarm.items()}
    for arm in refs:
        old=[c for c in prior['writeback'] if c['benchmark']=='locomo' and c['task']==task and c['arm']==arm]
        assert len(old)==1 and old[0]['n']==n;near(scores[arm],old[0]['score_percent'])
        reference_cells.append({'benchmark':'locomo','task':task,'arm':arm,'n':n,'score_percent':scores[arm],'rescored_this_round':False})
    for arm in targets:
        rr=perarm[arm];assert len(rr)==n
        cell={'benchmark':'locomo','task':task,'category':cat,'category_name':driver.CATEGORY_NAMES[cat],'arm':arm,'n':n,'expected_n':n,'complete':True,
              'score_percent':scores[arm],'raw_mean':scores[arm]/100,'display':f'{scores[arm]:.2f}',
              'metric':'adversarial option accuracy (explicit decoding adaptation)' if cat==5 else 'official F1',
              'generation_cap':50,'source_and_pack_verified':True,'exact_generation_cache_verified':True,
              'official_rescores':n,'exact_source_token_pack_own_generation_cache_checks':n,'sources':sources[arm],'paired_contrasts':{}}
        if cat==5:
            assert all(row['score'] in (0,1) for row in rr);cell['correct']=int(sum(row['score'] for row in rr))
        for ref in refs:
            other=perarm[ref];assert [x['id'] for x in rr]==[x['id'] for x in other]
            diffs=[x['score']-y['score'] for x,y in zip(rr,other)]
            p={'direction':'new baseline relative to previously audited reference','paired_n':n,
               'new_arm_wins':sum(x>1e-12 for x in diffs),'ties':sum(abs(x)<=1e-12 for x in diffs),'new_arm_losses':sum(x< -1e-12 for x in diffs),
               'new_arm_minus_reference_pp':scores[arm]-scores[ref],'reference_score_percent':scores[ref],
               'same_source_tokens_and_ordered_pack':True,'reference_scorer_calls':0,'reference_cache_lookups':0,'significance_test_performed':False}
            assert p['new_arm_wins']+p['ties']+p['new_arm_losses']==n
            cell['paired_contrasts'][ref]=p;paired.append({'benchmark':'locomo','task':task,'new_arm':arm,'reference_arm':ref,**p})
        comparisons.append({'benchmark':'locomo','task':task,'n':n,'new_arm':arm,'v2_minus_new_arm_pp':scores['fix_all']-scores[arm],'j0_minus_new_arm_pp':scores['j0']-scores[arm]})
        cells.append(cell)
assert checks['official_rescores']==checks['new_exact_source_token_pack_own_generation_cache_checks']==3972
assert checks['reconstructed_source_token_ordered_packs']==1986
assert not torch.cuda.is_initialized()
report={'snapshot_timestamp_utc':snapshot['timestamp_utc'],'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'New complete pub and pub_sink LoCoMo only; 1986 distinct outputs per arm partitioned into five categories.',
        'writeback':cells,'comparisons':comparisons,'paired_counts':paired,'reference_cells':reference_cells,
        'checks':dict(checks,models_constructed=0,cuda_initialized=False,CUDA_VISIBLE_DEVICES='',torch_threads=2,torch_interop_threads=16,reference_scorer_calls=0,reference_cache_lookups=0),
        'source_input_keys_sha256':input_keys.hexdigest(),'ranges':{k:{'min':min(v),'max':max(v)} for k,v in ranges.items()},
        'protocol':{'shared_options':shared_options,'shared_protocol':shared_protocol,'generation_cap':50,
         'metrics':'Official answerable-category F1 functions; multi-hop uses official f1, other answerable categories f1_score, open-domain answer before semicolon. Category5 robust explicit option-decoding adaptation with seed42 mapping; do not pool categories.',
         'generation':'Greedy short-answer cap50, native first-token EOS suppression then natural EOS; Qwen chat template with thinking disabled, chronological full history plus captions, explicit complete query and no truncation/padding.',
         'pairing':'Every new source token sequence and selected ordered pack was reconstructed once per source and matched both new arms own SQLite prediction/n_tokens key; references use old audited scores with matching source sample and ordered pack, zero old scorer/cache calls.',
         'limits':'Encbank retrieval/chat/greedy adaptation, not unmodified official HF generation; no overall mixed-category mean, no significance or local timing/memory claim. Original official_qa_driver logs do not print the RULER native admission line: completed remote queue GPU/job/exit records are checked, but per-job GPU admission memory samples were not persisted.'}}
dest=B/'heartbeat_locomo_20260909_1032_new.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'writeback':[{k:c[k] for k in ('task','arm','n','score_percent')} for c in cells],'comparisons':comparisons,'checks':report['checks'],'output':str(dest)}),flush=True)
