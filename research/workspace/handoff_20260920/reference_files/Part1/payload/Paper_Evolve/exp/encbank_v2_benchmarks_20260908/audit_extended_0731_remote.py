"""New extended baseline accuracy cells only; remote CPU, no models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib,random,zlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
sys.path[:0]=[str(B),str(R/'workspace/Encbank'),str(R/'workspace/exp')]
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer
from eval import longeval as le
from eval import ruler as ru
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-9,(a,b)
def native(run):
    m=read(run/'COMPLETED.json');assert m['status']=='completed'
    return Path(m['output_dir']),m,read(run/'run_config.json')
def key_for(ids,options):return hashlib.sha256(json.dumps({'tokens':ids.tolist(),'generation':options},sort_keys=True,separators=(',',':')).encode()).hexdigest()
prior={'writeback':read(B/'heartbeat_extended_20260909_0129.json')['writeback']+read(B/'heartbeat_extended_20260909_0229.json')['writeback']+read(B/'heartbeat_extended_20260909_0329.json')['writeback']+read(B/'heartbeat_extended_20260909_0429.json')['writeback']+read(B/'heartbeat_extended_20260909_0531.json')['writeback']+read(B/'heartbeat_extended_20260909_0631.json')['writeback']}
known={(c['benchmark'],c['task'],c.get('length'),c['arm']) for c in prior['writeback']}
snapshot=read(B/'heartbeat_extended_20260909_0731_snapshot.json')
state=snapshot['state']
eligible_native=set(snapshot['eligible_native_markers'])
eligible_ruler=set(snapshot['eligible_ruler_files'])
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU; CUDA hidden; no model construction or GPU tasks',
   'writeback':[],'comparisons':[],'pending':[],'checks':{},'prior_reference':['heartbeat_extended_20260909_0129.json','heartbeat_extended_20260909_0229.json','heartbeat_extended_20260909_0329.json','heartbeat_extended_20260909_0429.json','heartbeat_extended_20260909_0531.json','heartbeat_extended_20260909_0631.json']}
d['snapshot_timestamp_utc']=snapshot['timestamp_utc']
d['snapshot_file']=str(B/'heartbeat_extended_20260909_0731_snapshot.json')
new_ruler=[];ruler_checks=0
for arm in ('pub','pub_sink','cbos'):
    for p in sorted((O/'ruler'/arm).glob('*.json')):
        if str(p) not in eligible_ruler:continue
        obj=read(p);args=obj['args'];task=args['tasks'];length=args['lengths'];jobid=f'ruler__{arm}__{task}_{length}'
        job=state['jobs'].get(jobid,{})
        if job.get('status')!='completed' or job.get('exit_code')!=0:
            d['pending'].append({'benchmark':'ruler','task':task,'length':length,'arm':arm,'complete':False,'n_observed':len(obj['rows']),'score_percent':None,'status':job.get('status')});continue
        if ('ruler',task,length,arm) in known:continue
        assert obj['arms']==[arm] and obj['lora'] is None and args['adapter']==args['adapter_pt']==''
        assert args['model']==str(R/'models/Qwen3-8B') and args['j']==12 and args['n']==50 and args['topk']==12 and args['seed']==42 and args['selector']=='auto' and args['check']==0
        assert args['max_new_tokens']==48 and args['iter_hop_topk']==4
        rr=obj['rows'];assert len(rr)==50 and [r['i'] for r in rr]==list(range(50))
        log=Path(job['log']).read_text(errors='replace');assert 'PYTHONHASHSEED=0' in log and '[remote admission]' in log
        for r in rr:
            assert r['task']==task and r['length']==length and r[arm+'_out']!='[OOM]'
            near(ru._string_match_all_one(r[arm+'_out'],r['answers']),r[arm+'_recall']);ruler_checks+=1
        score=100*sum(r[arm+'_recall'] for r in rr)/50;near(score,obj['summary'][f'{task}/{length}'][arm])
        ref_scores={}
        for refarm in ('fix_all','j0'):
            ref=read(O/'ruler'/refarm/f'{task}_{length}.json')
            assert {k:v for k,v in args.items() if k not in ('arms','out')}=={k:v for k,v in ref['args'].items() if k not in ('arms','out')}
            assert len(ref['rows'])==50
            for a,b in zip(rr,ref['rows']):assert all(a[k]==b[k] for k in ('task','length','i','answers','n_tokens'))
            ref_scores[refarm]=ref['summary'][f'{task}/{length}'][refarm]
        cell={'benchmark':'ruler','task':task,'length':length,'arm':arm,'complete':True,'n':50,'expected_n':50,'score_percent':score,'display':f'{score:.2f}',
            'metric':'Encbank RULER adaptation: answer substring recall','source':str(p),'raw_rescores':50,'metadata_paired_with_v2_j0':True,
            'paired_reference_scores_percent':ref_scores,'exact_input_tokens_persisted':False,'runtime_selected_pack_persisted':False,'generation_cache_available':False}
        d['writeback'].append(cell);new_ruler.append(cell)
        print('RULER verified',arm,task,length,score,flush=True)

# Read only complete shards for coverage; never turn partial task means into final scores.
for arm in ('pub','pub_sink','cbos'):
    for task,n in (('longbook_qa_eng',351),('longbook_choice_eng',229)):
        if ('infinitebench',task,None,arm) in known:continue
        seen={};shards=[]
        for shard in range(4):
            run=O/'infinitebench'/arm/f'{task}_s{shard}of4'
            if str(run/'COMPLETED.json') not in eligible_native:continue
            out,m,c=native(run);rr=rows(out/f'{task}_{shard}.jsonl');expected=list(range(shard,n,4))
            assert [r['index'] for r in rr]==expected and m['new_generations']+m['reused_generations']==len(expected)
            for r in rr:assert r['index'] not in seen and r['status']=='ok';seen[r['index']]=r['id']
            shards.append(shard)
        assert len(set(seen.values()))==len(seen)
        # A newly full task would require a separate full source/token/pack/scorer audit.
        d['pending'].append({'benchmark':'infinitebench','task':task,'arm':arm,'completed_shards':shards,'n':len(seen),'expected_n':n,
            'coverage_complete':len(seen)==n,'writeback_eligible':False,'score_percent':None})
    seen={};shards=[];counts=Counter()
    for shard in range(10):
        run=O/'locomo'/arm/f'all_s{shard}of10'
        if str(run/'COMPLETED.json') not in eligible_native:continue
        m=read(run/'COMPLETED.json');c=read(run/'run_config.json');rr=rows(run/'predictions.jsonl')
        assert m['status']=='completed' and m['n']==len(rr)==len(c['expected'])
        assert [{k:r[k] for k in ('index','id','task')} for r in rr]==c['expected']
        assert [r['index'] for r in rr]==list(range(shard,1986,10))
        for r in rr:
            assert r['id'] not in seen and r['status']=='ok';seen[r['id']]=r['index'];counts[r['category']]+=1
        shards.append(shard)
    expected_counts={1:282,2:321,3:96,4:841,5:446}
    d['pending'].append({'benchmark':'locomo','task':'all','arm':arm,'completed_shards':shards,'n':len(seen),'expected_n':1986,'coverage_complete':len(seen)==1986,
        'categories':[{'category':cat,'n':counts[cat],'expected_n':n,'coverage_complete':counts[cat]==n,'score_percent':None} for cat,n in expected_counts.items()],
        'writeback_eligible':False,'score_percent':None})
d['checks']={'new_ruler_raw_rescores':ruler_checks,'old_outputs_rescored':0,'longeval_rescores':0,'models_constructed':0,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'cuda_initialized':torch.cuda.is_initialized(),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'snapshot_fixed':True,'partial_scores_reported':False}
assert not torch.cuda.is_initialized()
d['finished_at_utc']=datetime.now(timezone.utc).isoformat()
p=B/'heartbeat_extended_20260909_0731_ruler.json';assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d),flush=True)
