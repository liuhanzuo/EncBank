"""Remote stdlib-only reaggregation of copied local-5090 cost records."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import statistics
import zipfile

root = Path.cwd()
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))
def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8), (a,b)
def finite(obj):
    if isinstance(obj,float): assert math.isfinite(obj)
    elif isinstance(obj,dict):
        for value in obj.values(): finite(value)
    elif isinstance(obj,list):
        for value in obj: finite(value)
def tokens(path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if '/data/' in name or name.endswith('/data.pkl'):
                digest.update(name.encode()); digest.update(z.read(name))
    return digest.hexdigest()
meta = load(root/'results/protocol/cacheblend_cost_0005/bundle_metadata.json')
store_stats = {key.replace('\\','/'):v for key,v in meta['actual_store_file_bytes'].items()}
previous = load(root/'heartbeat_cost_20260908_2202.json')
previous_rows = {(s['arm'],s['context_tokens'],s['tier'],s['G'],s['Q']):s for s in previous['rows']}
state = load(root/'results/local/bootstrap_cacheblend_serving/status.json')
common_config_keys = ['model','context_file','j','chunk_size','topk','dtype','seed','source_tokens',
                      'query_counts','generation_lengths','tiers','fixed_generation_length']
hardware_keys = ['device','hostname','platform','python','torch','transformers','device_uuid',
    'device_name','torch_cpu_threads','torch_interop_threads','omp_num_threads','mkl_num_threads',
    'tokenizers_parallelism','torch_memory_cap_bytes']
configs, hardware, workloads, token_digests, packs, rows, raw, jobs = {},{}, {},{}, {},[],{},[]
checked_queries = 0
for job in meta['jobs']:
    arm,n = job['arm'],job['length']
    folder = root/job['relative_path']
    c = load(folder/'config.json'); complete=load(folder/'COMPLETED.json'); summary=load(folder/'summary.json')
    assert complete['status']=='complete' and complete['cells']==len(summary)==12
    assert not (folder/'INVALIDATED.json').exists()
    config_subset = {k:c[k] for k in common_config_keys}
    assert config_subset == configs.setdefault(n,config_subset)
    h = {k:c['hardware'][k] for k in hardware_keys}
    assert h == hardware.setdefault(n,h)
    assert h['device_name']=='NVIDIA GeForce RTX 5090' and h['platform']=='Windows'
    assert (h['torch_cpu_threads'],h['torch_interop_threads'])==(2,16)
    assert h['omp_num_threads']==h['mkl_num_threads']=='2' and h['tokenizers_parallelism']=='false'
    assert complete['hardware']==c['hardware']
    workload=load(folder/f'workload_{n}.json')
    assert workload == workloads.setdefault(n,workload)
    assert len(workload['questions'])==100 and not workload['source_repeated']
    store=folder/f'store_{n}_{arm}'
    token_digest=tokens(store/'tokens.pt')
    assert token_digest==token_digests.setdefault(n,token_digest)
    actual_bytes=sum(store_stats[str(store.relative_to(root))].values())
    storemeta=load(store/'store.json')
    is_cb=arm=='cacheblend16'
    if is_cb:
        a=c['hardware']['gpu_admission']
        assert a['initial_used_gib']<5 and a['recheck_used_gib']<5
        assert a['effective_idle_slack_gib']==5 and a['comparison']=='strictly_less_than'
        assert a['source']=='nvidia-smi MiB / 1024' and not a['other_python_compute_processes']
        assert c['adapter'] is None
        assert storemeta['signature']['bootstrap_full_layers']==2
        assert storemeta['signature']['recompute_ratio']==.16
        assert storemeta['signature']['chunk_write_sink'] is False
    files=list(folder.glob('queries_*.jsonl'))
    assert len(files)==4
    for path in files:
        tier=path.stem.split('_')[-2]; g=int(path.stem.split('_')[-1][1:])
        qs=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert [q['id'] for q in qs]==list(range(100))
        raw[arm,n,tier,g]=qs
        for q in qs:
            assert q['hardware']==c['hardware']
            assert q['generated_tokens']==len(q['generated_ids'])==g and q['decode_steps']==g-1
            assert q['capture_calls']==0 and len(set(q['selected_indices']))==12
            pack=(q['selected_indices'],q['query_tokens'],q['read_tokens'])
            assert pack==packs.setdefault((n,q['id']),pack)
            if is_cb:
                finite(q); near(q['total_s'],q['ttft_s']+q['decode_s'])
                assert all(v>=0 for k,v in q.items() if k.endswith('_s') and isinstance(v,(int,float)))
                assert q['document_capture_calls']==q['query_capture_calls']==0
                assert q['timing_eligible'] and not q['instrumentation_diagnostic']
                assert q['online_kv_inventory'] is None and 'fresh_reference' not in q
                cb=q['cacheblend']; context_len=q['read_tokens']-1-q['query_tokens']
                assert cb['cacheblend_variant']=='qwen3_layer1_v_full_two_layer_bootstrap'
                assert cb['recompute_ratio']==.16 and cb['bootstrap_full_layers']==2
                assert cb['n_context_tokens']==context_len
                assert cb['n_recompute_ctx']==math.floor(.16*context_len)
                selected=cb['selected_positions']
                assert sorted(set(selected))==selected
                assert len(selected)==1+q['query_tokens']+cb['n_recompute_ctx']
                assert selected[0]==0 and set(range(1+context_len,q['read_tokens'])).issubset(selected)
                assert cb['cacheblend_kv_bytes_per_tok']==147456
                checked_queries+=1
    for g in (16,128):
        assert [q['generated_ids'] for q in raw[arm,n,'cpu',g]]==[q['generated_ids'] for q in raw[arm,n,'disk',g]]
    cells=set()
    for s in summary:
        finite(s)
        cell=(s['tier'],s['G'],s['Q']); assert cell not in cells; cells.add(cell)
        assert s['write']['serialized_bytes']==actual_bytes
        assert s['source_preprocessing_charged'] is False
        assert s['hardware']==c['hardware']
        if is_cb:
            qs=raw[arm,n,s['tier'],s['G']][:s['Q']]
            for metric,total in s['query_totals'].items(): near(total,sum(q[metric] for q in qs))
            near(s['end_to_end_total_s'],s['write']['write_total_s']+s['startup']['startup_load_s']+s['query_totals']['total_s'])
            near(s['mean_ttft_s'],s['query_totals']['ttft_s']/s['Q'])
            near(s['decode_tokens_per_s'],s['query_totals']['decode_steps']/s['query_totals']['decode_s'])
            assert s['incremental_peak_bytes']==max(q['incremental_peak_bytes'] for q in qs)
            assert s['peak_allocated_bytes']==max(s['write']['write_peak_allocated_bytes'],max(q['peak_allocated_bytes'] for q in qs))
            assert s['write']['capture_calls']==n//512+1
            assert s['write']['payload_tensor_bytes']==(n+1)*147456
        compact={k:s[k] for k in ['arm','context_tokens','tier','G','Q','end_to_end_total_s','mean_ttft_s','decode_tokens_per_s','peak_allocated_bytes','incremental_peak_bytes','write','startup','query_totals']}
        if not is_cb:
            assert compact==previous_rows[arm,n,s['tier'],s['G'],s['Q']]
        rows.append(compact)
    assert cells=={(t,g,q) for t in ('cpu','disk') for g in (16,128) for q in (1,10,100)}
    jobs.append(job|{'actual_store_bytes':actual_bytes,'hardware':c['hardware'],
                     'token_tensor_identity':token_digest,'store_signature':storemeta['signature']})

smoke=root/'results/local/cacheblend_serving_reuse/smoke/cacheblend16_1024/attempts/0001'
proof=load(smoke/'FRESH_REFERENCE_CHECK.json')
assert proof['complete'] and proof['verified_queries']==proof['extra_reference_sequences']==len(proof['checks'])==8
assert all(v['equal_ids'] and v['equal_selected_positions'] and v['reuse_capture_calls']==0 and v['persistent_cpu_versions_and_disk_metadata_unchanged'] for v in proof['checks'])
idx={(s['arm'],s['context_tokens'],s['tier'],s['G'],s['Q']):s for s in rows}
comparisons=[]
for cb in (s for s in rows if s['arm']=='cacheblend16'):
    for other in ('pub','pub_sink','pub_lora','j0','fix_all'):
        ref=idx[other,cb['context_tokens'],cb['tier'],cb['G'],cb['Q']]
        comparisons.append({'context_tokens':cb['context_tokens'],'tier':cb['tier'],'G':cb['G'],'Q':cb['Q'],
            'reference_arm':other,'cacheblend_cumulative_s':cb['end_to_end_total_s'],'reference_cumulative_s':ref['end_to_end_total_s'],
            'cacheblend_over_reference_cumulative':cb['end_to_end_total_s']/ref['end_to_end_total_s'],
            'cacheblend_mean_ttft_s':cb['mean_ttft_s'],'reference_mean_ttft_s':ref['mean_ttft_s'],
            'cacheblend_over_reference_ttft':cb['mean_ttft_s']/ref['mean_ttft_s']})
distributions=[]
for (arm,n,tier,g),qs in raw.items():
    if arm not in ('cacheblend16','fix_all','j0'): continue
    vals=sorted(q['total_s'] for q in qs)
    distributions.append({'arm':arm,'context_tokens':n,'tier':tier,'G':g,'median':statistics.median(vals),
        'p95_nearest_rank':vals[94],'maximum':vals[-1],'mean':statistics.mean(vals)})
result={'audited_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'success':True,'aggregation_host':'longjing-1 CPU only, stdlib; CUDA not imported',
    'bundle_created_at':meta['created_at'],'checked_cacheblend_cells':sum(s['arm']=='cacheblend16' for s in rows),
    'checked_cacheblend_queries':checked_queries,'comparison_query_identity_records':len(raw)*100,
    'completed_lengths':sorted(configs),'incomplete_lengths':[n for n in (32768,131072) if n not in configs],
    'active_job_at_bundle':state.get('active_job'),'source_status_updated_at':state['updated_at'],
    'jobs':jobs,'rows':rows,'comparisons':comparisons,'untrimmed_distributions':distributions,
    'smoke_proof':{'verified_queries':8,'extra_fresh_reference_sequences':8,'same_gpu_8b_bf16_exact_ids_and_positions':True},
    'boundary':previous['boundary'],
    'limitations':['One sequential trace, ordinary OS page cache without eviction; no cold disk or repeated-run uncertainty claim.',
    'Unscored fixed-length excerpt cost workload; do not combine with natural-EOS QA quality as one Pareto point.',
    'CacheBlend-style Qwen3 port, two full bootstrap layers followed by 16% selective recomputation; not the released serving implementation.',
    'Actual KV inventory is null in timed records; allocated peak is not online KV bytes or nvidia-smi resident memory.',
    'Existing 32k V2 long tails are retained; no rerun or trimming.',
    'Earlier V2 and CoMem 32k have legacy gate logs only; do not claim exact structured dual-admission values for them.']}
result['writeback_summary']=[]
for n in sorted(configs):
    subset=[s for s in rows if s['arm']=='cacheblend16' and s['context_tokens']==n]
    w=subset[0]['write']
    result['writeback_summary'].append({'context_tokens':n,'payload_bytes':w['payload_tensor_bytes'],
        'payload_GiB':w['payload_tensor_bytes']/2**30,'serialized_bytes':w['serialized_bytes'],
        'serialized_GiB':w['serialized_bytes']/2**30,'write_total_s':w['write_total_s'],
        'max_query_incremental_peak_bytes':max(s['incremental_peak_bytes'] for s in subset),
        'max_query_incremental_peak_GiB':max(s['incremental_peak_bytes'] for s in subset)/2**30,
        'max_allocated_peak_bytes':max(s['peak_allocated_bytes'] for s in subset),
        'max_allocated_peak_GiB':max(s['peak_allocated_bytes'] for s in subset)/2**30,
        'startup_cpu_G16_s':idx['cacheblend16',n,'cpu',16,100]['startup']['startup_load_s'],
        'startup_cpu_G128_s':idx['cacheblend16',n,'cpu',128,100]['startup']['startup_load_s'],
        'cells':subset})
out=root/'heartbeat_cacheblend_cost_20260909_0005.json'
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['success','checked_cacheblend_cells','checked_cacheblend_queries','completed_lengths','incomplete_lengths']},indent=2))
for s in rows:
    if s['arm']=='cacheblend16':
        print(s['context_tokens'],s['tier'],s['G'],s['Q'],'cumulative',round(s['end_to_end_total_s'],3),'TTFT',round(s['mean_ttft_s'],3),'peakGiB',round(s['peak_allocated_bytes']/2**30,3),'storebytes',s['write']['serialized_bytes'],'write',round(s['write']['write_total_s'],3))
