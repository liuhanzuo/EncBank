"""One GPU, shared read-only weights, independent batch rows and split caches.

Equal lengths eliminate padding and position ambiguity. This is a finite fixed
workload experiment; it does not receive or alter live benchmark requests.
"""
from pathlib import Path
import gc,hashlib,json,os,platform,subprocess,time,traceback
from persistence import dump
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());R=H/'run_encbank'
def event(kind,**kw):
    with (R/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw))+'\n')
def main():
    H.resolve().relative_to(Path('/srv/encbank').resolve())
    event('imports_begin')
    from common import MODELS,load_model,load_state,torch
    from hybrid_reader import HybridReader
    from transformers.cache_utils import DynamicCache
    import transformers
    def stamp():
        torch.cuda.synchronize();return time.perf_counter()
    def thash(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def cache_info(*caches):
        seen=set();storage={};tensors=[]
        def walk(v,path):
            if id(v) in seen:return
            seen.add(id(v))
            if torch.is_tensor(v):
                s=v.untyped_storage();storage[(str(v.device),s.data_ptr())]=s.nbytes()
                tensors.append(dict(path=path,shape=list(v.shape),dtype=str(v.dtype),device=str(v.device)))
            elif isinstance(v,dict):
                for k,x in v.items():walk(x,path+'.'+str(k))
            elif isinstance(v,(tuple,list)):
                for i,x in enumerate(v):walk(x,path+'.'+str(i))
            elif type(v).__module__.startswith('transformers.cache_utils'):walk(vars(v),path)
        for i,c in enumerate(caches):walk(c,str(i))
        return dict(bytes=sum(storage.values()),tensors=tensors)
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(P['seed']);torch.backends.cuda.enable_cudnn_sdp(False)
    cfg=MODELS[1];assert cfg['j']==P['j'] and cfg['path']==P['model']
    admission=[]
    for n in range(4):
        free,total=torch.cuda.mem_get_info();assert free>=P['admission_free_gib']*2**30
        admission.append(dict(epoch=time.time(),free=free,total=total,
            gpus=subprocess.check_output(['nvidia-smi'],text=True),
            processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,etime,args'],text=True)))
        if n<3:time.sleep(15)
    torch.cuda.set_per_process_memory_fraction(P['device_cap_gib']*2**30/total)
    dump(R/'admission.json',dict(status='PASS',samples=admission))
    for name,digest in json.loads((H/'remote_source_manifest.json').read_text()).items():
        assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,name
    cp=Path(P['adapter_path']).resolve();cp.relative_to(Path('/srv/encbank').resolve())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']
    cold=time.monotonic();event('model_load_begin')
    model=load_model(cfg);reader=HybridReader(model,cfg['j']);reader.attach()
    ckpt=torch.load(cp,map_location='cpu',weights_only=False);assert ckpt['step']==4000
    load_state(reader,ckpt,cfg);del ckpt;model.requires_grad_(False)
    assert all(p.device.type=='cuda' for p in model.parameters())
    dev=torch.cuda.get_device_properties(0)
    dump(R/'worker_ready.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],hostname=platform.node(),
        gpu=dev.name,cc=[dev.major,dev.minor],uuid=str(getattr(dev,'uuid','unavailable')),
        torch=torch.__version__,transformers=transformers.__version__,cold_start_seconds=time.monotonic()-cold,
        model_weights_loaded_once=True,max_simultaneous_gpu_sequences=2,offload=False,
        plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((H/'inputs.json').read_bytes()).hexdigest(),adapter_sha256=P['adapter_sha256']))
    inputs=json.loads((H/'inputs.json').read_text());banks={};hashes={}
    with torch.inference_mode():
        for row in inputs:
            started=stamp();bank=[reader.write(c) for c in row['chunks']];ended=stamp()
            sid=row['session'];banks[sid]=bank;hashes[sid]=[thash(t) for t in bank]
            event('bank_ready',session=sid,seconds=ended-started,tokens=sum(map(len,row['chunks'])),
                bytes=sum(t.numel()*t.element_size() for t in bank))
        del bank
        def immutable():
            return all([thash(t) for t in banks[k]]==hashes[k] for k in banks)
        # Preassemble every measured shape before timings. Source banks remain
        # separate and immutable; batching copies rows rather than aliasing caches.
        packs={}
        for ids in [('A',),('B',),('A','B'),('B','A'),('A','C')]:
            rows=[next(r for r in inputs if r['session']==s) for s in ids]
            packs[ids]=(torch.cat([torch.cat(banks[s],dim=1) for s in ids],dim=0),
                torch.tensor([r['query'] for r in rows],device='cuda',dtype=torch.long),
                torch.tensor([r['replay'] for r in rows],device='cuda',dtype=torch.long))
        def run(ids,positions,capture=False):
            fixed,query,replay=packs[tuple(ids)]
            lower=DynamicCache(config=reader.config);upper=DynamicCache(config=reader.config)
            start=stamp()
            qh=reader.layers(reader.core.embed_tokens(query),0,reader.j,cache=lower)
            hidden=reader.layers(torch.cat([fixed,qh],dim=1),reader.j,reader.L,cache=upper)
            logits=reader.logits(hidden);del hidden,qh
            first=stamp();captured=[logits.clone()] if capture else []
            finite=torch.isfinite(logits).all()
            for step in range(positions-1):
                hidden=reader.core.embed_tokens(replay[:,step:step+1])
                hidden=reader.layers(hidden,0,reader.j,cache=lower,offset=query.shape[1]+step)
                hidden=reader.layers(hidden,reader.j,reader.L,cache=upper,offset=fixed.shape[1]+query.shape[1]+step)
                logits=reader.logits(hidden);del hidden
                finite=finite & torch.isfinite(logits).all()
                if capture:captured.append(logits.clone())
            end=stamp();assert bool(finite),'Nonfinite logits; no sanitization/retry'
            info=cache_info(lower,upper)
            result=dict(sessions=list(ids),positions_per_session=positions,forced_decode_forwards=positions-1,
                prefill_to_initial_logits_seconds=first-start,decode_seconds=end-first,model_seconds=end-start,
                aggregate_decode_tokens_per_second=len(ids)*(positions-1)/(end-first),cache_bytes=info['bytes'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            values=torch.cat(captured,dim=1) if capture else None
            del captured,logits,lower,upper,finite
            return result,values,info
        def compare(reference,test):
            a=reference.float();b=test.float();delta=(a-b).abs()
            tv=.5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)
            return dict(max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),
                allclose_atol_rtol_002=bool(torch.allclose(a,b,atol=.002,rtol=.002)),
                top1_matches=int((a.argmax(-1)==b.argmax(-1)).sum()),positions=a.shape[0]*a.shape[1],
                max_probability_total_variation=float(tv.max()),mean_probability_total_variation=float(tv.mean()),
                per_position_max_abs=delta.amax(-1).tolist(),per_position_total_variation=tv.tolist(),
                reference_top1=a.argmax(-1).tolist(),test_top1=b.argmax(-1).tolist())
        event('validation_start')
        _,a,_=run(['A'],P['validation_positions'],True)
        _,b,_=run(['B'],P['validation_positions'],True)
        _,ab,info=run(['A','B'],P['validation_positions'],True)
        serial_compare=compare(torch.cat([a,b],dim=0),ab);del a,b
        _,ba,_=run(['B','A'],P['validation_positions'],True)
        permutation=compare(ab,ba.flip(0));del ba
        _,ac,_=run(['A','C'],P['validation_positions'],True)
        isolation=compare(ab[:1],ac[:1]);del ac,ab
        invariance=immutable()
        passed=permutation['allclose_atol_rtol_002'] and isolation['allclose_atol_rtol_002'] and invariance
        validation=dict(status='PASS' if passed else 'FAIL',finite=True,H_immutable=invariance,
            serial_vs_batch2=serial_compare,row_permutation=permutation,replace_other_session=isolation,
            batch_cache_structure=info,
            boundary='Only equal-length fixed replay. Numerical difference reported; no sampled-output or unequal-length service equivalence claim.')
        dump(R/'validation.json',validation)
        assert passed,'Session isolation/permutation/H invariance failed; no timing or deployment'
        event('validation_complete',status='PASS',serial_batch_max_abs=serial_compare['max_abs'])
        del packs[('B','A')],packs[('A','C')],banks['C'],hashes['C']
        dump(R/'timing_resident_inputs.json',dict(
            persistent_H_bank_bytes=sum(t.numel()*t.element_size() for bank in banks.values() for t in bank),
            preassembled_pack_and_token_bytes=sum(t.numel()*t.element_size() for pack in packs.values() for t in pack),
            sessions=['A','B'],validation_C_released=True,
            note='Common resident inputs include both serial and batch layouts; phase peaks include these buffers and the model.'))
        for ids in [['A'],['B'],['A','B']]:run(ids,P['warmup_positions'])
        gc.collect();torch.cuda.empty_cache();stamp()
        trials=[]
        for repeat,order in enumerate(P['order']):
            for mode in order:
                gc.collect();torch.cuda.empty_cache();stamp();torch.cuda.reset_peak_memory_stats()
                event('trial_start',repeat=repeat,mode=mode)
                measurements=[];begin=stamp()
                for ids in ([['A'],['B']] if mode=='serial' else [['A','B']]):
                    result,_,_=run(ids,P['output_positions']);measurements.append(result)
                end=stamp();assert immutable()
                record=dict(repeat=repeat,mode=mode,measurements=measurements,pair_wall_seconds=end-begin,
                    sum_model_seconds=sum(m['model_seconds'] for m in measurements),
                    pair_decode_seconds=sum(m['decode_seconds'] for m in measurements),
                    aggregate_decode_tokens_per_second=2*(P['output_positions']-1)/sum(m['decode_seconds'] for m in measurements),
                    aggregate_output_positions_per_second=2*P['output_positions']/(end-begin),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),H_immutable=True)
                dump(R/f'trial_{repeat}_{mode}.json',record);trials.append(record)
                event('trial_complete',repeat=repeat,mode=mode,seconds=end-begin)
        dump(R/'result.json',dict(status='complete',trials=trials,validation=validation,
            limitation=P['purpose'],forced_replay=True,sampled_quality=False))
        del packs,banks;gc.collect();torch.cuda.empty_cache();stamp()
        dump(R/'worker_complete.json',dict(status='complete',epoch=time.time(),trials=len(trials),
            allocated_after_state_release=torch.cuda.memory_allocated(),reserved_after_state_release=torch.cuda.memory_reserved(),
            model_remains_loaded_until_process_exit=True))
try:main()
except BaseException as exc:
    dump(R/'worker_failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),epoch=time.time()))
    raise
