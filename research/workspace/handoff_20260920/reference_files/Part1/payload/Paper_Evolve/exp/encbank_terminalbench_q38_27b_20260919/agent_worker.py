"""Shared native Encbank weights; task-local H banks and transient split caches."""
from pathlib import Path
import gc,hashlib,json,os,platform,subprocess,time,traceback
from persistence import dump
from session import Session,digest,request_seed
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());R=H;B=R/'mailbox'
def event(kind,**kw):
    with (R/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw))+'\n')
def main():
    assert H.name.startswith('qcm-q38-tb-liuhanzuo-') and H.parent==Path('/tmp')
    B.mkdir(exist_ok=True);(H/'incoming').mkdir(exist_ok=True)
    event('imports_begin')
    from common import MODELS,load_model,load_state,tokenizer,torch
    from hybrid_reader import HybridReader
    from memory_selectors import iter_bm25_indices
    from transformers.cache_utils import DynamicCache
    import transformers
    def stamp():torch.cuda.synchronize();return time.perf_counter()
    def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def cache_bytes(c):
        seen=set();storage={}
        def walk(v):
            if id(v) in seen:return
            seen.add(id(v))
            if torch.is_tensor(v):
                s=v.untyped_storage();storage[(str(v.device),s.data_ptr())]=s.nbytes()
            elif isinstance(v,dict):
                for x in v.values():walk(x)
            elif isinstance(v,(tuple,list)):
                for x in v:walk(x)
            elif type(v).__module__.startswith('transformers.cache_utils'):walk(vars(v))
        walk(c);return sum(storage.values())
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(P['seed']);torch.backends.cuda.enable_cudnn_sdp(False)
    cfg=MODELS[1];assert cfg['j']==P['j'] and cfg['path']==P['model']
    admission=[]
    for n in range(4):
        free,total=torch.cuda.mem_get_info();assert free>=P['admission_free_gib']*2**30
        admission.append(dict(epoch=time.time(),free=free,total=total,gpus=subprocess.check_output(['nvidia-smi'],text=True),
            processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,etime,args'],text=True)))
        if n<3:time.sleep(15)
    torch.cuda.set_per_process_memory_fraction(P['device_cap_gib']*2**30/total)
    dump(R/'admission.json',dict(status='PASS',samples=admission))
    cold=time.monotonic();tok=tokenizer(cfg);event('model_load_begin')
    # Hash before allocation/copy; every formal launch uses the frozen adapter.
    cp=Path(P['adapter_path']).resolve();cp.relative_to(Path('/srv/encbank').resolve())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']
    model=load_model(cfg);reader=HybridReader(model,cfg['j']);reader.attach()
    ckpt=torch.load(cp,map_location='cpu',weights_only=False);assert ckpt['step']==4000
    load_state(reader,ckpt,cfg);del ckpt;model.requires_grad_(False)
    assert all(p.device.type=='cuda' for p in model.parameters())
    stop=model.generation_config.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop or tok.eos_token_id])
    dev=torch.cuda.get_device_properties(0)
    dump(R/'environment.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],hostname=platform.node(),gpu=dev.name,
        cc=[dev.major,dev.minor],uuid=str(getattr(dev,'uuid','unavailable')),torch=torch.__version__,transformers=transformers.__version__,
        cold_start_seconds=time.monotonic()-cold,model_weights_loaded_once=True,max_live_sessions=P['max_live_sessions'],
        max_simultaneous_gpu_generations=P['decode_batch_size'],offload=False,plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
        sdpa_cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled(),serialization=P['serialization'],adapter_sha256=P['adapter_sha256']))
    from selection import attention_scores
    with torch.inference_mode():
        fixture=tok.encode('Inspect prior logs and locate the failing dependency. ',add_special_tokens=False)
        fixture=(fixture*100)[:512]
        static=[reader.write(fixture[:1])];mem=[reader.write(fixture),reader.write(list(reversed(fixture)))]
        query=reader.write(fixture[:128])
        _,check=attention_scores(reader,static,mem,query,32,verify=True)
        dump(R/'probe_correctness.json',check)
        del static,mem,query
    dump(R/'ready.json',dict(ready=True,probe_check_passed=True))
    from service_loop import serve
    serve(P,R,B,model,reader,tok,stop,event,torch,DynamicCache,iter_bm25_indices)
if __name__=='__main__':
    try:main()
    except BaseException:
        dump(R/'failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
