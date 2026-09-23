import argparse,ctypes,datetime,gc,hashlib,json,os,platform,random,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from engine import Engine,METHODS,BUDGET,sync
from queue_metrics import closed_loop,summarize,percentile

MODEL='/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B'
ADAPTER='/srv/encbank/encbank_infra_recheck_20260912/adapter'
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)
def rss():
    return int(next(l for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')).split()[1])*1024

class Nvml:
    class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self,uuid):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(self.handle))==0
        self.stop_event=threading.Event();self.samples=[]
    def sample(self):
        value=self.Memory();assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(value))==0
        self.samples.append(dict(at=time.perf_counter(),used_bytes=value.used,total_bytes=value.total))
    def loop(self):
        while not self.stop_event.wait(.1):self.sample()
    def start(self):self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join();self.sample();return max(x['used_bytes'] for x in self.samples)

def profiles(docs):
    diverse=[(di,qi) for qi in range(32) for di in range(3)];hot=[];seen=set()
    for di,qi in diverse:
        key=(di,tuple(docs[di]['queries'][qi]['selected']))
        if key not in seen:seen.add(key);hot.append((di,qi))
        if len(hot)==8:break
    rng=random.Random(92026);rng.shuffle(hot);rng.shuffle(diverse)
    return dict(hot=hot,diverse=diverse)

@torch.inference_mode()
def validate(model,tok,docs):
    result={};picks=[(0,0),(1,1),(2,2)];references={}
    for method in ['raw','h_gpu']:
        engine=Engine(model,tok,docs,method,3)
        first,end,data=engine.infer(picks,count=8,capture=True)
        references['raw' if method=='raw' else 'hidden']=data
        if method=='h_gpu':
            scalar=[]
            for row,pick in enumerate(picks):
                one=engine.infer([pick],count=8,capture=True,forced_tokens=[data['generated_ids'][row]])[2]
                scalar.append(torch.stack(one['logits_trace']))
            actual=torch.cat(scalar,dim=1);expected=torch.stack(data['logits_trace']);difference=actual-expected
            result['uncached_hidden_batch_variation']=dict(max_abs=float(difference.abs().max()),max_rms=float(difference.square().mean(-1).sqrt().max()),argmax_changes=int(actual.argmax(-1).ne(expected.argmax(-1)).sum()))
            dump(ROOT/'uncached_batch_variation.json',result['uncached_hidden_batch_variation'])
            del scalar,actual,expected,difference
        if method=='raw':
            ids=torch.tensor([[tok.bos_token_id]+[t for i in sel for t in docs[di]['source'][i*512:(i+1)*512]]+docs[di]['queries'][qi]['query'] for (di,qi),sel in zip(picks,data['selected'])],device='cuda')
            actual=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
            diff=actual-data['first_logits'];assert float(diff.abs().max())<1.0 and float(diff.square().mean().sqrt())<.15
            assert actual.argmax(-1).eq(data['first_logits'].argmax(-1)).all()
            result['raw_stock_equal']=dict(max_abs=float(diff.abs().max()),top1_equal=True)
            del actual,ids,diff
        del engine;gc.collect();torch.cuda.empty_cache()
    for method in ['h_cpu','raw_kv','hybrid']:
        engine=Engine(model,tok,docs,method,3);ref=references['raw' if method=='raw_kv' else 'hidden'];checks=[]
        # Different prefix lengths inside one batch: request0 fully primed;
        # request1 primes a prefix differing only in its last document block;
        # request2 is cold except possibly the common sink.
        if engine.store:
            engine.infer([picks[0]],count=1)
            alt=list(docs[1]['queries'][1]['selected']);alt[-1]=next(i for i in range(64) if i not in alt)
            engine.infer([picks[1]],count=1,forced_selected=[alt])
        for repeat in range(2):
            data=engine.infer(picks,count=8,capture=True)[2];delta=data['first_logits']-ref['first_logits']
            check=dict(repeat=repeat,max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                       top1_equal=bool(data['first_logits'].argmax(-1).eq(ref['first_logits'].argmax(-1)).all()),
                       generated_equal=data['generated_ids']==ref['generated_ids'],matched_positions=data['matched_positions'])
            assert check['max_abs']<1.0 and check['rms']<.15 and check['top1_equal'],(method,check)
            if not check['generated_equal']:
                # Diagnose the same token history before accepting numerical parity.
                forced=engine.infer(picks,count=8,capture=True,forced_tokens=ref['generated_ids'])[2]
                actual=torch.stack(forced['logits_trace']);expected=torch.stack(ref['logits_trace']);delta=actual-expected
                changes=actual.argmax(-1).ne(expected.argmax(-1));top2=expected.topk(2,dim=-1).values;margins=top2[:,:,0]-top2[:,:,1]
                diagnostic=dict(forced_max_abs=float(delta.abs().max()),forced_max_rms=float(delta.square().mean(-1).sqrt().max()),
                    argmax_changes=int(changes.sum()),changed_reference_margins=margins[changes].tolist(),
                    free_generated_actual=data['generated_ids'],free_generated_reference=ref['generated_ids'])
                check['diagnostic']=diagnostic;dump(ROOT/('numerics_'+method+'_'+str(repeat)+'.json'),check)
                # A nonidentical free-running sequence is explicitly retained. Only
                # small-margin BF16 differences with bounded same-history logits pass.
                assert diagnostic['forced_max_abs']<1.0 and diagnostic['forced_max_rms']<.15,diagnostic
                assert diagnostic['changed_reference_margins'] and max(diagnostic['changed_reference_margins'])<=.125,diagnostic
            checks.append(check)
        result[method]=checks;del engine;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'correctness.json',dict(passed=True,checks=result,criterion='bounded BF16 same-history logits; free-generation mismatches recorded, not bitwise equality'))

def run_point(model,tok,docs,profile,templates,method,concurrency,uuid,requests):
    out=ROOT/'results'/(profile+'_'+method+'_c'+str(concurrency));out.mkdir(parents=True,exist_ok=False)
    dump(ROOT/'status.json',dict(phase='RUNNING',profile=profile,method=method,concurrency=concurrency,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    engine=None;telemetry=None;phase='prepare'
    try:
        gc.collect();torch.cuda.empty_cache();engine=Engine(model,tok,docs,method,concurrency)
        offline=dict(write_s=engine.write_s,index_build_s=engine.index_s,H_bytes=engine.hbytes,GPU_H_bytes=engine.persistent_gpu_h,cpu_rss_bytes=rss(),persistent_gpu_budget=BUDGET)
        dump(out/'offline.json',offline)
        service=lambda ids:engine.infer([templates[i%len(templates)] for i in ids])
        phase='warmup';warmups,wall=closed_loop(service,concurrency,16,16);dump(out/'warmup.json',dict(requests=16,wall_s=wall,stats=dict(engine.stats)))
        engine.reset_stats();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        phase='formal';telemetry=Nvml(uuid);telemetry.start()
        with (out/'requests.jsonl').open('w',encoding='utf-8') as f:
            def record(batch,n):
                for row in batch:f.write(json.dumps(row)+'\n')
                f.flush()
                dump(out/'progress.json',dict(requests=n,target=requests))
            rows,wall=closed_loop(service,concurrency,requests,16,on_batch=record)
        nvml_peak=telemetry.stop();dump(out/'nvml_samples.json',telemetry.samples);telemetry=None
        tpot=[(r['end']-r['first'])/31*1000 for r in rows]
        # Batch detail stores absolute token timestamps; every request gets its own batch row.
        itls=[1000*(b-a) for r in rows for a,b in zip(r['detail']['token_times'],r['detail']['token_times'][1:])]
        result=dict(status='ok',method=method,profile=profile,concurrency=concurrency,**summarize(rows,wall,32),
                    tpot_ms={f'p{p}':percentile(tpot,p) for p in [50,95]},itl_ms={f'p{p}':percentile(itls,p) for p in [50,95]},
                    failed_requests=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    nvml_sampled_peak_bytes=nvml_peak,nvml_period_ms=100,persistent_gpu_H_bytes=engine.persistent_gpu_h,
                    persistent_gpu_KV_bytes=engine.store.bytes if engine.store else 0,prefix_stats=dict(engine.stats),
                    cpu_rss_bytes=rss(),cache_evictions=engine.store.evictions if engine.store else 0)
        dump(out/'complete.json',result)
    except torch.OutOfMemoryError as exc:
        partial=getattr(exc,'serving_progress',{})
        dump(out/'complete.json',dict(status='oom',phase=phase,completed_requests=len(partial.get('completed',[])),failed_batch=partial.get('failed_batch'),error=str(exc)))
    finally:
        if telemetry is not None:telemetry.stop()
        del engine;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');parser.add_argument('--requests',type=int,default=256);a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(42)
    prop=torch.cuda.get_device_properties(0);assert (prop.major,prop.minor)==(10,3) and prop.total_memory>200*2**30
    torch.cuda.set_per_process_memory_fraction(128*2**30/prop.total_memory,0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    uuid=str(prop.uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    dump(ROOT/'environment.json',dict(name=prop.name,hardware='B300 architecture sm_103, driver alias L20D',compute_capability=[prop.major,prop.minor],gpu_uuid=uuid,total_memory_bytes=prop.total_memory,
         torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,node=platform.node(),job=os.environ['SLURM_JOB_ID'],cpu_threads=4,cpu_affinity=sorted(os.sched_getaffinity(0)),
         allocator_cap_bytes=128*2**30,model=MODEL,adapter=ADAPTER,adapter_sha256=hashlib.sha256((Path(ADAPTER)/'adapter_model.safetensors').read_bytes()).hexdigest(),
         nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used','--format=csv,noheader'],text=True)))
    token=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);token.bos_token_id=model.config.bos_token_id
    assert token.bos_token_id==151643
    docs=json.loads((ROOT/'workloads.json').read_text());access=profiles(docs)
    dump(ROOT/'access_patterns.json',dict(patterns=access,source_documents=3,source_tokens=32768,query_tokens=512,selected_chunks=12,output_tokens=32,
         unique_prefixes={k:len({(di,tuple(docs[di]['queries'][qi]['selected'])) for di,qi in v}) for k,v in access.items()}))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        dump(ROOT/'status.json',dict(phase='CORRECTNESS'))
        validate(model,token,docs)
        if a.validate_only:return
        points=[(profile,method,c) for profile in access for method in METHODS for c in [1,4,16]]
        random.Random(20260919).shuffle(points);dump(ROOT/'order.json',points)
        for profile,method,c in points:run_point(model,token,docs,profile,access[profile],method,c,uuid,a.requests)
    dump(ROOT/'complete.json',dict(complete=True,points=30,requests_per_point=a.requests,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'status.json',dict(phase='COMPLETE',points=30))

if __name__=='__main__':
    try:main()
    except BaseException as e:dump(ROOT/'failure.json',dict(error=repr(e),at=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
