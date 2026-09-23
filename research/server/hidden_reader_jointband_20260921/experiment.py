"""Controlled causal joint-depth sweep, fixed original COMem weights, server only."""
import datetime,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RUN=sys.argv[1]
CFG=json.loads((ROOT/'configs.json').read_text())[RUN];OUT=ROOT/RUN/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(phase,**kw):
    x=dict(run=RUN,phase=phase,at=datetime.datetime.now().astimezone().isoformat(),**kw);dump('status.json',x);print(json.dumps(x),flush=True)
def main():
    assert os.environ.get('SLURM_JOB_ID') and not (OUT/'predictions.jsonl').exists()
    status('IMPORTS')
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from unittest.mock import patch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from band_reader import JointBand
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(20260921)
    assert torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info();assert free>100*2**30,(free,total)
    torch.cuda.set_per_process_memory_fraction(96*2**30/total)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    MODEL='/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
    ADAPTER='/srv/encbank/comem_infra_recheck_20260912/adapter'
    assert sha(Path(ADAPTER)/'adapter_model.safetensors')=='1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13'
    p=Path(CFG['input']);assert sha(p)==CFG['input_sha256']
    all_cases=json.loads(p.read_text());cases=all_cases[CFG['start']:CFG['end']]
    arms=CFG['arms']
    if arms=='selected':arms=json.loads((ROOT/'selection.json').read_text())['confirm_arms']
    status('LOAD_MODEL')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);band=JointBand(model,tok)
    dump('environment.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,transformers=transformers.__version__,model=MODEL,adapter=ADAPTER,
        input_sha256=sha(p),config=CFG,resolved_arms=arms,trainable_parameters=0,new_heads=False,
        upper_local_attention='strictly within each chunk; shared read sink is its own block',
        dtype='BF16 backbone, original unmerged FP32 LoRA',labels_read=False))
    def divergence(actual,reference):
        a=F.log_softmax(actual.float(),-1);r=F.log_softmax(reference.float(),-1)
        return dict(kl=float((r.exp()*(r-a)).sum(-1).mean()),argmax_equal=bool((actual.argmax(-1)==reference.argmax(-1)).all()),
            max_abs=float((actual.float()-reference.float()).abs().max()))
    def timed(fn):
        torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
        return result,dict(wall_ms=(time.perf_counter()-start)*1000,cuda_ms=a.elapsed_time(b))
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        status('QUALIFY')
        # Ragged blocks catch boundary, packing, and RoPE errors independently of benchmark scores.
        first=cases[0];small=band.write([first['selected_chunks'][0][:32],first['selected_chunks'][1][:48],first['selected_chunks'][2][:32]])
        qualification=[]
        for n in [12,16,24,36]:
            actual=band.memory(small,n);reference=band.dense_mask_reference(small,n)
            rel=max(float((actual.layers[l].keys.float()-reference.layers[l].keys.float()).square().mean().sqrt()/(reference.layers[l].keys.float().square().mean().sqrt()+1e-9)) for l in range(12,36))
            a,*rest=band.query(first['query_token_ids'],actual,113)
            r,*rest2=band.query(first['query_token_ids'],reference,113)
            row=dict(n=n,KV_relative_rms=rel,**divergence(a,r));qualification.append(row)
            assert rel<.04 and abs(row['kl'])<.01,row
            del actual,reference,a,r,rest,rest2
        dump('qualification.json',qualification);del small
        if CFG['phase']=='screen':
            status('TIME_RECONSTRUCTION')
            timings=[]
            for chunk_count in [1,4,12]:
                states=band.write(first['selected_chunks'][:chunk_count]);nt=sum(x.shape[1] for x in states)
                for n in [12,14,16,18,20,24,28,32,36]:
                    for _ in range(2):warm=band.memory(states,n);del warm
                    observations=[]
                    for _ in range(5):cache,t=timed(lambda:band.memory(states,n));observations.append(t);del cache
                    timings.append(dict(n=n,chunks=chunk_count,memory_tokens=nt,
                        median_wall_ms=statistics.median(x['wall_ms'] for x in observations),
                        median_cuda_ms=statistics.median(x['cuda_ms'] for x in observations),observations=observations,
                        scope='H12 ready -> all upper memory KV ready; excludes H12 write, retrieval, query, and decode'))
                del states
            dump('runtime.json',timings)
        start=time.perf_counter()
        with (OUT/'predictions.jsonl').open('x') as output:
            for i,case in enumerate(cases):
                states=band.write(case['selected_chunks']);nt=sum(x.shape[1] for x in states)
                row=dict(id=case['id'],index=CFG['start']+i,cell=CFG['cell'],phase=CFG['phase'],
                    source_document_sha256=case['source_document_sha256'],source_document_tokens=case['source_document_tokens'],
                    selected_indices=case['selected_indices'],memory_tokens=nt,arms={})
                native_logits=None
                # Rotate measurement order to avoid systematically timing one depth first.
                order=arms[i%len(arms):]+arms[:i%len(arms)]
                for arm in order:
                    torch.cuda.synchronize();begin=time.perf_counter();build=None
                    if arm=='native':logits,bottom,top,qpos=band.native(states,case['query_token_ids'])
                    else:
                        cache,build=timed(lambda:band.memory(states,int(arm[1:])))
                        logits,bottom,top,qpos=band.query(case['query_token_ids'],cache,nt)
                    torch.cuda.synchronize();prefill_ms=(time.perf_counter()-begin)*1000
                    if i==0 and arm in ['native','n36']:
                        if arm=='native':native_logits=logits.clone()
                        else:n36_logits=logits.clone()
                    generated=[];stop='length';eos=case['eos_token_id'];cap=case['max_new_tokens']
                    decode_start=time.perf_counter()
                    for step in range(cap):
                        if step==0:logits[:,-1,eos]=-torch.inf
                        token=logits[:,-1].argmax(-1);value=int(token.item());generated.append(value)
                        if value==eos:stop='eos';break
                        if step+1<cap:logits=band.reader.decode_step(token,bottom,top,qpos+step,nt+qpos+step)
                    torch.cuda.synchronize()
                    row['arms'][arm]=dict(prediction=tok.decode(generated,skip_special_tokens=True),token_ids=generated,
                        generated_tokens=len(generated),stop_reason=stop,memory_build=build,prefill_ms=prefill_ms,
                        decode_ms=(time.perf_counter()-decode_start)*1000,total_ms=(time.perf_counter()-begin)*1000)
                    del logits,bottom,top
                    if arm!='native':del cache
                if i==0:
                    control=divergence(n36_logits,native_logits);dump('native_equivalence.json',control)
                    assert abs(control['kl'])<.02,control
                    del n36_logits,native_logits
                output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush();del states
                if (i+1)%4==0 or i+1==len(cases):status('EVALUATE',completed=i+1,total=len(cases),elapsed_s=time.perf_counter()-start)
        dump('summary.json',dict(items=len(cases),arms=arms,generations=len(cases)*len(arms),
            predictions_sha256=sha(OUT/'predictions.jsonl'),elapsed_s=time.perf_counter()-start,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        status('COMPLETE',items=len(cases))
if __name__=='__main__':
    try:main()
    except BaseException as e:dump('failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
