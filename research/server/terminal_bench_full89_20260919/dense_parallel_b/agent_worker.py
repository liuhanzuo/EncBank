"""One native BF16 vLLM engine; independent concurrent requests, no offload."""
from pathlib import Path
import asyncio, dataclasses, hashlib, importlib.metadata, json, os, socket, subprocess, time, traceback
from persistence import dump
H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text());R=H/'run_dense';B=R/'mailbox'
def event(kind,**kw):
    with (R/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw))+'\n')
def command(args):return subprocess.check_output(args,text=True,timeout=30)
async def main():
    H.resolve().relative_to(Path('/srv/encbank').resolve())
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    event('imports_begin')
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    assert torch.cuda.device_count()==1
    versions={x:importlib.metadata.version(x) for x in ['vllm','torch','transformers','tokenizers']}
    assert versions['vllm']==P['engine_version']
    device=torch.cuda.get_device_properties(0);snapshots=[]
    for n in range(3):
        free,total=torch.cuda.mem_get_info();assert free>=P['admission_free_gib']*2**30,(free,total)
        snapshots.append(dict(epoch=time.time(),free=free,total=total,gpu=command(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free','--format=csv,noheader']),processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid','--format=csv,noheader'])))
        if n<2:await asyncio.sleep(5)
    dump(R/'admission.json',dict(status='PASS',samples=snapshots,device=str(device),visible=os.environ['CUDA_VISIBLE_DEVICES']))
    tokenizer=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
    stop_ids=json.loads((Path(P['model'])/'generation_config.json').read_text())['eos_token_id']
    if isinstance(stop_ids,int):stop_ids=[stop_ids]
    # Small decode graphs only. Explicit KV pool and owned-process watchdog cap VRAM.
    args=AsyncEngineArgs(model=P['model'],tokenizer=P['model'],dtype=P['dtype'],tensor_parallel_size=1,
        max_model_len=P['context_tokens'],max_num_seqs=P['max_num_seqs'],max_num_batched_tokens=P['max_num_batched_tokens'],
        gpu_memory_utilization=(P['device_cap_gib']*2**30)/total,kv_cache_memory_bytes=P['kv_pool_gib']*2**30,
        language_model_only=True,enable_prefix_caching=True,mamba_cache_mode='align',
        enable_chunked_prefill=True,max_cudagraph_capture_size=4,cpu_offload_gb=0,kv_offloading_size=None,
        enable_lora=False,seed=P['seed'],generation_config='vllm',disable_log_stats=False,
        download_dir=str(H/'hf_cache'),safetensors_load_strategy='eager')
    dump(R/'engine_args.json',json.loads(json.dumps(dataclasses.asdict(args),default=str)))
    event('engine_load_begin',versions=versions)
    cold=time.monotonic();engine=AsyncLLM.from_engine_args(args)
    dump(R/'worker_ready.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
        device=str(device),versions=versions,cold_start_seconds=time.monotonic()-cold,
        plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),max_concurrent_requests=4,
        model_weights_loaded_once=True,offload=False))
    active={};seen=set();task_steps={};last_activity=time.monotonic()
    async def generate(path):
        req=json.loads(path.read_text());rid=req['request_id'];started=time.monotonic()
        reply=dict(request_id=rid,task=req['task'],task_id=req['task_id'],step=req['step'],status='error')
        gen=None;last=None
        try:
            expected=task_steps.get(req['task_id'],0);assert req['step']==expected,'Task request duplicated or out of order'
            assert req['task'] in P['tasks'];task_steps[req['task_id']]=expected+1
            ids=tokenizer.apply_chat_template(req['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,
                enable_thinking=True,reasoning_effort=P['reasoning_effort'],preserve_thinking=True)
            assert isinstance(ids,list) and all(isinstance(i,int) for i in ids)
            reply.update(prompt_tokens=len(ids),prompt_ids=ids)
            dump(B/(rid+'.prompt.json'),dict(ids=ids,task=req['task'],step=req['step'],sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
            if len(ids)>=P['context_tokens']:
                reply['status']='context_limit';return
            seed=(P['seed']+int(hashlib.sha256((req['task']+':'+str(req['step'])).encode()).hexdigest()[:8],16))%2**31
            params=SamplingParams(temperature=P['temperature'],top_p=P['top_p'],top_k=P['top_k'],seed=seed,
                max_tokens=min(P['max_new_tokens'],P['context_tokens']-len(ids)),stop_token_ids=stop_ids,
                skip_special_tokens=True)
            event('request_start',request_id=rid,task=req['task'],prompt_tokens=len(ids),seed=seed,active=len(active))
            gen=engine.generate(dict(prompt_token_ids=ids),params,rid)
            pending=asyncio.create_task(anext(gen));first=None
            while True:
                done,_=await asyncio.wait([pending],timeout=.15)
                if (B/(rid+'.cancel.json')).exists() or time.time()>=req['deadline_epoch']:
                    await engine.abort(rid);pending.cancel();await asyncio.gather(pending,return_exceptions=True)
                    reply['status']='cancelled' if (B/(rid+'.cancel.json')).exists() else 'deadline';break
                if not done:continue
                try:last=pending.result()
                except StopAsyncIteration:break
                if last.outputs and last.outputs[0].token_ids and first is None:first=time.monotonic()
                if last.finished:break
                pending=asyncio.create_task(anext(gen))
            if last is not None and last.outputs:
                out=last.outputs[0];finished=time.monotonic();metrics=vars(last.metrics) if last.metrics is not None else {}
                if metrics.get('is_corrupted'):raise FloatingPointError('vLLM reports nonfinite/corrupted logits; retain as failure')
                reply.update(text=out.text,generated_ids=list(out.token_ids),generated_tokens=len(out.token_ids),
                    cached_prefix_tokens=getattr(last,'num_cached_tokens',0),finish_reason=out.finish_reason,stop_reason=out.stop_reason,
                    ttft_seconds=(first-started) if first else None,request_seconds=finished-started,
                    decode_seconds=(finished-first) if first else None,engine_metrics=metrics,seed=seed)
                if last.finished and reply['status']=='error':reply['status']='ok'
        except BaseException as exc:
            reply.update(status='error',error=traceback.format_exc())
            await engine.abort(rid)
            if isinstance(exc,asyncio.CancelledError):raise
        finally:
            if gen is not None:await gen.aclose()
            dump(B/(rid+'.response.json'),reply)
            event('request_complete',request_id=rid,status=reply['status'],seconds=time.monotonic()-started,generated_tokens=reply.get('generated_tokens',0))
    try:
        while True:
            for rid,fut in list(active.items()):
                if fut.done():fut.result();del active[rid];last_activity=time.monotonic()
            for path in sorted(B.glob('*.request.json')):
                if path.stem in seen:continue
                if len(active)>=4:break
                seen.add(path.stem);active[path.stem]=asyncio.create_task(generate(path));last_activity=time.monotonic()
            if (B/'stop.json').exists() and not active:break
            if not active and time.monotonic()-last_activity>14400:raise TimeoutError('No requests for four hours; owner needs inspection')
            await asyncio.sleep(.05)
    finally:
        for f in active.values():f.cancel()
        await asyncio.gather(*active.values(),return_exceptions=True)
        engine.shutdown()
    dump(R/'worker_complete.json',dict(requests=len(seen),actual_shutdown=True,epoch=time.time()))
if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException:
        dump(R/'worker_failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
