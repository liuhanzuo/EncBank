"""Native Qwen chat capability pilot, fresh full transcript each turn; not an infra claim."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model',default='/srv/encbank/comem_new_backbones_20260915/models/Qwen3.8-27B');a=p.parse_args()
R=Path(a.run).resolve();R.relative_to(H.resolve());plan=json.loads((H/'agent_plan.json').read_text())
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def stable_free():
    start=None
    while True:
        q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        ok=int(q.strip().splitlines()[0])>=26*1024 and not any('python' in x.lower() for x in apps.splitlines())
        start=(start or time.monotonic()) if ok else None
        save(R/'admission.json',dict(free_mib=int(q.strip().splitlines()[0]),stable_seconds=0 if start is None else time.monotonic()-start,ready=False))
        if start is not None and time.monotonic()-start>=45: return
        time.sleep(5)
try:
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoConfig,AutoTokenizer,StoppingCriteria,StoppingCriteriaList
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(plan['seed'])
    free,total=torch.cuda.mem_get_info();cap=min(80*2**30,int(total*.90))
    assert free>=70*2**30, 'Insufficient fresh free memory for all-resident 27B'
    torch.cuda.set_per_process_memory_fraction(cap/total)
    admission=dict(job=os.environ['SLURM_JOB_ID'],free_bytes=free,total_bytes=total,cap_bytes=cap,
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
    config=AutoConfig.from_pretrained(a.model,local_files_only=True)
    cls=AutoModelForImageTextToText if hasattr(config,'text_config') else AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=cls.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,device_map='cuda').eval()
    assert all(p.device.type=='cuda' for p in model.parameters())
    save(R/'worker_ready.json',dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,model=a.model,admission=admission,cap_bytes=cap))
    box=R/'mailbox';seen=set();last=time.monotonic()
    while not (box/'stop.json').exists():
        todo=sorted(p for p in box.glob('*.request.json') if p.name not in seen)
        if not todo:
            if time.monotonic()-last>1800: raise TimeoutError('actor idle timeout')
            time.sleep(.2);continue
        for req in todo:
            data=json.loads(req.read_text());rendered=tokenizer.apply_chat_template(data['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').input_ids
            n=ids.shape[1];response=dict(task_id=data['task_id'],step=data['step'],prompt_tokens=n)
            if n>plan['max_prompt_tokens']: response.update(status='context_limit',text='')
            else:
                try:
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                    class FirstToken(StoppingCriteria):
                        first=None
                        def __call__(self,input_ids,scores,**kwargs):
                            if self.first is None:
                                torch.cuda.synchronize();self.first=time.perf_counter()
                            return False
                    timer=FirstToken()
                    with torch.inference_mode(): output=model.generate(ids.to('cuda'),max_new_tokens=plan['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([timer]))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-t
                    new=output[0,n:].tolist()
                    response.update(status='ok',text=tokenizer.decode(new,skip_special_tokens=True),generated_ids=new,generated_tokens=len(new),model_seconds=elapsed,
                        ttft_seconds=timer.first-t,decode_tokens_per_second=(len(new)-1)/max(1e-9,t+elapsed-timer.first),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),hit_generation_cap=len(new)==plan['max_new_tokens'])
                    del output;torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    response.update(status='OOM',text='');torch.cuda.empty_cache()
            save(req.with_name(req.name.replace('.request.','.response.')),response);seen.add(req.name);last=time.monotonic()
    save(R/'worker_complete.json',dict(completed=True,requests=len(seen)))
except BaseException as exc:
    save(R/'worker_failure.json',dict(type=type(exc).__name__,traceback=traceback.format_exc()));raise
