import gc,json,os,sys,time,traceback
from pathlib import Path
from common import ROOT,PLAN as P,save,verify_sources,sha
sys.path.insert(0,str(ROOT/'vendor'))
import torch
from band_reader import JointBand
from stream_session import StreamSession
from layout import partition
from gpu_experiment import qualify
TASK=P['tasks'][0];ARM=P['arms'][0];OUT=ROOT/'pairs'/TASK;BOX=ROOT/'mailbox'/TASK
CONFIGS={'cold36':dict(policy='cold',depth=36),'exact36_c24':dict(policy='exact',depth=36,capacity=24),
    'hot36_c24':dict(policy='chunk',depth=36,capacity=24,promote=True),'hot24_c24':dict(policy='chunk',depth=24,capacity=24,promote=True)}
def status(phase,**kw):save(OUT/'status.json',dict(phase=phase,epoch=time.time(),**kw))
def main():
    verify_sources();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(0)
    free,total=torch.cuda.mem_get_info();save(OUT/'gpu_admission.json',dict(free=free,total=total,node=os.uname().nodename,job=os.environ['SLURM_JOB_ID']))
    assert free>60*2**30;torch.cuda.set_per_process_memory_fraction(64*2**30/total)
    assert sha(Path(P['adapter'])/'adapter_model.safetensors')==P['adapter_sha256']
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from unittest.mock import patch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**k:False
    status('LOAD_MODEL');tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(P['model'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,P['adapter'],autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);band=JointBand(model,tok)
    prop=torch.cuda.get_device_properties(0)
    save(OUT/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu=prop.name,uuid=str(prop.uuid),config=CONFIGS[ARM]))
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        qualify(band,tok);gc.collect();torch.cuda.empty_cache();session=StreamSession(band,tok,**CONFIGS[ARM])
        save(OUT/'ready.json',dict(ready=True,epoch=time.time()));last=-1
        while not (BOX/'stop.json').exists():
            requests=sorted([p for p in BOX.glob('*.request.json') if not p.with_name(p.name.replace('.request.','.response.')).exists()],key=lambda p:p.stat().st_mtime_ns)
            if not requests:time.sleep(.05);continue
            path=requests[0];req=json.loads(path.read_text());assert req['arm']==ARM and req['step']==last+1
            status('PREFILL',arm=ARM,step=req['step']);counter=[0]
            def cancelled():
                counter[0]+=1
                if counter[0]%64==0:status('GENERATE',arm=ARM,step=req['step'],tokens_attempted=counter[0]-1)
                return path.with_name(path.name.replace('.request.','.cancel.')).exists()
            result=session.run(req['messages'],cancelled=cancelled)
            ids=tok.apply_chat_template(req['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            chunks,query=partition(ids,session.anchor);selected=result['events'][0]['selected_indices']
            prompt=([151643]+sum([chunks[i] for i in selected],[])+query) if selected else query
            state='ok' if result['stop_reason']=='eos' else ('context_limit' if result['stop_reason']=='context_capacity' else result['stop_reason'])
            save(path.with_name(path.name.replace('.request.','.response.')),dict(request_sha256=sha(path),status=state,arm=ARM,task=TASK,step=req['step'],prompt_ids=prompt,prompt_tokens=len(prompt),**result))
            last=req['step'];status('WAIT_REQUEST',arm=ARM,step=last)
        save(OUT/'complete.json',dict(closed=True,epoch=time.time()));status('COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException:save(OUT/'worker_failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
