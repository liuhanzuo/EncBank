"""Frozen real Terminal-Bench trace replay plus >2-chunk generation qualification."""
import gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'vendor'))
import torch
import torch.nn.functional as F
from common import save,verify_sources,PLAN as P,sha
from band_reader import JointBand
from hot_memory import HotMemory,Pool,token_key
from stream_session import StreamSession

VARIANTS=[('cold36',dict(policy='cold',depth=36)),('cold24',dict(policy='cold',depth=24)),
    ('exact36_c24',dict(policy='exact',depth=36,capacity=24)),
    ('hot36_c12',dict(policy='chunk',depth=36,capacity=12,promote=True)),
    ('hot36_c24',dict(policy='chunk',depth=36,capacity=24,promote=True)),
    ('hot36_c48',dict(policy='chunk',depth=36,capacity=48,promote=True)),
    ('hot36_c24_no_promote',dict(policy='chunk',depth=36,capacity=24,promote=False)),
    ('hot24_c24',dict(policy='chunk',depth=24,capacity=24,promote=True))]
def kl(a,b):
    return float((F.softmax(a.double(),-1)*(F.log_softmax(a.double(),-1)-F.log_softmax(b.double(),-1))).sum())
def clean(result):return {k:v for k,v in result.items() if k not in ['text','generated_ids','first_logits','boundary_logits']}
def status(phase,**kw):save(ROOT/'status.json',dict(phase=phase,epoch=time.time(),**kw))
def qualify(band,tok):
    pool=Pool(16)
    def entry(x):return dict(kv=[(torch.full((2,),x,dtype=torch.float16),torch.full((2,),x,dtype=torch.float16))],position=0,tokens=1,origin='test')
    pool.put('a',entry(1));pool.put('b',entry(2));pool.get('a');pool.put('c',entry(3));assert set(pool.items)=={'a','c'} and pool.bytes==16
    chunks=[]
    for i in range(4):
        ids=tok.encode(f'Terminal observation {i}: key = {i*137+3}. Inspect the files carefully.\n',add_special_tokens=False)
        chunks.append((ids*512)[:512])
    states=band.write(chunks);query=tok.encode('What is the requested key? Return a JSON object.',add_special_tokens=False);rows=[]
    for depth in [36,24]:
        hot=HotMemory(band,'exact',depth,8)
        for selected,expected in [([0,1,2],0),([0,1,2],3),([0,1,3],2),([1,2,3],0)]:
            parts=[(-1,'sink')]+[(i,token_key(chunks[i])) for i in selected];hs=[states[0]]+[states[i+1] for i in selected]
            cold=HotMemory(band,'cold',depth)
            a,*_=cold.prefill(parts,hs,query);b,*_=hot.prefill(parts,hs,query)
            divergence=kl(a[0,-1].float().cpu(),b[0,-1].float().cpu())
            assert divergence<.03,(depth,selected,divergence)
            assert hot.last.get('hit_chunks',0)==expected,(depth,selected,dict(hot.last))
            rows.append(dict(depth=depth,selected=selected,hits=expected,kl=divergence))
    hot=HotMemory(band,'chunk',36,8)
    parts=[(-1,'sink')]+[(i,token_key(chunks[i])) for i in [0,1,2]]
    _,bottom,top,qpos,nt=hot.prefill(parts,[states[0]]+states[1:4],query)
    reference=[tuple(t.clone() for t in pair) for pair in next(iter(hot.pool.items.values()))['kv']]
    band.reader.decode_step(torch.tensor([query[0]],device='cuda'),bottom,top,qpos,nt+qpos)
    stored=next(iter(hot.pool.items.values()))['kv'];assert all(torch.equal(a,b) for aa,bb in zip(reference,stored) for a,b in zip(aa,bb))
    parts=[(-1,'sink')]+[(i,token_key(chunks[i])) for i in [1,2,3]]
    hot.prefill(parts,[states[0]]+states[2:5],query)
    assert hot.last['hit_chunks']==2 and hot.last['rebuilt_chunks']==1 and hot.last['rebase_operations']>0
    save(ROOT/'qualification.json',dict(passed=True,exact_prefix_cases=rows,chunk_partial_hit_rebase=True,active_decode_does_not_mutate_pool=True,lru_byte_budget=True))

def main():
    verify_sources();assert torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(0)
    free,total=torch.cuda.mem_get_info();save(ROOT/'gpu_admission.json',dict(free=free,total=total,job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename))
    assert free>60*2**30;(torch.cuda.set_per_process_memory_fraction(64*2**30/total))
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
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        status('QUALIFY');qualify(band,tok);gc.collect();torch.cuda.empty_cache()
        traces=json.loads((ROOT/'traces.json').read_text());results=[];refs={}
        # Independent fresh sessions each repeat; reverse treatment order on the second pass.
        for repeat in range(2):
            for name,config in (VARIANTS if repeat==0 else list(reversed(VARIANTS))):
                for task,requests in traces.items():
                    session=StreamSession(band,tok,**config)
                    for index,req in enumerate(requests):
                        status('TRACE_REPLAY',repeat=repeat,variant=name,task=task,index=index)
                        result=session.run(req['messages'],forced=[req['first_output_token']],capture_logits=True)
                        refkey=(config['depth'],task,index)
                        if name.startswith('cold'):refs[refkey]=result['first_logits']
                        reference=refs.get(refkey)
                        divergence=kl(reference,result['first_logits']) if reference is not None else None
                        if name.startswith('exact'):assert divergence is not None and divergence<.03,(name,task,index,divergence)
                        results.append(dict(repeat=repeat,variant=name,task=task,index=index,source_request_sha256=req['source_request_sha256'],
                            next_token_kl_vs_cold=divergence,next_token_argmax_matches_cold=bool(reference.argmax()==result['first_logits'].argmax()) if reference is not None else None,**clean(result)))
                    del session;gc.collect();torch.cuda.empty_cache();save(ROOT/'trace_results.json',results)
        # These forced tokens are a mechanical streaming test, not an agent-quality score.
        seed=next(iter(traces.values()))[0]['messages']
        forced=tok.encode('Inspect the terminal output and record the observed values.\n',add_special_tokens=False)*120
        forced=forced[:1056];assert len(forced)==1056
        streaming=[];reference=None
        for name,config in [VARIANTS[0],VARIANTS[2],VARIANTS[4],VARIANTS[6],VARIANTS[7]]:
            status('STREAMING_1056',variant=name);session=StreamSession(band,tok,**config)
            result=session.run(seed,forced=forced,capture_logits=True)
            assert [e.get('generated_boundary') for e in result['events'][1:]]==[512,1024]
            if name=='cold36':reference=result['boundary_logits']
            divergences=[kl(a,b) for a,b in zip(reference,result['boundary_logits'])]
            if name=='exact36_c24':assert max(divergences)<.03,divergences
            if name=='hot36_c24':assert result.get('promoted_hits',0)>0,result
            streaming.append(dict(variant=name,boundary_kls_vs_cold36=divergences,**clean(result)))
            save(ROOT/'stream_results.json',streaming);del session;gc.collect();torch.cuda.empty_cache()
        save(ROOT/'complete.json',dict(passed=True,trace_measurements=len(results),stream_variants=len(streaming),epoch=time.time(),quality_claim=False));status('COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
