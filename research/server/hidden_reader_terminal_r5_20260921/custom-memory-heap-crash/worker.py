"""One model owner for a task's two sequential live trials, followed by replay."""
import collections,gc,json,os,platform,sys,time,traceback
from pathlib import Path
from common import ROOT,PLAN as P,save,sha,digest,verify_sources
sys.path.insert(0,str(ROOT/'vendor'))
import torch
import torch.nn.functional as F
from band_reader import JointBand
from bm25_index import BM25Index
from layout import partition,fill_recent
TASK=sys.argv[1];OUT=ROOT/'pairs'/TASK;BOX=ROOT/'mailbox'/TASK
def stamp():torch.cuda.synchronize();return time.perf_counter()
def status(phase,**kw):save(OUT/'status.json',dict(phase=phase,epoch=time.time(),**kw))
class Session:
    def __init__(self,band,tok,arm):
        self.band=band;self.reader=band.reader;self.tok=tok;self.arm=arm;self.chunks=[];self.states=[];self.anchor=None;self.last_step=-1
        self.sink=None
    @torch.inference_mode()
    def run(self,req,forced=None):
        began=stamp();torch.cuda.reset_peak_memory_stats();messages=req['messages']
        if forced is None:assert req['step']==self.last_step+1
        ids=self.tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        if self.anchor is None:
            first=next(i for i,m in enumerate(messages) if m['role']=='user')
            self.anchor=self.tok.apply_chat_template(messages[:first+1],tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)
            self.goal=self.tok.encode(messages[first]['content'],add_special_tokens=False)[-512:]
        chunks,query=partition(ids,self.anchor,P['recent_tokens'],512)
        assert chunks[:len(self.chunks)]==self.chunks,'Earlier archived tokens changed'
        if len(chunks)*512*4096*2>P['hidden_limit_gib']*2**30:
            return dict(status='memory_limit',error='H12 bank exceeds registered physical memory budget')
        start=stamp();additions=chunks[len(self.chunks):]
        for offset in range(0,len(additions),8):
            batch=self.reader.write_chunk(torch.tensor(additions[offset:offset+8],device='cuda'))
            # Cache tensors need normal version counters for the read-only audit.
            # Their values/dtype are unchanged; all model operations remain inference-only.
            with torch.inference_mode(False):
                self.states.extend(h[None].detach().clone() for h in batch)
        self.chunks=chunks
        if chunks and self.sink is None:self.sink=self.reader.write_chunk([151643])
        write_s=stamp()-start;start=time.perf_counter()
        lexical=self.goal+self.tok.encode(messages[-1]['content'],add_special_tokens=False)[-2048:]
        selected=fill_recent(BM25Index(chunks).select(lexical,P['topk'],P['hop']),len(chunks),P['topk']) if chunks else []
        select_s=time.perf_counter()-start;nt=len(selected)*512+int(bool(selected))
        packed=nt+len(query)
        if packed>=P['context_tokens']:
            return dict(status='context_limit',error='Packed input reaches native40960 positions; no cropping',prompt_tokens=packed)
        versions=[x._version for x in self.states];prefill_start=stamp();memory_s=None
        states=([self.sink]+[self.states[i] for i in selected]) if selected else []
        if not states:
            h,bottom,qpos=self.reader.write_prefill(query)
            logits,top,_=self.reader.read_prefill(None,[],h)
        elif self.arm=='native':logits,bottom,top,qpos=self.band.native(states,query)
        else:
            mstart=stamp();top=self.band.memory(states,int(self.arm[1:]));memory_s=stamp()-mstart
            logits,bottom,top,qpos=self.band.query(query,top,nt)
        prefill_s=stamp()-prefill_start;gen=[];arrivals=[];decode_start=stamp()
        stop=self.reader.model.generation_config.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop]);stop.add(self.tok.eos_token_id)
        limit=P['context_tokens']-packed;cap=min(limit,len(forced)) if forced is not None else limit
        reason='context_capacity';greedy_first=int(logits[0,-1].argmax())
        for step in range(cap):
            if forced is None and (BOX/(req['request_id']+'.cancel.json')).exists():reason='cancelled';break
            token=int(logits[0,-1].argmax()) if forced is None else forced[step]
            gen.append(token);arrivals.append(time.perf_counter()-began)
            if token in stop:reason='eos';break
            if forced is None and (step+1)%64==0:status('GENERATE',arm=self.arm,step=req['step'],generated=len(gen),history_tokens=len(ids),selected_chunks=len(selected))
            if step+1<cap:logits=self.reader.decode_step(torch.tensor([token],device='cuda'),bottom,top,qpos+step,packed+step)
        ended=stamp();assert versions==[x._version for x in self.states],'Read mutated cached H12'
        self.last_step=req['step']
        state='ok' if reason=='eos' else ('replay_ok' if forced is not None else ('context_limit' if reason=='context_capacity' else reason))
        return dict(status=state,arm=self.arm,task=TASK,step=req['step'],text=self.tok.decode(gen,skip_special_tokens=True),
            generated_ids=gen,generated_tokens=len(gen),stop_reason=reason,greedy_first_token=greedy_first,
            prompt_tokens=packed,prompt_ids=([151643]+sum([chunks[i] for i in selected],[])+query) if selected else query,
            logical_prompt_tokens=len(ids),logical_prompt_sha256=digest(ids),query_tokens=len(query),
            archived_chunks=len(chunks),selected_chunks=len(selected),selected_indices=selected,
            newly_written_tokens=len(additions)*512,hidden_bytes=sum(h.numel()*h.element_size() for h in self.states),
            write_seconds=write_s,selection_seconds=select_s,memory_build_seconds=memory_s,prefill_seconds=prefill_s,
            ttft_seconds=arrivals[0] if arrivals else None,decode_seconds=ended-decode_start,request_seconds=ended-began,
            token_times_seconds=arrivals,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            H12_versions_unchanged=True,configured_generation_limit=None)

def qualify(band,tok,model):
    ids=tok.encode('Check causal memory reconstruction and return a short JSON object.',add_special_tokens=False)
    chunk=(ids*(-(-512//len(ids))))[:512]
    pieces=[chunk,chunk];states=band.write(pieces);query=ids
    result=[]
    for n in [24,36]:
        a=band.memory(states,n);b=band.dense_mask_reference(states,n)
        errors={name:max(float((getattr(a.layers[l],name).float()-getattr(b.layers[l],name).float()).square().mean().sqrt()/(getattr(b.layers[l],name).float().square().mean().sqrt()+1e-9)) for l in range(12,36)) for name in ['keys','values']}
        la,*_=band.query(query,a,1025);lb,*_=band.query(query,b,1025)
        kl=float((F.softmax(lb.float(),-1)*(F.log_softmax(lb.float(),-1)-F.log_softmax(la.float(),-1))).sum())
        assert max(errors.values())<.04 and abs(kl)<.01,(n,errors,kl)
        result.append(dict(n=n,relative_rms=errors,query_kl=kl))
    native,*_=band.native(states,query);cache=band.memory(states,36);split,*_=band.query(query,cache,1025)
    kl=float((F.softmax(native.float(),-1)*(F.log_softmax(native.float(),-1)-F.log_softmax(split.float(),-1))).sum())
    assert abs(kl)<.02,kl
    rounds=[]
    for arm in ['native','n24']:
        session=Session(band,tok,arm)
        messages=[dict(role='system',content='Return a short JSON object.'),dict(role='user',content='Inspect the terminal log.')]
        for step in range(3):
            req=dict(messages=messages,step=step,request_id='qualification-only')
            response=session.run(req,forced=ids[:2])
            assert response['H12_versions_unchanged'] and response['generated_tokens']==2
            rounds.append(dict(arm=arm,step=step,archived_chunks=response['archived_chunks'],selected_chunks=response['selected_chunks']))
            messages=messages+[dict(role='assistant',content='{"commands": []}'),dict(role='user',content='observed terminal log line alpha beta gamma\n'*1000)]
        assert len(session.states)>12 and all(not h.is_inference() for h in session.states)
        del session
    save(OUT/'qualification.json',dict(passed=True,mask_checks=result,native_split_kl=kl,multiturn_cache_checks=rounds))

def replay(band,tok):
    candidates=[]
    for path in BOX.glob('native_*.response.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='ok' and row.get('archived_chunks',0)>0:
            candidates.append((row['archived_chunks'],row['step'],path,row))
    candidates=sorted(candidates,key=lambda x:(-x[0],x[1]))[:3];results=[]
    for _,_,path,row in candidates:
        req=json.loads(path.with_name(path.name.replace('.response.','.request.')).read_text());forced=row['generated_ids'][:128]
        for repeat in range(2):
            for arm in (['native','n36','n24'] if repeat==0 else ['n24','n36','native']):
                session=Session(band,tok,arm);result=session.run(req,forced=forced)
                results.append(dict(source_request=req['request_id'],repeat=repeat,forced_tokens=len(forced),**{k:v for k,v in result.items() if k not in ['text','prompt_ids','generated_ids','token_times_seconds']}));del session
        save(OUT/'replay.json',results)
    save(OUT/'replay_receipt.json',dict(source_requests=len(candidates),measurements=len(results),quality_claim=False))

def main():
    verify_sources();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(P['seed'])
    free,total=torch.cuda.mem_get_info()
    save(OUT/'gpu_admission.json',dict(free_bytes=free,total_bytes=total,gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    assert free>100*2**30,(free,total)
    torch.cuda.set_per_process_memory_fraction(P['allocator_gib']*2**30/total)
    assert sha(Path(P['adapter'])/'adapter_model.safetensors')==P['adapter_sha256']
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from unittest.mock import patch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import transformers,transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**k:False
    status('LOAD_MODEL');tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(P['model'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,P['adapter'],autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);band=JointBand(model,tok)
    prop=torch.cuda.get_device_properties(0)
    save(OUT/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],node=platform.node(),gpu=prop.name,gpu_uuid=str(prop.uuid),torch=torch.__version__,transformers=transformers.__version__,model=P['model'],adapter_sha256=P['adapter_sha256']))
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        qualify(band,tok,model);save(OUT/'ready.json',dict(ready=True,epoch=time.time()));session=None;current=None
        while not (BOX/'stop.json').exists():
            requests=sorted([p for p in BOX.glob('*.request.json') if not p.with_name(p.name.replace('.request.','.response.')).exists()],key=lambda p:p.stat().st_mtime_ns)
            if not requests:time.sleep(.05);continue
            path=requests[0];req=json.loads(path.read_text());assert req['task']==TASK and req['arm'] in P['arms']
            if req['arm']!=current:
                if session is not None:del session;gc.collect();torch.cuda.empty_cache()
                session=Session(band,tok,req['arm']);current=req['arm']
            status('PREFILL',arm=current,step=req['step']);result=session.run(req)
            save(path.with_name(path.name.replace('.request.','.response.')),dict(request_sha256=sha(path),**result));status('WAIT_REQUEST',arm=current,step=req['step'])
        if session is not None:del session;gc.collect();torch.cuda.empty_cache()
        status('REPLAY');replay(band,tok);save(OUT/'complete.json',dict(live_closed=True,replay_closed=True,epoch=time.time()));status('COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException:save(OUT/'worker_failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
