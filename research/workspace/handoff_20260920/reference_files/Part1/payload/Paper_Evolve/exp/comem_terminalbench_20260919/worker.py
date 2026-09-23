"""Single GPU, single request at a time; incremental historical hidden bank."""
import contextlib,gc,hashlib,json,os,platform,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from torch.nn.attention import sdpa_kernel,SDPBackend
from comem import CoMem
from adaptive import nucleus,boundary_attention_mass
from bm25_index import BM25Index
from layout import partition,validate_append,fill_recent,lcp
from io_utils import save
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text());BOX=ROOT/'mailbox'

def sync():torch.cuda.synchronize();return time.perf_counter()
def check(req):
    if (BOX/(req['request_id']+'.cancel.json')).exists() or time.time()>req['deadline_epoch']:
        raise TimeoutError('Request cancelled or agent deadline elapsed')

class Reader(CoMem):
    def _as_ids(self,ids):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        if ids.ndim==1:ids=ids.unsqueeze(0)
        assert ids.ndim==2
        return ids

class Session:
    def __init__(self,reader,tok,arm):
        self.reader=reader;self.tok=tok;self.arm=arm;self.chunks=[];self.states=[];self.anchor=None
        self.kv=None;self.kv_ids=[];self.last_step=-1
    def release(self):
        self.states.clear();self.kv=None;gc.collect();torch.cuda.empty_cache()

    @torch.inference_mode()
    def run(self,req,model):
        assert req['step']==self.last_step+1,'Missing or duplicate step'
        check(req);began=sync();torch.cuda.reset_peak_memory_stats()
        messages=req['messages']
        ids=self.tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if self.anchor is None:
            first=next(i for i,m in enumerate(messages) if m['role']=='user')
            self.anchor=self.tok.apply_chat_template(messages[:first+1],tokenize=True,add_generation_prompt=False,enable_thinking=False)
            self.goal_ids=self.tok.encode(messages[first]['content'],add_special_tokens=False)[-512:]
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);eos.add(self.tok.eos_token_id)
        base=dict(logical_prompt_ids=ids,logical_history_tokens=len(ids),arm=self.arm,step=req['step'],trial=req['trial'],task=req['task'])
        if self.arm=='dense_base':
            if len(ids)+P['max_new_tokens']>P['context_tokens']:
                return dict(base,status='context_limit',error='Native 40960-token context cannot fit full prompt and reserved output; no crop')
            n=lcp(self.kv_ids,ids) if self.kv is not None else 0
            n=min(n,len(ids)-1)
            if self.kv is None:self.kv=DynamicCache(config=model.config)
            else:self.kv.crop(n)
            check(req)
            out=model(input_ids=torch.tensor([ids[n:]],device='cuda'),past_key_values=self.kv,use_cache=True,logits_to_keep=1)
            logits=out.logits;self.kv=out.past_key_values;self.kv_ids=list(ids)
            base.update(prompt_tokens=len(ids),cached_prefix_tokens=n,hidden_bank_tokens=0,selected_chunks=0,write_seconds=0,selection_seconds=0,query_tokens=len(ids))
            del out
        else:
            chunks,query=partition(ids,self.anchor,P['recent_tokens'],P['chunk_tokens'])
            validate_append(self.chunks,chunks)
            additions=chunks[len(self.chunks):]
            if len(chunks)*512*self.reader.hidden_size*2>P['max_hidden_gpu_gib']*2**30:
                return dict(base,status='memory_limit',error='Registered hidden-bank GPU cap reached')
            write_start=sync()
            for offset in range(0,len(additions),8):
                check(req)
                batch=self.reader.write_chunk(torch.tensor(additions[offset:offset+8],device='cuda'))
                self.states.extend([h.unsqueeze(0).clone() for h in batch]);del batch
            write_s=sync()-write_start;self.chunks=chunks
            if len(query)+P['max_new_tokens']+1>P['context_tokens']:
                return dict(base,status='context_limit',error='Pinned instruction/query exceeds native context')
            check(req);query_start=sync();qh,bottom,qpos=self.reader.write_prefill(query);query_s=sync()-query_start
            select_start=sync();selected=[];selection={};index=BM25Index(chunks)
            lexical=self.goal_ids+self.tok.encode(messages[-1]['content'],add_special_tokens=False)[-2048:]
            if chunks:
                if self.arm.startswith('iter_k'):
                    selected,filled=fill_recent(index.select(lexical,int(self.arm.split('_k')[1]),P['hop']),len(chunks),int(self.arm.split('_k')[1]))
                    selection=dict(recency_filled=filled)
                elif self.arm=='bm25_p90':
                    selection=nucleus(index.scores(lexical),p=P['p'],min_chunks=P['min_chunks'],max_chunks=P['max_chunks'])
                    selected=selection['selected']
                elif self.arm=='iter48_qk_p90':
                    candidates,filled=fill_recent(index.select(lexical,48,P['hop']),len(chunks),48)
                    memory=torch.cat([self.states[i] for i in candidates],dim=1)
                    scores=boundary_attention_mass(self.reader,memory,qh,probe_tokens=P['probe_tokens']).cpu().tolist()
                    selection=nucleus(scores,p=P['p'],min_chunks=P['min_chunks'],max_chunks=48,ids=candidates)
                    selection.update(candidates=candidates,scores=scores,recency_filled=filled)
                    selected=selection['selected'];del memory
                else:raise ValueError(self.arm)
            selection_s=sync()-select_start
            packed=len(query)+len(selected)*512+int(bool(selected))
            if packed+P['max_new_tokens']>P['context_tokens']:
                return dict(base,status='context_limit',error='Packed hidden/query plus reserved output exceeds native context')
            sink=self.reader.write_chunk([self.reader._sink_prefix_id()]) if selected else None
            check(req)
            logits,top,ppos=self.reader.read_prefill(sink,[self.states[i] for i in selected],qh)
            base.update(prompt_tokens=packed,cached_prefix_tokens=0,query_tokens=len(query),hidden_bank_tokens=len(chunks)*512,
                hidden_bank_bytes=sum(h.numel()*h.element_size() for h in self.states),newly_written_tokens=len(additions)*512,
                selected_chunks=len(selected),selected_indices=selected,selected_age_chunks=[len(chunks)-1-i for i in selected],
                omitted_history_chunks=len(chunks)-len(selected),write_seconds=write_s,query_write_seconds=query_s,selection_seconds=selection_s,selection=selection)
            del qh,sink
        generated=[];token_times=[]
        for step in range(P['max_new_tokens']):
            check(req);token=int(logits[0,-1].argmax());generated.append(token);token_times.append(sync()-began)
            if token in eos:break
            if step+1==P['max_new_tokens']:break
            if self.arm=='dense_base':
                out=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=self.kv,use_cache=True,logits_to_keep=1)
                logits=out.logits;self.kv=out.past_key_values;self.kv_ids.append(token);del out
            else:logits=self.reader.decode_step(token,bottom,top,qpos+step,ppos+step)
        text=self.tok.decode(generated,skip_special_tokens=True)
        self.last_step=req['step']
        return dict(base,status='ok',text=text,generated_ids=generated,generated_tokens=len(generated),
            stopped_on_eos=generated[-1] in eos,ttft_seconds=token_times[0],request_seconds=sync()-began,
            token_times_seconds=token_times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def smoke(model,wrapper,tok,reader):
    """No-memory split equals native same-adapter forward; history layout is separate."""
    ids=tok.apply_chat_template([dict(role='user',content='Return only the integer two.')],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    with torch.inference_mode():
        q,b,n=reader.write_prefill(ids);a,c,k=reader.read_prefill(None,[],q)
        ref=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits
        d=(a-ref).float();result=dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),top1_equal=bool(a.argmax(-1).eq(ref.argmax(-1)).all()))
        assert result['rms']<.15 and result['top1_equal'],result
    with torch.inference_mode(),wrapper.disable_adapter():
        cache=DynamicCache(config=model.config);cut=max(1,len(ids)//2)
        model(input_ids=torch.tensor([ids[:cut]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
        a=model(input_ids=torch.tensor([ids[cut:]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1).logits
        b=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits
        d=(a-b).float();kv=dict(rms=float(d.square().mean().sqrt()),max_abs=float(d.abs().max()),top1_equal=bool(a.argmax(-1).eq(b.argmax(-1)).all()))
        assert kv['rms']<.15 and kv['top1_equal'],kv
        cache.crop(cut)
        a=model(input_ids=torch.tensor([ids[cut:]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1).logits
        assert float((a-b).float().square().mean().sqrt())<.15
    from types import SimpleNamespace
    from qk_validation import validate_attention
    with torch.inference_mode():
        chunk=(ids*((512+len(ids)-1)//len(ids)))[:512]
        bank=reader.write_chunk(torch.tensor([chunk,chunk],device='cuda'))
        validate_attention(SimpleNamespace(reader=reader,bank=bank,sink=reader.write_chunk([reader._sink_prefix_id()]),docs=[dict(queries=[dict(query=chunk)])]))
    save(ROOT/'smoke.json',dict(passed=True,no_memory_split=result,dense_incremental_kv=kv,attention_probe_reference='correctness.json'))

def main():
    assert torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID');BOX.mkdir(exist_ok=True)
    prop=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(P['allocator_gib']*2**30/prop.total_memory)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**k:False
    tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(P['model'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,P['adapter'],autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);tok.bos_token_id=model.config.bos_token_id
    reader=Reader(model,P['split_depth'],tokenizer=tok)
    save(ROOT/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=prop.name,gpu_uuid=str(prop.uuid),total_memory_bytes=prop.total_memory,
        node=platform.node(),torch=torch.__version__,transformers=transformers.__version__,adapter_sha256=hashlib.sha256((Path(P['adapter'])/'adapter_model.safetensors').read_bytes()).hexdigest()))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        smoke(model,wrapper,tok,reader);save(ROOT/'ready.json',dict(ready=True,at_epoch=time.time()))
        current=None;session=None;last_contact=time.monotonic();done=0
        while not (ROOT/'stop.json').exists():
            paths=[p for p in sorted(BOX.glob('*.request.json')) if not p.with_name(p.name.replace('.request.json','.response.json')).exists()]
            if not paths:
                if time.monotonic()-last_contact>7200:raise TimeoutError('No controller request for two hours')
                time.sleep(.1);continue
            req=json.loads(paths[0].read_text());rid=req['request_id'];last_contact=time.monotonic()
            if req['task_id']!=current:
                if session is not None:session.release()
                current=req['task_id'];session=Session(reader,tok,req['arm'])
            assert session.arm==req['arm']
            save(ROOT/'status.json',dict(phase='RUNNING',request_id=rid,arm=req['arm'],task=req['task'],step=req['step'],completed_requests=done))
            try:
                with wrapper.disable_adapter() if req['arm']=='dense_base' else contextlib.nullcontext():
                    result=session.run(req,model)
            except TimeoutError as exc:result=dict(status='cancelled',error=str(exc),arm=req['arm'],task=req['task'],step=req['step'])
            except torch.OutOfMemoryError as exc:
                result=dict(status='oom',error=str(exc),arm=req['arm'],task=req['task'],step=req['step']);session.release()
            except BaseException:
                save(ROOT/'failure.json',dict(error=traceback.format_exc(),request=req['request_id']));raise
            save(BOX/(rid+'.response.json'),result);done+=1;last_contact=time.monotonic()
        save(ROOT/'complete.json',dict(completed_requests=done,controller_stopped=True))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'failure.json',dict(error=traceback.format_exc()));raise
