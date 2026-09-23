"""New tokenwise schedule qualification only. Random eager CPU Qwen; no KIVI retest."""
import argparse,datetime,hashlib,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
import torch,transformers,peft
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model
import tokenwise_runtime as core

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(items):
    # CPU diagnostic observations only; never used in runtime timing or as offload.
    return [(n,str(t.dtype),list(t.shape),hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for n,t in items]
class Tokenizer:
    bos_token_id=1;eos_token_id=2
    def decode(self,ids,**kw):return ' '.join(map(str,ids))
class CPUProfiler:
    def __init__(self):self.rows=[]
    @contextmanager
    def phase(self,name):
        row={'name':name,'completed':False};self.rows.append(row);start=time.perf_counter()
        try:yield row;row['completed']=True
        finally:row['seconds']=time.perf_counter()-start

@torch.inference_mode()
def batch_reference(method,entry,query,bare,continuation):
    """Pinned original whole-query prefill, same prefix and teacher-forced IDs."""
    logs=[]
    if method.memory:
        e=method.memory.engine;sel=method.memory.materialize_selected(entry,method.memory.select(entry,bare))
        hj,lower,qpos=e.write_prefill(query)
        logits,upper,ppos=e.read_prefill(sel.sink_hidden,sel.hidden,hj);logs.append(logits.detach().clone());del hj,logits
        for token in continuation[:-1]:
            logits=e.decode_step(token,lower,upper,qpos,ppos);qpos+=1;ppos+=1
            logs.append(logits.detach().clone());del logits
        sel.release();del sel,lower,upper
    else:
        model=method.model
        output=model(input_ids=torch.tensor([[1]+entry.raw_ids.tolist()+query]),use_cache=True,logits_to_keep=1,return_dict=True)
        cache=output.past_key_values;logs.append(output.logits.detach().clone());del output
        for token in continuation[:-1]:
            output=model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
            cache=output.past_key_values;logs.append(output.logits.detach().clone());del output
        del cache
    return logs

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    checks=[];cases=[]
    report={'status':'running','started_at':datetime.datetime.now().astimezone().isoformat(),'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},'source_sha256':{str(f.relative_to(HERE)):sha(f) for f in [HERE/'tokenwise_runtime.py',Path(__file__).resolve(),*sorted((HERE/'runtime').rglob('*.py'))]},'checks':checks,'cases':cases,'quality_or_speed_evidence':False,'real_checkpoint_loaded':False,'KIVI_CPU_or_GPU_checks_repeated':False,'tolerance':{'all_step_teacher_forced_logits_atol':0.005,'rtol':0.01,'reason':'FP16 eager batched versus one-token arithmetic; not a bitwise or task-quality claim'},'test_model':{'random_seed':20260912,'vocab':127,'hidden':64,'layers':2,'head_dim':16,'split':1,'dtype':'FP16 backbone; FP32 active upper-layer LoRA','attention':'official eager','no_backward_or_training':True}}
    def save():out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def check(name,ok,detail=None):
        checks.append({'name':name,'passed':bool(ok),'detail':detail});save()
        if not ok:raise AssertionError(name)
    save()
    try:
        check('Exact target versions',transformers.__version__=='5.5.4' and torch.__version__.startswith('2.10.0') and peft.__version__=='0.20.0')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20260912)
        check('CUDA uninitialized before model',not torch.cuda.is_initialized())
        config=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=8192,bos_token_id=1,eos_token_id=2,pad_token_id=0,attention_dropout=0.)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).to(dtype=torch.float16).eval();tok=Tokenizer()
        query=[55,10,22,56];bare=[10,22]
        dense=core.Method(model,tok,'dense',bos_token_id=1,eos_token_id=2)
        entry=dense.write([10+(i%7) for i in range(65)],'short-dense');before=digest(entry.tensor_items());refs=core.weak_refs(entry.tensor_items())
        profiler=CPUProfiler();head_calls=[];hook=model.lm_head.register_forward_hook(lambda m,a,o:head_calls.append(list(o.shape)))
        result=core.read_fixed(dense,entry,query,bare,tok,profiler,capture_logits=True);hook.remove()
        compare=batch_reference(dense,entry,query,bare,result['generated_token_ids'])
        errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(result['diagnostic_logits'],compare)]
        check('Dense common tokenwise versus fullprefix teacher-forced logits',len(compare)==32 and all(torch.allclose(a,b,atol=.005,rtol=.01) for a,b in zip(result['diagnostic_logits'],compare)),{'maxabs':max(errors)})
        check('Dense exactly32 needed heads and4 query forwards',len(head_calls)==32 and result['query_prefill_calls']==4 and result['decode_forward_count']==31)
        check('Dense entry unchanged and request released',before==digest(entry.tensor_items()) and result['request_release']['all_tracked_tensor_objects_released'])
        check('Dense timing interface four closed phases',len(profiler.rows)==4 and all(x['completed'] for x in profiler.rows))
        cases.append({'arm':'dense','query_tokens':query,'generated_ids':result['generated_token_ids'],'reference_top1_match_count':sum(core.greedy(a)==core.greedy(b) for a,b in zip(result['diagnostic_logits'],compare)),'maxabs':max(errors)})
        entry.release();del entry,result,compare;dense.close();del dense
        check('Dense persistent release',core.released(refs)['all_tracked_tensor_objects_released'])
        # FP16 conversion above occurs BEFORE PEFT construction. Never global .half().
        adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,bias='none',target_modules=r'model.layers.1.self_attn.(q_proj|v_proj)',task_type='CAUSAL_LM'),autocast_adapter_dtype=True).eval()
        with torch.no_grad():
            for n,t in adapted.named_parameters():
                if 'lora_B' in n:t.copy_(torch.randn_like(t)*.025)
        adapters=[(n,t) for n,t in adapted.named_parameters() if 'lora_' in n]
        check('Active nonzero FP32 LoRA withFP16 backbone',len(adapters)==4 and all(t.dtype==torch.float32 for _,t in adapters) and all(t.dtype==torch.float16 for n,t in adapted.named_parameters() if 'lora_' not in n))
        adapter_before=digest(adapters);adapter_ids=[id(t) for _,t in adapters]
        document=sum(([10+i]*512 for i in range(12)),[])+[22]*5
        report['input_fixture']={'document_tokens':document,'query':query,'bare_question':bare,'empty_selection_question':[120],'source':'fixed synthetic, no dataset or checkpoint'};save()
        for bits in (16,8,4):
            method=core.Method(adapted,tok,f'h{bits}',resume_j=1,reader_binding='new-FP16-tokenwise-tiny',bos_token_id=1,eos_token_id=2)
            entry=method.write(document,'thirteen-independent-chunks');before=digest(entry.tensor_items());refs=core.weak_refs(entry.tensor_items())
            check(f'H{bits} thirteen chunks, last5 and no lower document KV',entry.inventory()['chunk_token_lengths']==[512]*12+[5] and entry.inventory()['document_lower_KV_bytes']==0)
            check(f'H{bits} actual non-contiguous selection remains belowcap',method.memory.select(entry,bare)==(0,12))
            check(f'H{bits} empty official selection remains empty',method.memory.select(entry,[120])==())
            check(f'H{bits} metadata dtype and no redundant nativeH',all(p.scales.dtype==torch.bfloat16 and p.biases.dtype==torch.bfloat16 and p.data.dtype==torch.uint8 for p in entry.chunks) if bits<16 else all(p.data.dtype==torch.float16 and p.scales is None and p.biases is None for p in entry.chunks))
            for label,bq in [('non_contiguous',bare),('empty',[120])]:
                with torch.inference_mode():
                    request=method.open_request(entry,bq);selected=tuple(request.selected_indices);prefix=1+sum(entry.chunks[i].original_shape[1] for i in selected)
                    check(f'H{bits}/{label} prefix reconstructed withoutquery',request.query_position==0 and request.pack_position==prefix and request.query_calls==request.head_calls==0)
                    check(f'H{bits}/{label} materializedselectedonly',len(request.selected.hidden)==len(selected) and request.selected.selected_indices==selected and all(core.storage(t) not in {core.storage(e) for _,e in entry.tensor_items()} for _,t in request.tensor_items()))
                    for index,token in enumerate(query):
                        step=request.consume(token,emit_head=index==len(query)-1,phase='query');request._check_lengths()
                        check(f'H{bits}/{label} positions afterquerytoken{index}',request.query_position==index+1 and request.pack_position==prefix+index+1 and ((step is None)==(index<len(query)-1)))
                        del step
                    rrefs=core.weak_refs(request.tensor_items());request.close();del request
                    check(f'H{bits}/{label} directrequest release',core.released(rrefs)['all_tracked_tensor_objects_released'])
                profiler=CPUProfiler();head_calls=[];hook=method.model.lm_head.register_forward_hook(lambda m,a,o:head_calls.append(list(o.shape)))
                result=core.read_fixed(method,entry,query,bq,tok,profiler,capture_logits=True);hook.remove()
                compare=batch_reference(method,entry,query,bq,result['generated_token_ids'])
                errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(result['diagnostic_logits'],compare)]
                check(f'H{bits}/{label} originalbatched versusnewtokenwise logits',len(compare)==32 and all(torch.allclose(a,b,atol=.005,rtol=.01) for a,b in zip(result['diagnostic_logits'],compare)),{'maxabs':max(errors)})
                check(f'H{bits}/{label} exactly32 heads/4query/31decode',len(head_calls)==32 and result['query_prefill_calls']==4 and len(result['generated_token_ids'])==32 and result['decode_forward_count']==31)
                check(f'H{bits}/{label} packedentry unchanged plusrelease',digest(entry.tensor_items())==before and result['request_release']['all_tracked_tensor_objects_released'])
                check(f'H{bits}/{label} four timing phases close',len(profiler.rows)==4 and all(x['completed'] for x in profiler.rows))
                cases.append({'arm':f'h{bits}','case':label,'selected_ids':list(selected),'prefix_tokens':prefix,'generated_ids':result['generated_token_ids'],'maxabs':max(errors),'teacher_forced_top1_match_count':sum(core.greedy(a)==core.greedy(b) for a,b in zip(result['diagnostic_logits'],compare)),'entry_sha256':hashlib.sha256(json.dumps(before).encode()).hexdigest()})
                del result,compare
            entry.release();del entry;method.close();del method
            check(f'H{bits} entryrelease aftertwo queries',core.released(refs)['all_tracked_tensor_objects_released'])
        check('FP32 activeLoRA sameobjects andbytes afterall methods',adapter_before==digest(adapters) and adapter_ids==[id(t) for _,t in adapters])
        check('CPU only, no CUDA context created',not torch.cuda.is_initialized() and all(t.device.type=='cpu' for t in adapted.parameters()))
        report.update(status='PASS_NEW_TOKENWISE_FP16_EAGER_CPU_ONLY',total_checks=len(checks),passed=sum(c['passed'] for c in checks),finished_at=datetime.datetime.now().astimezone().isoformat(),limitations=['Tiny random model, not Qwen8B or real adapter qualification','Not GPU timing/quality; original KIVI GPU gate reused as dependency, not repeated','Formal six-method loader/worker/plan not yet bound; no GPU launch readiness'])
        save();print(json.dumps({'status':report['status'],'checks':len(checks),'report_sha256':sha(out)}))
    except BaseException as e:
        report.update(status='FAILED_PRESERVED',error={'type':type(e).__name__,'message':str(e)});save();raise
if __name__=='__main__':main()
