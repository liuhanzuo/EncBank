"""Only the new native-SDPA seam versus frozen eager, random CPU tensors."""
import argparse,copy,datetime,hashlib,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'scientific'))
import torch,transformers,peft
import torch.nn.functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model
import tokenwise_runtime as core

class Tokenizer:
    bos_token_id=1;eos_token_id=2
    def decode(self,ids,**kw):return ' '.join(map(str,ids))
class Profiler:
    def __init__(self):self.rows=[]
    @contextmanager
    def phase(self,name):
        row={'name':name,'completed':False};self.rows.append(row);start=time.perf_counter()
        try:yield row;row['completed']=True
        finally:row['seconds']=time.perf_counter()-start
def digest(items):return [(n,str(t.dtype),list(t.shape),hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for n,t in items]
def clone_sdpa(model):
    other=copy.deepcopy(model);base=core.backbone(other);base.config._attn_implementation='sdpa'
    assert all(l.self_attn.config._attn_implementation=='sdpa' for l in base.model.layers)
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),other.state_dict().values()))
    return other

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists()
    report={'status':'running','started_at':datetime.datetime.now().astimezone().isoformat(),'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},'checks':[],'cases':[],'tolerance':{'atol':.005,'rtol':.01,'frozen_before_execution':True},'scope':'New native PyTorchSDPA CPU correctness seam only. No real checkpoint/GPU/kernel/quality/timing result.','inputs':{'seed':20260912,'Dense_document':[10+(i%7) for i in range(65)],'H_document':sum(([10+i]*512 for i in range(12)),[])+[22]*5,'query':[55,10,22,56],'bare_noncontiguous':[10,22],'bare_empty':[120],'fixed_outputs':32},'source_sha256':json.loads((HERE/'source_manifest.json').read_text())['source_sha256']}
    def save():out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def ck(name,ok,detail=None):
        report['checks'].append({'name':name,'passed':bool(ok),'detail':detail});save()
        if not ok:raise AssertionError(name)
    save();original_sdpa=F.scaled_dot_product_attention
    observations=[];active_layer=[None];hooks=[]
    def observed_sdpa(q,k,v,*args,**kwargs):
        mask=kwargs.get('attn_mask',args[0] if args else None)
        observations.append({'layer':active_layer[0],'q':list(q.shape),'k':list(k.shape),'v':list(v.shape),'dtype':str(q.dtype),'device':str(q.device),'mask_shape':None if mask is None else list(mask.shape),'mask_dtype':None if mask is None else str(mask.dtype),'is_causal':kwargs.get('is_causal',args[2] if len(args)>2 else False),'dropout':kwargs.get('dropout_p',args[1] if len(args)>1 else 0.),'scale':kwargs.get('scale'),'enable_gqa':kwargs.get('enable_gqa',False)})
        return original_sdpa(q,k,v,*args,**kwargs)
    def attach(model):
        for i,layer in enumerate(core.backbone(model).model.layers):
            hooks.append(layer.self_attn.register_forward_pre_hook(lambda m,a,kw,idx=i:active_layer.__setitem__(0,idx),with_kwargs=True))
    def compare(arm,eager,sdpa,document,bare,label):
        query=report['inputs']['query'];em=core.Method(eager,Tokenizer(),arm,resume_j=1,bos_token_id=1,eos_token_id=2);sm=core.Method(sdpa,Tokenizer(),arm,resume_j=1,bos_token_id=1,eos_token_id=2)
        ee=em.write(document,'synthetic');start=len(observations);se=sm.write(document,'synthetic');write_calls=observations[start:]
        eb=digest(ee.tensor_items());sb=digest(se.tensor_items());erefs=core.weak_refs(ee.tensor_items());srefs=core.weak_refs(se.tensor_items())
        ep=Profiler();sp=Profiler();before_eager=len(observations)
        er=core.read_fixed(em,ee,query,bare,Tokenizer(),ep,capture_logits=True)
        ck(arm+'/'+label+' eager reference does not call SDPA',len(observations)==before_eager)
        start=len(observations);sr=core.read_fixed(sm,se,query,bare,Tokenizer(),sp,capture_logits=True);read_calls=observations[start:]
        # Teacher force the frozen eager token sequence through the new SDPA
        # request so logit residuals never confound a diverging prefix.
        request=sm.open_request(se,bare);prefix=request.prefix_tokens;positions=[];logs=[]
        for i,token in enumerate(query):
            step=request.consume(token,emit_head=i==len(query)-1,phase='query')
            if arm.startswith('h'):
                request._check_lengths();positions.append([request.query_position,request.pack_position])
            else:positions.append([request.position])
            if i==len(query)-1:logs.append(step.detach().clone())
            else:assert step is None
            del step
        for token in er['generated_token_ids'][:-1]:
            step=request.consume(token,emit_head=True,phase='decode');logs.append(step.detach().clone());del step
            if arm.startswith('h'):request._check_lengths()
        rrefs=core.weak_refs(request.tensor_items());request.close();del request
        residuals=[float((x.float()-y.float()).abs().max()) for x,y in zip(er['diagnostic_logits'],logs)]
        ck(arm+'/'+label+' all32 same-context SDPA/eager logits',len(logs)==32 and all(torch.allclose(x,y,atol=.005,rtol=.01) for x,y in zip(er['diagnostic_logits'],logs)),{'per_step_maxabs':residuals,'maxabs':max(residuals)})
        ck(arm+'/'+label+' selection prefix and completequery positions',er['selected_chunk_indices']==sr['selected_chunk_indices']==(None if arm=='dense' else [0,12] if label=='noncontiguous' else []) and er['packed_read_tokens']==sr['packed_read_tokens'] and positions==([[i+1,prefix+i+1] for i in range(4)] if arm!='dense' else [[prefix+i+1] for i in range(4)]))
        ck(arm+'/'+label+' actualSDPA allnative layers plus square and cached mask',set(x['layer'] for x in write_calls+read_calls)=={0,1} and all(x['device']=='cpu' and x['dtype']=='torch.float16' and x['dropout']==0 for x in write_calls+read_calls) and any(x['q'][-2]>1 for x in write_calls) and any(x['q'][-2]==1 and x['k'][-2]>1 and not x['is_causal'] for x in read_calls) and all(x['mask_shape'] is None or x['mask_shape'][-1]==x['k'][-2] for x in write_calls+read_calls))
        ck(arm+'/'+label+' fixed32/31/head/query/lifetime unchanged',all(r['fixed_generated_tokens']==len(r['generated_token_ids'])==r['head_calls']==32 and r['decode_forward_count']==31 and r['query_prefill_calls']==4 and r['query_tokens_per_call']==1 and r['request_release']['all_tracked_tensor_objects_released'] and not r['eos_stopping_enabled'] and not r['first_step_eos_suppressed'] for r in (er,sr)) and core.released(rrefs)['all_tracked_tensor_objects_released'] and len(ep.rows)==len(sp.rows)==4 and all(r['completed'] for r in ep.rows+sp.rows))
        ck(arm+'/'+label+' eachown entry unchanged afterqueries',eb==digest(ee.tensor_items()) and sb==digest(se.tensor_items()))
        if arm!='dense':
            ck(arm+'/'+label+' nativeFP16sink/BF16metadata/no lowerdocKV',se.inventory()['document_lower_KV_bytes']==0 and se.inventory()['chunk_token_lengths']==[512]*12+[5] and all(p.scales.dtype==torch.bfloat16 and p.biases.dtype==torch.bfloat16 and p.data.dtype==torch.uint8 for p in se.chunks) if arm!='h16' else se.inventory()['document_lower_KV_bytes']==0 and all(p.data.dtype==torch.float16 for p in se.chunks))
        report['cases'].append({'arm':arm,'case':label,'selected_ids':sr['selected_chunk_indices'],'prefix_tokens':prefix,'positions_after_each_query':positions,'per_step_maxabs':residuals,'teacher_forced_top1_matches':sum(core.greedy(x)==core.greedy(y) for x,y in zip(er['diagnostic_logits'],logs)),'eager_generated_ids':er['generated_token_ids'],'SDPA_generated_ids':sr['generated_token_ids'],'free_running_exact_tokens':er['generated_token_ids']==sr['generated_token_ids'],'actual_write_SDPA_calls':write_calls,'actual_read_SDPA_calls':read_calls,'eager_entry_digest':eb,'SDPA_entry_digest':sb});save()
        ee.release();se.release();del ee,se,er,sr,logs;em.close();sm.close();del em,sm
        ck(arm+'/'+label+' entryrelease',core.released(erefs)['all_tracked_tensor_objects_released'] and core.released(srefs)['all_tracked_tensor_objects_released'])
    try:
        ck('Exact target CPU environment',transformers.__version__=='5.5.4' and torch.__version__.startswith('2.10.0') and peft.__version__=='0.20.0' and not torch.cuda.is_initialized())
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20260912)
        config=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=8192,bos_token_id=1,eos_token_id=2,pad_token_id=0,attention_dropout=0.)
        config._attn_implementation='eager';eager=Qwen3ForCausalLM(config).to(dtype=torch.float16).eval();sdpa=clone_sdpa(eager);attach(sdpa);F.scaled_dot_product_attention=observed_sdpa
        compare('dense',eager,sdpa,report['inputs']['Dense_document'],report['inputs']['bare_noncontiguous'],'fullnative')
        for h in hooks:h.remove()
        hooks.clear();del sdpa
        eager=get_peft_model(eager,LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,bias='none',target_modules=r'model.layers.1.self_attn.(q_proj|v_proj)',task_type='CAUSAL_LM'),autocast_adapter_dtype=True).eval()
        for n,t in eager.named_parameters():
            if 'lora_B' in n:t.copy_(torch.randn_like(t)*.025)
        sdpa=clone_sdpa(eager);attach(sdpa);before_adapters=[digest((n,t) for n,t in model.named_parameters() if 'lora_' in n) for model in (eager,sdpa)]
        for bits in (16,8,4):
            for label,bare in [('noncontiguous',report['inputs']['bare_noncontiguous']),('empty',report['inputs']['bare_empty'])]:compare('h'+str(bits),eager,sdpa,report['inputs']['H_document'],bare,label)
        ck('FP32 activeLoRA bytes unchanged; FP16 backbone CPU',before_adapters==[digest((n,t) for n,t in model.named_parameters() if 'lora_' in n) for model in (eager,sdpa)] and all(t.device.type=='cpu' and t.dtype==(torch.float32 if 'lora_' in n else torch.float16) for model in (eager,sdpa) for n,t in model.named_parameters()))
        ck('No CUDA context created',not torch.cuda.is_initialized())
        report.update(status='PASS_NATIVE_SDPA_SEAM_CPU_ONLY',checks_passed=sum(x['passed'] for x in report['checks']),checks_total=len(report['checks']),finished_at=datetime.datetime.now().astimezone().isoformat(),limitations=['CPU nativeSDPA compared with eager under originalatol.005/rtol.01; not bitwise equivalence','No CUDA SDPA dispatcher identity or optimizedGPU qualification claimed','Tiny random Qwen with synthetic activeFP32LoRA; no real8B/checkpoint','Original tokenwise runtime/Hcodec/kernel mathematics unchanged; no repeatedKIVI gate'])
        save();print(json.dumps({'status':report['status'],'checks':len(report['checks'])}))
    except BaseException as e:report.update(status='FAILED_PRESERVED',error={'type':type(e).__name__,'message':str(e)});save();raise
    finally:
        F.scaled_dot_product_attention=original_sdpa
        for h in hooks:h.remove()
if __name__=='__main__':main()
