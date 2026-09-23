"""Train a fresh CacheBlend-style suffix LoRA with the exact Encbank recipe."""
import argparse,gc,json,os,random,time,traceback
import config
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import transformers
from encbank import Encbank
from kv_reference import KVControl,assemble
from trainable_cacheblend import TrainableBlend
import train_support as support

class Reader(support.PublishedReader):
    def __init__(self,model,j):
        super().__init__(model,j,True)
        self.blend=TrainableBlend(model,j,True)
    def hidden(self,window,bos,chunk,n_ctx,teacher=False):
        if teacher:return super().hidden(window,bos,chunk,n_ctx,True)
        segments=[[bos]]+[x.tolist() for x in window.split(chunk)]
        assert len(segments)==n_ctx+2
        return self.blend.hidden(segments,.15)

@torch.no_grad()
def parity(reader,tok):
    ids=tok.encode('Alice keeps 42 blue books. Bob keeps 73 red books. '*4,add_special_tokens=False)
    segments=[[151643],ids[:24],ids[24:48],ids[48:]]
    assert segments[-1]
    pack=torch.tensor([sum(segments,[])],device='cuda')
    ref=KVControl(Encbank(reader.model,0))
    pp,kv=assemble(ref,segments[1:-1],segments[-1],151643)
    a,_,_=ref.read(pp,kv,1,len(segments[-1]),.15)
    b=reader.model.lm_head(reader.blend.hidden(segments,.15)[:,-1:])
    err=float((a-b).abs().max());assert err<=.125 and torch.equal(a.argmax(-1),b.argmax(-1)),err
    full=reader.model(pack,use_cache=False,logits_to_keep=1).logits
    recompute=reader.model.lm_head(reader.blend.hidden(segments,1.)[:,-1:])
    dense_error=float((full-recompute).abs().max())
    assert dense_error<=.125 and torch.equal(full.argmax(-1),recompute.argmax(-1)),dense_error
    return dict(r015_reference_max_abs=err,r1_stock_max_abs=dense_error,top1_equal=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=config.ROOT/'training';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        assert json.loads((out/'complete.json').read_text())['steps']==4000
        print('Training already complete');return
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert a.resume or not (out/'last.pt').exists(),'Explicit --resume required'
    original=json.loads((config.ROOT/'encbank_training_metadata.json').read_text())
    args=support.parser().parse_args(['--model',config.MODEL,'--data',config.DATA,'--out',str(out),'--devices','cuda:0'])
    comparable=('j','rank','alpha','chunk','n_ctx','topk','lam','loss','adapter_dtype','steps','lr','warmup','seed','grad_accum','skip_windows')
    assert {k:getattr(args,k) for k in comparable}=={k:original['recipe'][k] for k in comparable}
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    data=support.prepare_tokens(config.DATA,tok,out/'pg19_tokens.u32',config.MODEL)
    assert data['source_sha256']==original['recipe']['source_sha256']
    assert data['documents']==64 and data['tokens']==original['recipe']['data_tokens']==7231927
    torch.manual_seed(args.seed);random.seed(args.seed)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    model=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    modules=support.attach_lora(model,12,32,32,torch.float32)
    params=[p for m in modules.values() for p in (m.A,m.B)]
    assert len(modules)==168 and sum(p.numel() for p in params)==original['trainable_parameters']==58195968
    # Evaluation sets configured BOS explicitly. Training exactly preserves the
    # original trainer's tokenizer-BOS-else-EOS rule, even when those IDs differ.
    bos=int(tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id)
    reader=Reader(model,12)
    support.atomic_json(out/'correctness_initial.json',parity(reader,tok))
    optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),weight_decay=0.,foreach=False)
    stream=support.TokenStream(np.memmap(out/'pg19_tokens.u32',dtype='<u4',mode='r'),4096,8*4096)
    metadata=dict(recipe=config.RECIPE,source=data,matched_recipe_keys=list(comparable),training_bos=bos,
        evaluation_bos=151643,trainable_parameters=58195968,torch=torch.__version__,transformers=transformers.__version__,
        reference_torch=original['torch'],reference_transformers=original['transformers'],
        software_note='Same scientific settings; executed software versions are recorded, not claimed bitwise identical',
        job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),fixed_final_step=4000,
        trained_for='CacheBlend-style independent writer and sparse repair, not transferred Encbank weights')
    completed=0
    if a.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert saved['metadata']['recipe']==metadata['recipe']
        support.restore_flat(modules,saved['named']);optimizer.load_state_dict(saved['optimizer'])
        stream.cursor=saved['token_cursor'];completed=saved['step']
        support.restore_rng(saved['rng'],[torch.device('cuda:0')])
    support.atomic_json(out/'metadata.json',metadata)
    def save(step):
        payload=dict(args=vars(args),j=12,path='cacheblend_style',rank=32,alpha=32,targets=list(support.TARGETS),
            step=step,named=support.flat_state(modules),metadata=metadata)
        support.atomic_save(out/'last.pt',dict(**payload,optimizer=optimizer.state_dict(),token_cursor=stream.cursor,
            rng=support.rng_state([torch.device('cuda:0')])))
        support.export_adapter(out/('final' if step==4000 else f'step{step}'),payload,args,36)
    started=time.monotonic()
    with (out/'train.jsonl').open('a',encoding='utf-8') as log:
        for step in range(completed,4000):
            optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:g['lr']=support.lr_at(step,args)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=support.batch_loss(reader,modules,stream.take(),bos,args)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward()
            if step==0:
                assert all(p.grad is not None for p in params)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False)
            assert float(norm)>0
            optimizer.step();completed=step+1
            row=dict(step=completed,target_steps=4000,loss=float(loss.detach()),grad_norm=float(norm),
                lr=support.lr_at(step,args),elapsed_s=time.monotonic()-started,token_cursor=stream.cursor,
                training_tokens=completed*4096,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(row)+'\n');log.flush()
            support.atomic_json(out/'progress.json',row)
            if completed==1 or completed%10==0:print(json.dumps(row),flush=True)
            if completed%250==0:save(completed)
    support.atomic_json(out/'correctness_adapted.json',parity(reader,tok))
    support.atomic_json(out/'complete.json',dict(complete=True,steps=4000,training_tokens=16384000,adapter=str(out/'final')))

if __name__=='__main__':
    try:main()
    except BaseException:
        support.atomic_json(config.ROOT/f'failure-{os.environ.get("SLURM_JOB_ID","local")}.json',
            dict(phase='training',traceback=traceback.format_exc(),time=time.time()))
        raise
