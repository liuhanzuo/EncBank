"""Append-only real tool history; native incremental Dense and matched retrieval controls."""
import argparse,gc,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=Path(a.run).resolve();R.relative_to(H.resolve())
plan=json.loads((H/'agent_plan.json').read_text());arm=R.name.removeprefix('run_')
sys.path.insert(0,str(H))
from common import MODELS,load_model,load_state,tokenizer,dump,torch
from hybrid_reader import HybridReader
from memory_selectors import iter_bm25_indices
from transformers.cache_utils import DynamicCache
import transformers
def stamp():torch.cuda.synchronize();return time.perf_counter()
def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def state_digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def cache_bytes(c):
    seen=set();storage={}
    def walk(v):
        if id(v) in seen:return
        seen.add(id(v))
        if torch.is_tensor(v):
            s=v.untyped_storage();storage[(str(v.device),s.data_ptr())]=s.nbytes()
        elif isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,(tuple,list)):
            for x in v:walk(x)
        elif type(v).__module__.startswith('transformers.cache_utils'):
            walk(vars(v))
    walk(c);return sum(storage.values())
try:
    assert arm in plan['arms'] and os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);cfg=MODELS[1]
    free,total=torch.cuda.mem_get_info();cap=min(80*2**30,int(total*.9));assert free>=70*2**30
    torch.cuda.set_per_process_memory_fraction(cap/total)
    admission=dict(free=free,total=total,cap=cap,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,comm'],text=True))
    tok=tokenizer(cfg);model=load_model(cfg);assert all(p.device.type=='cuda' for p in model.parameters())
    reader=HybridReader(model,cfg['j'])
    if arm in ['raw_shared','comem']:
        reader.attach();cp=Path(plan['adapter_path']);ckpt=torch.load(cp,map_location='cpu',weights_only=False)
        assert ckpt['step']==4000;load_state(reader,ckpt,cfg);del ckpt
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==plan['adapter_sha256']
    model.requires_grad_(False)
    stop=model.generation_config.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop or tok.eos_token_id]);imend=tok.convert_tokens_to_ids('<|im_end|>')
    # Native one-turn serialization must not inject a second system message.
    def render(ms,gen):return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=gen,enable_thinking=False)
    probe=[dict(role='user',content='probe')];plain=render(probe,False);withgen=render(probe,True)
    assert withgen.startswith(plain) and plain.startswith('<|im_start|>user\n')
    generation_prefix=tok.encode(withgen[len(plain):],add_special_tokens=False);assert generation_prefix
    dev=torch.cuda.get_device_properties(0)
    dump(R/'worker_ready.json',dict(pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],host=platform.node(),gpu=dev.name,uuid=str(getattr(dev,'uuid','unavailable')),cc=[dev.major,dev.minor],torch=torch.__version__,transformers=transformers.__version__,model=cfg,arm=arm,admission=admission,stop_ids=sorted(stop),serialization='native initial prompt; append actual generated IDs, explicit im_end if absent, newline, native user observation; preserve thinking/whitespace IDs'))
    seen=set();box=R/'mailbox';last=time.monotonic();task=None;static=[];blocks=[];generated=[];bank={};bankhash={};dense_cache=None;cached_prefix=[]
    with torch.inference_mode():
      while not (box/'stop.json').exists():
        pending=sorted(p for p in box.glob('*.request.json') if p.name not in seen)
        if not pending:
            if time.monotonic()-last>1800:raise TimeoutError('actor idle timeout')
            time.sleep(.2);continue
        for req in pending:
            data=json.loads(req.read_text());gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();began=stamp()
            newtask=data['task_id']!=task
            if newtask:
                assert data['step']==0;task=data['task_id'];blocks=[];bank={};bankhash={};dense_cache=None;cached_prefix=[];generated=[]
                # Reset multimodal text-position bookkeeping between independent worlds.
                for obj in [model,model.model]:
                    if hasattr(obj,'rope_deltas'):obj.rope_deltas=None
                initial=render(data['messages'],False);assert render(data['messages'],True)==initial+withgen[len(plain):]
                static=tok.encode(initial,add_special_tokens=False)
            else:
                assert data['step']==len(blocks)+1
                observation=data['messages'][-1];assert observation['role']=='user'
                closure=[] if generated[-1]==imend else [imend]
                block=generation_prefix+generated+closure+tok.encode('\n'+render([observation],False),add_special_tokens=False)
                blocks.append(block)
            ledger=static+sum(blocks,[])+generation_prefix
            archive=sum(blocks[:-1],[]);query=(blocks[-1] if blocks else [])+generation_prefix
            static_chunks=[[static[0]]]+[static[i:i+512] for i in range(1,len(static),512)]
            history_chunks=[archive[i:i+512] for i in range(0,len(archive),512)]
            selection_start=stamp()
            search=tok.encode(data['messages'][1]['content']+'\n'+data['messages'][-1]['content'],add_special_tokens=False)
            ix=iter_bm25_indices([torch.tensor(c) for c in history_chunks],search,12,iter_hop_topk=4,iter_rounds=0) if history_chunks else []
            selection_end=stamp();fixed=static_chunks+[history_chunks[i] for i in ix];pack=sum(fixed,[])+query
            response=dict(task_id=task,step=data['step'],arm=arm,full_history_tokens=len(ledger),prompt_tokens=len(ledger) if arm=='dense' else len(pack),archived_tokens=len(archive),selected_history_chunks=ix,selected_history_tokens=sum(len(history_chunks[i]) for i in ix),selected_pack_sha256=digest(pack),query_tokens=len(query),selection_seconds=selection_end-selection_start)
            if len(ledger)>plan['max_archive_tokens'] or response['prompt_tokens']>plan['max_prompt_tokens']:
                response.update(status='context_limit',text='');dump(req.with_name(req.name.replace('.request.','.response.')),response);seen.add(req.name);last=time.monotonic();continue
            write_start=stamp();encoded=0;reused=0
            if arm=='comem':
                oldkeys=set(bank);newbank={};newhash={}
                for c in static_chunks+history_chunks:
                    k=digest(c)
                    if k in newbank:continue
                    if k in bank:
                        assert state_digest(bank[k])==bankhash[k];newbank[k]=bank[k];newhash[k]=bankhash[k];reused+=len(c)
                    else:
                        t=reader.write(c).cpu().pin_memory();newbank[k]=t;newhash[k]=state_digest(t);encoded+=len(c)
                bank,bankhash=newbank,newhash
            write_end=stamp();active_bytes=0;prefill_start=stamp()
            if arm=='dense':
                assert ledger[:len(cached_prefix)]==cached_prefix,'Append-only cache token prefix changed'
                delta=ledger[len(cached_prefix):];assert delta
                old=len(cached_prefix)
                result=model(input_ids=reader.tensor(delta),attention_mask=torch.ones((1,len(ledger)),device='cuda',dtype=torch.long),past_key_values=dense_cache,use_cache=True,logits_to_keep=1)
                dense_cache=result.past_key_values;logits=result.logits;cached_prefix=list(ledger);del result
                assert dense_cache.get_seq_length()==len(cached_prefix)
                # New incremental-hybrid path: compare the second actual prompt against native full prefill.
                if data['step']==1:
                    check=model(input_ids=reader.tensor(ledger),use_cache=False,logits_to_keep=1).logits
                    parity=dict(max_abs=float((check-logits).abs().max()),top1_equal=bool(check.argmax(-1).eq(logits.argmax(-1)).all()))
                    dump(R/(task+'.incremental_check.json'),parity);assert parity['top1_equal'] and parity['max_abs']<=.5,parity;del check
                response['cached_prefix_tokens']=old;response['new_prefill_tokens']=len(delta)
            elif arm in ['raw_base','raw_shared']:
                upper=DynamicCache(config=reader.config)
                hidden=reader.layers(reader.core.embed_tokens(reader.tensor(pack)),0,reader.L,cache=upper)
                logits=reader.logits(hidden);position=len(pack);del hidden
            else:
                lower,upper=DynamicCache(config=reader.config),DynamicCache(config=reader.config)
                states=[bank[digest(c)].to('cuda',non_blocking=True) for c in fixed]
                qh=reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lower)
                hidden=reader.layers(torch.cat(states+[qh],dim=1),reader.j,reader.L,cache=upper)
                logits=reader.logits(hidden);position=len(pack);qposition=len(query);del hidden,qh,states
            generated=[];first=None
            for step in range(plan['max_new_tokens']):
                token=int(logits[0,-1].argmax().item());generated.append(token)
                if first is None:first=stamp()
                if token in stop or step==plan['max_new_tokens']-1:break
                if arm=='dense':
                    result=model(input_ids=reader.tensor([token]),past_key_values=dense_cache,use_cache=True,logits_to_keep=1)
                    dense_cache=result.past_key_values;logits=result.logits;cached_prefix.append(token);del result
                else:
                    hidden=reader.core.embed_tokens(reader.tensor([token]))
                    if arm=='comem':
                        hidden=reader.layers(hidden,0,reader.j,cache=lower,offset=qposition);qposition+=1
                        hidden=reader.layers(hidden,reader.j,reader.L,cache=upper,offset=position)
                    else:hidden=reader.layers(hidden,0,reader.L,cache=upper,offset=position)
                    logits=reader.logits(hidden);position+=1;del hidden
            ended=stamp();active_bytes=cache_bytes(dense_cache) if arm=='dense' else cache_bytes(upper)+(cache_bytes(lower) if arm=='comem' else 0)
            response.update(status='ok',text=tok.decode(generated,skip_special_tokens=True),generated_ids=generated,generated_tokens=len(generated),hit_generation_cap=len(generated)==plan['max_new_tokens'] and generated[-1] not in stop,model_seconds=ended-prefill_start,ttft_seconds=first-prefill_start,request_seconds=ended-began,write_seconds=write_end-write_start,newly_written_tokens=encoded,reused_written_tokens=reused,persistent_H_bytes=sum(t.numel()*t.element_size() for t in bank.values()),raw_history_int64_bytes=len(ledger)*8,active_native_cache_bytes=active_bytes,decode_tokens_per_second=(len(generated)-1)/max(1e-9,ended-first),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),state_hashes_unchanged=all(state_digest(t)==bankhash[k] for k,t in bank.items()))
            assert response['state_hashes_unchanged']
            del logits
            if arm!='dense':
                del upper
                if arm=='comem':del lower
            response['post_request_allocated_bytes']=torch.cuda.memory_allocated()
            dump(req.with_name(req.name.replace('.request.','.response.')),response);seen.add(req.name);last=time.monotonic()
    dump(R/'worker_complete.json',dict(completed=True,requests=len(seen)))
except BaseException:
    dump(R/'worker_failure.json',dict(traceback=traceback.format_exc()));raise
