"""Single GPU process: alternating same-BGE store-ready TTFT and paired QA."""
import argparse,gc,gzip,hashlib,json,random,sys,time
import config
import common as original
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from comem.selectors import DenseBGERetriever
from authorized_admission import acquire_gpu

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def stamp():torch.cuda.synchronize();return time.perf_counter()

@torch.inference_mode()
def forward(reader,sink,states,query,eos,budget):
    q,bottom,qpos=reader.write_prefill(query);logits,top,ppos=reader.read_prefill(sink,states,q)
    lg=logits[0,-1].float();lg[eos]=float('-inf');token=int(lg.argmax().item());ids=[token];first=stamp()
    terminal_eos=False
    for _ in range(1,budget):
        logits=reader.decode_step(token,bottom,top,qpos,ppos);qpos+=1;ppos+=1
        token=int(logits[0,-1].float().argmax().item())
        if token==eos:terminal_eos=True;break
        ids.append(token)
    end=stamp();return ids,first,end,terminal_eos

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','extended','quality'],required=True)
    p.add_argument('--process',type=int,default=0);a=p.parse_args()
    dest=config.ROOT/a.phase/f'process_{a.process:02d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    assert not (dest/'records.jsonl').exists(),'Interrupted timing process: use a fresh named attempt, never append measurements blindly.'
    spec=json.loads((config.ROOT/'data/complete.json').read_text());assert spec['complete']
    assert json.loads((config.ROOT/'existing_analysis/generation_audit.json').read_text())['complete']
    if a.phase=='quality':
        decision=json.loads((config.ROOT/'budget_decision.json').read_text());ks=decision['quality_raw_ks']
        assert decision['frozen_before_quality']
    else:ks=config.KS if a.phase=='calibration' else config.EXTENDED
    name='evaluation' if a.phase=='quality' else 'calibration'
    with gzip.open(config.ROOT/'data'/f'{name}.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    assert len(rows)==(200 if a.phase=='quality' else 20)
    rankings={r['id']:r for r in json.loads((config.ROOT/'data/rankings.json').read_text())}
    torch.set_num_threads(2);torch.set_num_interop_threads(16);torch.manual_seed(42)
    admission=acquire_gpu(cap_gb=27*2**30/1e9,poll=10,max_wait=21600,tag='comem_bge_matched_TTFT')
    assert '5090' in torch.cuda.get_device_name()
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=original.attach(base,config.ADAPTER);model.requires_grad_(False)
    tok.bos_token_id=model.config.bos_token_id;assert tok.bos_token_id==151643
    readers=dict(raw=original.CoMem(model,0,tokenizer=tok),comem=original.CoMem(model,12,tokenizer=tok))
    encoder=DenseBGERetriever(str(config.BGE),device='cpu',dtype=torch.float32,batch_size=8)
    adapter_sha=hashlib.sha256((config.ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest();assert adapter_sha==spec['adapter_sha256']
    ac=json.loads((config.ADAPTER/'adapter_config.json').read_text());assert ac['r']==ac['lora_alpha']==32
    metadata=dict(phase=a.phase,process=a.process,model_revision=spec['model_revision'],adapter_sha256=adapter_sha,
        bge_revision=spec['bge_revision'],bge_device='cpu',bge_dtype='float32',pooling='CLS+L2',cpu_threads=2,
        gpu=torch.cuda.get_device_name(),gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
        admission=admission,cap_bytes=27*2**30,raw_ks=ks,comem_k=12,torch=torch.__version__,
        budget=128 if a.phase=='quality' else 1,greedy=True,EOS=tok.eos_token_id,first_eos_suppressed=True,min_new_tokens=None,
        ttft_boundary='Pretokenized bare query IDs -> real BGE query encoding/search -> source-order selection -> fetch/H2D -> sink/query Write -> prefill -> first token ID on CPU',
        excluded='document H Write and BGE index build (separately recorded); model/tokenizer load; input tokenization; external IO/network; decode',
        document_bank_scope='Complete 512-token chunks before Question marker; remaining prompt suffix processed online. Original full prompt/token IDs unchanged. No query text precomputed.')
    dump(dest/'metadata.json',metadata)
    rng=random.Random(config.SEED+a.process);rng.shuffle(rows);n=0;started=time.perf_counter()
    # Warm up the CPU query encoder before measuring; real online calls remain inside every timed region.
    encoder.encode(['warmup query'],is_query=True)
    with (dest/'records.jsonl').open('w',encoding='utf-8') as f,(dest/'write.jsonl').open('w') as wf:
        for row in rows:
            chunks=list(torch.tensor(row['input_ids'][:row['source_tokens']]).split(512))
            query=row['input_ids'][row['source_tokens']:];key=row['source_sha256']
            vectors=torch.load(config.ROOT/'data/indexes'/(key+'.pt'),map_location='cpu',weights_only=True)
            assert vectors.dtype==torch.float32 and len(vectors)==len(chunks)
            t=stamp();bank=[readers['comem'].write_chunk(c).cpu().pin_memory() for c in chunks];write_s=stamp()-t
            wf.write(json.dumps(dict(id=row['id'],source_sha256=key,write_s=write_s,
                H_bytes=sum(h.numel()*h.element_size() for h in bank),index_bytes=vectors.numel()*vectors.element_size()))+'\n');wf.flush()
            arms=[('comem',12)]+[('raw',k) for k in ks]
            # Calibration: each input/config has one excluded warmup and three formal repetitions.
            # Quality: one first-token warmup per arm, then exactly one full answer.
            for rep in ([-1,0] if a.phase=='quality' else [-1,0,1,2]):
                order=list(arms);rng.shuffle(order)
                for method,k in order:
                    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    reader=readers[method];begin=stamp()
                    qtext=tok.decode(row['question_ids'],skip_special_tokens=True)
                    vec=encoder.encode([qtext],is_query=True)
                    scores=(vectors@vec[0]).tolist();ranking=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
                    selected=sorted(ranking[:k]);search_end=time.perf_counter()
                    assert ranking==rankings[row['id']]['ranking']
                    states=[bank[i].to('cuda',non_blocking=True) if method=='comem' else reader.write_chunk(chunks[i]) for i in selected]
                    sink=reader.write_chunk([model.config.bos_token_id])
                    budget=128 if a.phase=='quality' and rep>=0 else 1
                    ids,first,end,stopped=forward(reader,sink,states,query,tok.eos_token_id,budget)
                    text=tok.decode(ids,skip_special_tokens=True)
                    rec=dict(id=row['id'],context_sha256=row['context_sha256'],source_sha256=key,
                        phase=a.phase,process=a.process,rep=rep,warmup=rep<0,method=method,k_max=k,selected=selected,
                        pack_tokens=1+sum(len(chunks[i]) for i in selected)+len(query),source_chunks=len(chunks),
                        ttft_ms=(first-begin)*1000,online_ms=(end-begin)*1000,query_embedding_search_ms=(search_end-begin)*1000,
                        generated_ids=ids,generated_tokens=len(ids),prediction=text,terminal_eos=stopped,
                        length_cap_reached=budget==128 and len(ids)==128 and not stopped,status='ok',
                        score=original.score(row,text) if a.phase=='quality' and rep>=0 else None,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    assert rec['peak_reserved_bytes']<=27*2**30
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();n+=1
                    del states,sink
                    dump(dest/'progress.json',dict(records=n,last_id=row['id'],method=method,k=k,rep=rep,elapsed_s=time.perf_counter()-started))
            del bank,vectors;gc.collect();torch.cuda.empty_cache()
    expected=len(rows)*len(arms)*(2 if a.phase=='quality' else 4)
    assert n==expected
    dump(dest/'complete.json',dict(complete=True,records=n,formal_records=len(rows)*len(arms)*(1 if a.phase=='quality' else 3),
        elapsed_s=time.perf_counter()-started))

if __name__=='__main__':main()
