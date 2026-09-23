"""Same 5090, same fixed inputs/packs: offline H Write and online TTFT/read-prefill."""
import argparse,gc,gzip,json,random,time
import config
import torch
from encbank import Encbank
from bm25_index import BM25Index
from controlled_train_core import atomic_json,digest
from serving import load,stamp,gpu_metadata
from torch.nn.attention import sdpa_kernel,SDPBackend
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);a=p.parse_args();assert a.j in config.DEPTHS
    out=config.ROOT/'latency'/config.tag(a.j,42);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'records.jsonl').exists(),'Retain interrupted timing, do not append unrelated sessions.'
    adapter=config.ROOT/'training'/config.tag(a.j,42)/'final'
    ac=json.loads((adapter/'adapter_config.json').read_text());assert ac['layers_to_transform']==list(range(18,36))
    wrapper,model,tok=load(adapter)
    with gzip.open(config.ROOT/'data/quality.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    rows=[r for r in rows if r['timing_subset']];assert len(rows)==15
    rng=random.Random(72019);rng.shuffle(rows)
    readers={arm:Encbank(model,j,tokenizer=tok) for arm,j in [('raw',0),('encbank',a.j)]}
    atomic_json(out/'protocol.json',dict(**gpu_metadata(),j=a.j,adapter_sha256=digest(adapter/'adapter_model.safetensors'),
        samples_sha256=digest(config.ROOT/'data/quality.jsonl.gz'),inputs=15,repeats=3,warmups_per_input_arm=1,
        read_definition='suffix prefill + LM head + first token argmax available on CPU; no decode',
        ttft_definition='online BM25 scoring -> selected evidence fetch -> query Write -> read-prefill -> first CPU token',
        write_definition='full document H and sink creation + CPU-pinned store; index construction separately',
        exclusions='network, tokenization, model/adapter load; online excludes document H/index preparation',
        quality_output=False,cpu_threads=2))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]),(out/'records.jsonl').open('w') as f:
        for row in rows:
            chunks=list(torch.tensor(row['input_ids'][:row['source_tokens']]).split(512));query=row['input_ids'][row['source_tokens']:]
            start=time.perf_counter();index=BM25Index(chunks);index_ms=(time.perf_counter()-start)*1000
            assert index.select(row['question_ids'],12,4)==row['selected']
            for rep in [-1,0,1,2]:
                start=stamp();bank=[readers['encbank'].write_chunk(ch).cpu().pin_memory() for ch in chunks]
                sink_bank=readers['encbank'].write_chunk([151643]).cpu().pin_memory();write_ms=(stamp()-start)*1000
                cache_bytes=sum(h.numel()*h.element_size() for h in [sink_bank,*bank])
                arms=list(readers);rng.shuffle(arms)
                for arm in arms:
                    reader=readers[arm];start=stamp()
                    selected=index.select(row['question_ids'],12,4);retrieved=stamp();assert selected==row['selected']
                    if arm=='encbank':
                        states=[bank[i].to('cuda',non_blocking=True) for i in selected];sink=sink_bank.to('cuda',non_blocking=True)
                    else:
                        states=[reader.write_chunk(chunks[i]) for i in selected];sink=reader.write_chunk([151643])
                    fetched=stamp();qh,bc,qp=reader.write_prefill(query);qw=stamp()
                    logits,tc,pp=reader.read_prefill(sink,states,qh)
                    lg=logits[0,-1].float();lg[tok.eos_token_id]=float('-inf');token=int(lg.argmax().item());first=stamp()
                    record=dict(id=row['id'],cell=row['cell'],source_sha256=row['source_sha256'],j=a.j,arm=arm,rep=rep,warmup=rep<0,
                        selected=selected,source_tokens=row['source_tokens'],pack_positions=1+sum(len(chunks[i]) for i in selected)+len(query),
                        write_ms=write_ms if arm=='encbank' else 0.,H_bytes=cache_bytes if arm=='encbank' else 0,index_build_ms=index_ms,
                        retrieval_ms=(retrieved-start)*1000,fetch_ms=(fetched-retrieved)*1000,query_write_ms=(qw-fetched)*1000,
                        read_ms=(first-qw)*1000,ttft_ms=(first-start)*1000,first_token=token,status='ok')
                    f.write(json.dumps(record)+'\n');f.flush()
                    del states,sink,qh,bc,tc,logits,lg
                del bank,sink_bank
            atomic_json(out/'progress.json',dict(last=row['id'],j=a.j))
    atomic_json(out/'complete.json',dict(complete=True,records=120,formal_records=90,errors=0))
if __name__=='__main__':main()
