"""Each independently trained adapter: matched j=0 raw and own-j Encbank, fixed 400 inputs."""
import argparse,gzip,json,os,random,time
import config
import common as original
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from encbank import Encbank
from encbank.selectors import iter_bm25_indices
from eval.ruler import _string_match_all_one
from controlled_train_core import atomic_json,digest
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);a=p.parse_args()
    assert a.j in config.DEPTHS and os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(42)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**k:False
    training=config.ROOT/'training'/config.tag(a.j,42)
    complete=json.loads((training/'complete.json').read_text());assert complete['step']==4000
    ac=json.loads((training/'final/adapter_config.json').read_text())
    assert ac['layers_to_transform']==list(range(18,36)) and ac['r']==ac['lora_alpha']==32
    assert digest(training/'final/adapter_model.safetensors')==complete['adapter_sha256']
    out=config.ROOT/'quality'/config.tag(a.j,42);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=original.attach(base,training/'final');model.requires_grad_(False);tok.bos_token_id=model.config.bos_token_id
    with gzip.open(config.ROOT/'data/quality.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    assert len(rows)==400 and len({r['id'] for r in rows})==400
    readers={arm:Encbank(model,j,tokenizer=tok) for arm,j in [('raw',0),('encbank',a.j)]}
    atomic_json(out/'protocol.json',dict(j=a.j,adapter_layers=list(range(18,36)),rank=32,alpha=32,steps=4000,
        model=config.MODEL,adapter_sha256=complete['adapter_sha256'],samples_sha256=digest(config.ROOT/'data/quality.jsonl.gz'),
        arms=['raw','encbank'],rows_per_arm=400,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],
        qasper_boundary='only complete source chunks before Question marker; corrected 200 fixed inputs',
        ruler_scope='saved multikey subset, 8k/16k100 each; not full RULER macro',
        generation='original greedy natural EOS, first EOS suppressed; Qasper128 / multikey48 max tokens',
        quality_elapsed_is_not_infrastructure=True))
    seen=set();path=out/'predictions.jsonl'
    if path.exists():
        for s in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(s);key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
    assert seen<={(r['id'],arm) for r in rows for arm in readers}
    # Focused decode equality for each actual split before formal answers.
    checks={}
    for arm,reader in readers.items():
        ids=tok.encode('Alice keeps number 42 in a blue notebook. '*30+' What number? Answer:',add_special_tokens=False)
        chunks=list(torch.tensor(ids).split(64));sink=reader.write_chunk([151643]);states=[reader.write_chunk(x) for x in chunks[:-1]]
        cached=original.decode(reader,sink,states,chunks[-1].tolist(),tok.eos_token_id,8)[0]
        ref=reader._decode_from_pack(sink,states,chunks[-1].tolist(),tok.eos_token_id,8,False)
        assert cached==ref,(arm,cached,ref);checks[arm]=dict(cached_recompute_equal=True)
        del states,sink
    atomic_json(out/'correctness.json',checks)
    rng=random.Random(20260919);started=time.monotonic()
    with path.open('a',encoding='utf-8') as f:
        for r in rows:
            chunks=list(torch.tensor(r['input_ids'][:r['source_tokens']]).split(512));query=r['input_ids'][r['source_tokens']:]
            selected=iter_bm25_indices(chunks,r['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
            assert selected==r['selected'] and query
            arms=list(readers);rng.shuffle(arms)
            for arm in arms:
                if (r['id'],arm) in seen:continue
                reader=readers[arm];sink=reader.write_chunk([151643]);states=[reader.write_chunk(chunks[i]) for i in selected]
                generated=original.decode(reader,sink,states,query,tok.eos_token_id,r['budget'])[0]
                text=tok.decode(generated,skip_special_tokens=True)
                score=_string_match_all_one(text,r['answers']) if r['benchmark']=='ruler' else original.score(r,text)
                record=dict(id=r['id'],benchmark=r['benchmark'],cell=r['cell'],j=a.j,arm=arm,selected=selected,
                    budget=r['budget'],generated_ids=generated,prediction=text,score=score,status='ok',
                    adapter_sha256=complete['adapter_sha256'],source_tokens=r['source_tokens'],source_sha256=r['source_sha256'])
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();seen.add((r['id'],arm));del sink,states
                if len(seen)%10==0:atomic_json(out/'progress.json',dict(records=len(seen),target=800,elapsed_s=time.monotonic()-started))
    assert len(seen)==800
    atomic_json(out/'complete.json',dict(complete=True,records=800,errors=0,j=a.j))
if __name__=='__main__':main()
