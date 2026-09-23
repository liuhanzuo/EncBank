"""Paired quality evaluation; same supports, budgets and scoring at every depth."""
import argparse,contextlib,gc,gzip,json,os,time
import config
import common as original
import torch
from comem import CoMem
from transformers import AutoModelForCausalLM,AutoTokenizer
from controlled_train_core import atomic_json,digest

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();assert a.j in config.DEPTHS and a.seed in config.SEEDS
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    out=config.ROOT/'quality'/config.tag(a.j,a.seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    torch.set_num_threads(2);torch.manual_seed(42)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    training=config.ROOT/'training'/config.tag(a.j,a.seed)
    assert json.loads((training/'complete.json').read_text())['step']==4000
    cfg=json.loads((training/'final/adapter_config.json').read_text())
    assert cfg['layers_to_transform']==list(range(24,36)) and cfg['r']==32
    wrapper,model=original.attach(base,training/'final');model.requires_grad_(False)
    tok.bos_token_id=model.config.bos_token_id;assert tok.bos_token_id==151643
    with gzip.open(config.ROOT/'data/quality.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    assert len(rows)==800 and len({r['id'] for r in rows})==800
    arms=['adapted','selected_replay']+(['frozen'] if a.seed==42 else [])
    atomic_json(out/'protocol.json',dict(j=a.j,seed=a.seed,arms=arms,rows=800,adapter_layers=list(range(24,36)),
        gpu=torch.cuda.get_device_name(),samples_sha256=digest(config.ROOT/'data/quality.jsonl.gz'),
        same_saved_retrieval=True,first_eos_suppressed=True,overlap=0,quality_elapsed_is_not_infra=True))
    for arm in arms:
        dest=out/arm;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        reader=CoMem(model,0 if arm=='selected_replay' else a.j,tokenizer=tok)
        cm=wrapper.disable_adapter() if arm=='frozen' else contextlib.nullcontext()
        with cm:
            ids=tok.encode('Alice keeps number 42 in a blue notebook. '*30+' What number? Answer:',add_special_tokens=False)
            ch=list(torch.tensor(ids).split(64));sink=reader.write_chunk([151643]);hs=[reader.write_chunk(x) for x in ch[:-1]]
            actual=original.decode(reader,sink,hs,ch[-1].tolist(),tok.eos_token_id,8)[0]
            expected=reader._decode_from_pack(sink,hs,ch[-1].tolist(),tok.eos_token_id,8,False)
            assert actual==expected,(arm,actual,expected)
            atomic_json(dest/'correctness.json',dict(cached_recompute_equal=True,ids=actual))
            del sink,hs
            path=dest/'predictions.jsonl';seen=set()
            if path.exists():
                for line in path.read_text().splitlines():
                    row=json.loads(line);assert row['id'] not in seen;seen.add(row['id'])
            assert seen<={r['id'] for r in rows}
            start=time.monotonic()
            with path.open('a',encoding='utf-8') as f:
                for r in rows:
                    if r['id'] in seen:continue
                    chunks,query=original.parts(r);selected=original.selection(chunks,r)
                    sink=reader.write_chunk([151643]);hs=[reader.write_chunk(chunks[i]) for i in selected]
                    budget=int(r['budget']);assert budget>0
                    generated=original.decode(reader,sink,hs,query,tok.eos_token_id,budget)[0]
                    text=tok.decode(generated,skip_special_tokens=True)
                    record=dict(id=r['id'],benchmark=r['benchmark'],split=r.get('split','original'),cell=r['cell'],
                        j=a.j,seed=a.seed,arm=arm,budget=budget,selected=selected,generated_ids=generated,
                        prediction=text,score=original.score(r,text),status='ok')
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();seen.add(r['id']);del sink,hs
                    atomic_json(out/'progress.json',dict(arm=arm,records=len(seen),target=800,elapsed_s=time.monotonic()-start))
            assert len(seen)==800
            atomic_json(dest/'complete.json',dict(complete=True,records=800))
    atomic_json(out/'complete.json',dict(complete=True,arms=arms,records=800*len(arms)))

if __name__=='__main__':main()
