"""Evaluate final checkpoint and original references on identical fixed supports."""
import argparse, contextlib, gc, gzip, json, os, time
import config
import common as original
from comem import CoMem
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_support import atomic_json, digest


@torch.inference_mode()
def run(arm, name, mode, path):
    out = config.ROOT/'evaluation'/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        return
    torch.set_num_threads(2); torch.manual_seed(42)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    tok = AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(path) if mode=='full' else config.MODEL,
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper = None
    if mode == 'lora':
        wrapper, model = original.attach(base,path)
    else:
        model = base
    model.requires_grad_(False)
    tok.bos_token_id = model.config.bos_token_id
    assert tok.bos_token_id == 151643
    reader = CoMem(model,0 if mode=='replay' else 12,tokenizer=tok)
    # Catch evaluation-cache errors on the actual trained model before reporting scores.
    ids = tok.encode('Alice keeps the number 42 in a blue notebook. '*30+' What number? Answer:',add_special_tokens=True)
    chunks = list(torch.tensor(ids).split(64))
    sink = reader.write_chunk([151643]); states = [reader.write_chunk(c) for c in chunks[:-1]]
    actual,_,_ = original.decode(reader,sink,states,chunks[-1].tolist(),tok.eos_token_id,8)
    expected = reader._decode_from_pack(sink,states,chunks[-1].tolist(),tok.eos_token_id,8,False)
    assert actual == expected, (actual,expected)
    del sink,states
    atomic_json(out/'correctness.json',dict(cached_decode_matches_recompute=True,generated_ids=actual))
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    assert len(rows)==600 and len({r['id'] for r in rows})==600
    atomic_json(out/'protocol.json',dict(arm=arm,name=name,mode=mode,checkpoint=str(path),samples_sha256=digest(config.SAMPLES),
        j=reader.resume_j,budget=16,bos=151643,eos=tok.eos_token_id,selection='saved iter-BM25 top12,512-token chunks,source order',
        greedy=True,chat_template=False,first_token_eos_suppressed=True,overlap=0,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),timing_is_infrastructure_result=False))
    file = out/'predictions.jsonl'; seen = set()
    if file.exists():
        for line in file.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); assert r['id'] not in seen; seen.add(r['id'])
    assert seen <= {r['id'] for r in rows}
    started=time.monotonic()
    with file.open('a',encoding='utf-8') as f:
        for row in rows:
            if row['id'] in seen:
                continue
            chunks,query = original.parts(row); selected = original.selection(chunks,row)
            sink = reader.write_chunk([151643]); states = [reader.write_chunk(chunks[i]) for i in selected]
            generated,_,_ = original.decode(reader,sink,states,query,tok.eos_token_id,16)
            text = tok.decode(generated,skip_special_tokens=True)
            rec=dict(id=row['id'],split=row['split'],cell=row['cell'],arm=name,generated_ids=generated,
                     prediction=text,score=original.score(row,text),selected=selected,budget=16,status='ok')
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush(); seen.add(row['id'])
            del sink,states
            atomic_json(out/'progress.json',dict(records=len(seen),target_records=600,last_id=row['id'],elapsed_s=time.monotonic()-started))
    assert len(seen)==600
    atomic_json(out/'complete.json',dict(complete=True,records=600))
    del reader,model,base,wrapper
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',type=int,required=True); args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    arm=config.ARMS[args.arm]
    training=config.ROOT/'runs'/arm['name']/'training'
    assert json.loads((training/'status.json').read_text())['complete']
    run(arm,arm['name'],arm['mode'],training/'final')
    if arm['name']=='lora32_4000':
        run(None,'original_lora32_4000','lora',config.PRINCIPAL)
        run(None,'frozen_j12','frozen',config.MODEL)
        run(None,'selected_text_replay','replay',config.MODEL)


if __name__=='__main__':
    main()
