"""Four freshly generated paired arms using the unchanged 500-example protocol."""
import contextlib,gzip,json,os,sys,time,traceback
from unittest.mock import patch
import config
sys.path.insert(0,str(config.FOLLOW))
import common as original
from kv_control import KVControl,assemble,generate,checks
from encbank import Encbank
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
ARMS=('cacheblend_frozen','cacheblend_own_lora','encbank_frozen','encbank_own_lora')

@torch.inference_mode()
def main():
    out=config.ROOT/'evaluation';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('Evaluation already complete');return
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert json.loads((config.ROOT/'training/complete.json').read_text())['steps']==4000
    torch.set_num_threads(2);torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):
        wrapper=PeftModel.from_pretrained(base,str(config.ROOT/'training/final'),adapter_name='cacheblend',autocast_adapter_dtype=True)
        wrapper.load_adapter(config.PRINCIPAL,adapter_name='encbank',is_trainable=False)
    wrapper.requires_grad_(False);model=wrapper.base_model.model.eval()
    tok.bos_token_id=model.config.bos_token_id;assert tok.bos_token_id==151643
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    assert len(rows)==500 and len({r['id'] for r in rows})==500
    expected={'longeval_8k':(100,16),'longeval_16k':(100,16),'longeval_32k':(100,16),'qasper':(200,128)}
    for c,(n,b) in expected.items():assert sum(r['cell']==c and r['budget']==b for r in rows)==n
    wrapper.set_adapter('cacheblend')
    original.dump(out/'correctness_cacheblend.json',checks(model,tok))
    original.dump(out/'protocol.json',dict(arms=ARMS,samples=500,records=2000,bos=151643,eos=tok.eos_token_id,
        budget_by_cell={c:b for c,(n,b) in expected.items()},chat_template=False,greedy=True,first_token_eos_suppressed=True,
        source_samples=str(config.SAMPLES),selection='exact saved iter-BM25 top12 / 512-token chunks / source order',
        reference='encbank_followups_20260913/run_kv_quality.py; all four arms freshly generated on the same model instance',
        own_adapter=str(config.ROOT/'training/final'),encbank_adapter=config.PRINCIPAL,
        recompute_ratio=.15,bootstrap_layers=2,timing_is_infrastructure_result=False))
    seen=set();file=out/'predictions.jsonl'
    if file.exists():
        for line in file.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
    assert seen.issubset({(r['id'],a) for r in rows for a in ARMS})
    cm=Encbank(model,12,tokenizer=tok);cb=KVControl(Encbank(model,0,tokenizer=tok))
    started=time.monotonic()
    with file.open('a',encoding='utf-8') as f:
        for row in rows:
            chunks,query=original.parts(row);ix=original.selection(chunks,row)
            for arm in ARMS:
                if (row['id'],arm) in seen:continue
                is_kv=arm.startswith('cacheblend')
                wrapper.set_adapter('cacheblend' if is_kv else 'encbank')
                with wrapper.disable_adapter() if arm.endswith('frozen') else contextlib.nullcontext():
                    if is_kv:
                        pack,merged=assemble(cb,[chunks[i] for i in ix],query,tok.bos_token_id)
                        ids,_,_,stats=generate(cb,pack,merged,len(query),tok.eos_token_id,row['budget'])
                        del pack,merged
                    else:
                        states=[cm.write_chunk(chunks[i]) for i in ix];sink=cm.write_chunk([tok.bos_token_id])
                        ids,_,_=original.decode(cm,sink,states,query,tok.eos_token_id,row['budget']);stats={}
                        del states,sink
                text=tok.decode(ids,skip_special_tokens=True)
                rec=dict(id=row['id'],cell=row['cell'],arm=arm,generated_ids=ids,prediction=text,
                    score=original.score(row,text),selected=ix,budget=row['budget'],status='ok',stats=stats)
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();seen.add((row['id'],arm))
                original.dump(out/'progress.json',dict(records=len(seen),target_records=2000,last_id=row['id'],last_arm=arm,elapsed_s=time.monotonic()-started))
    assert len(seen)==2000
    original.dump(out/'complete.json',dict(complete=True,samples=500,records=2000))
if __name__=='__main__':
    try:main()
    except BaseException:
        original.dump(config.ROOT/f'failure-eval-{os.environ.get("SLURM_JOB_ID","local")}.json',
            dict(phase='evaluation',traceback=traceback.format_exc(),time=time.time()))
        raise
