"""Full paired quality evaluation after the fixed 4,000-step adapter completes."""
import argparse, collections, gc, gzip, json, os, time, traceback
import torch
from common import *
from hybrid_reader import HybridReader
from evaluation_scope import read_scope, shard_count, results_directory
from eval import ruler, longeval, longbench, locomo
import prepare_babilong as babi

def scores(row,text):
    b=row['benchmark']
    if b=='ruler': return dict(score=ruler._string_match_all_one(text,row['answers']))
    if b=='longeval': return dict(score=float(longeval.extract_prediction(text)==row['answers'][0]))
    if b=='longbench': return dict(score=longbench.compute_f1_multi(text,row['answers']))
    if b=='babilong': return dict(score=babi.score_prediction(text,row['extra'],row['task']))
    sc=locomo.score_sample(dict(pred=text,answers=row['answers'],**row['extra']))
    # Lexical F1 is a diagnostic, never a substitute for the Table1 Judge column.
    return dict(score=None, lexical=sc, judge_score=sc['acc'] if row['extra']['is_abstention'] else None,
                judge_status='local_abstention' if row['extra']['is_abstention'] else 'pending')

@torch.no_grad()
def dense(model,reader,ids,budget,eos):
    x=reader.tensor(ids)
    generated=model.generate(input_ids=x,max_new_tokens=budget,do_sample=False,
        pad_token_id=min(eos),eos_token_id=sorted(eos),use_cache=True)
    return generated[0,len(ids):].tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-index',type=int,required=True)
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); cfg=MODELS[args.model_index];scope=read_scope()
    out=ROOT/('evaluation_smoke' if args.smoke else results_directory())/cfg['name']/f'shard{args.shard}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        assert json.loads((out/'complete.json').read_text()).get('scope_id')==scope['scope_id'], 'Completion belongs to a different scope'
        print('Active-scope shard already complete.'); return
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()).get('scope',{}).get('scope_id')==scope['scope_id'], 'Do not append across evaluation scopes'
    try:
        seed_all(); tok=tokenizer(cfg); model=load_model(cfg)
        reader=HybridReader(model,cfg['j']); reader.attach()
        if not args.smoke:
            training=ROOT/'training'/cfg['name']
            assert json.loads((training/'complete.json').read_text())['steps']==STEPS
            ckpt=torch.load(training/'adapter-final.pt',map_location='cpu',weights_only=False)
            assert ckpt['step']==STEPS; load_state(reader,ckpt,cfg); del ckpt
        reader.model.requires_grad_(False)
        dump(out/'correctness.json',reader.validate(tok))
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos if eos is not None else tok.eos_token_id])
        prep=ROOT/('sample_smoke' if args.smoke else 'samples')/cfg['name']
        spec=json.loads((prep/'complete.json').read_text()); assert spec['complete']
        expected=shard_count(spec,args.shard,scope)
        dump(out/'protocol.json',dict(**metadata(cfg),shard=args.shard,arms=ARMS,expected_samples=expected,
            trained_steps=0 if args.smoke else STEPS,samples=spec,scope=scope,stop_ids=sorted(eos),
            streaming_sink=4,streaming_recent_window=6653,
            hcache='same split, independent chunks, no retrieval, no adapter; mechanism reference, not native HCache',
            kvdirect='stock full-source forward and decode; no retrieval, adapter or artificial context extension'))
        seen=set()
        file=out/'predictions.jsonl'
        if file.exists():
            with file.open(encoding='utf-8') as f:
                for line in f:
                    rec=json.loads(line)
                    assert rec['benchmark'] in scope['benchmarks'], 'Existing predictions outside active scope'
                    key=(rec['id'],rec['arm']); assert key not in seen; seen.add(key)
        started=time.monotonic(); sample_count=0
        with gzip.open(prep/f'shard{args.shard}.jsonl.gz','rt',encoding='utf-8') as inputs, file.open('a',encoding='utf-8') as outputs:
            for line in inputs:
                row=json.loads(line)
                if row['benchmark'] not in scope['benchmarks']:continue
                sample_count+=1
                for arm in ARMS:
                    if (row['id'],arm) in seen: continue
                    t0=time.monotonic(); generated=None
                    record=dict(id=row['id'],arm=arm,benchmark=row['benchmark'],task=row['task'],
                                length=row['length'],index=row['index'],answers=row['answers'],extra=row['extra'],budget=row['budget'])
                    try:
                        enabled=arm in ('cache_lora','replay_shared_lora')
                        with reader.adapter(enabled),torch.inference_mode():
                            if arm in ('kvdirect','streamingllm'):
                                ids=row['input_ids']
                                if arm=='streamingllm' and len(ids)>6657: ids=ids[:4]+ids[-6653:]
                                generated=dense(model,reader,ids,row['budget'],eos)
                            elif arm=='hcache_style':
                                ids=row['input_ids']; chunks=[ids[i:i+CHUNK] for i in range(0,len(ids),CHUNK)]
                                generated=reader.generate([[row['sink']]]+chunks,'cache',row['budget'],eos)
                            else:
                                mode='replay' if arm.startswith('replay_') else 'cache'
                                generated=reader.generate(row['segments'],mode,row['budget'],eos)
                        text=tok.decode(generated,skip_special_tokens=True)
                        record.update(status='ok',generated_ids=generated,text=text,**scores(row,text),
                                      reached_cap_without_stop=len(generated)==row['budget'] and generated[-1] not in eos)
                    except torch.OutOfMemoryError:
                        gc.collect(); torch.cuda.empty_cache()
                        record.update(status='OOM',generated_ids=None,text=None,score=None,
                                      failure='CUDA allocation failed; no truncation/offload/zero-score substitution')
                    record['elapsed_s_nonbenchmark']=time.monotonic()-t0
                    outputs.write(json.dumps(record,ensure_ascii=False)+'\n');outputs.flush()
                    seen.add((row['id'],arm))
                    dump(out/'progress.json',dict(phase='evaluation',records=len(seen),target_records=expected*len(ARMS),
                        last_id=row['id'],last_arm=arm,last_status=record['status'],elapsed_s=time.monotonic()-started))
                if sample_count%5==0: print(json.dumps(dict(samples=sample_count,expected=expected,records=len(seen))),flush=True)
        assert sample_count==expected and len(seen)==expected*len(ARMS)
        dump(out/'complete.json',dict(generation_complete=True,samples=sample_count,records=len(seen),
             model=cfg,shard=args.shard,scope_id=scope['scope_id'],benchmarks=scope['benchmarks'],
             deferred_benchmarks=scope['deferred'],judge_required=scope['judge_required'],verification_pending=True))
        dump(out/'progress.json',dict(phase='generation_complete',records=len(seen),target_records=len(seen),
             scope_id=scope['scope_id'],deferred_benchmarks=scope['deferred']))
    except BaseException:
        dump(out/('failure-'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__': main()
