"""New precision arms on identical old prompt-packs; no fixture regeneration."""
import argparse, collections, gc, gzip, json, os, time, traceback
from pathlib import Path
import torch
from common import ROOT, MODELS, dump, seed_all, tokenizer, load_model, load_state, metadata
from binding import OLD, verify_model, verify_code, sha256, inference_sha, old_predictions
from hybrid_reader import HybridReader
from packed_reader import split_fixture, encode_entries, materialize, generate
from scoring import scores

CELL = ('ruler','niah_single_2','8k')

def in_cell(row): return tuple(row[k] for k in ('benchmark','task','length')) == CELL

def rows_for(cfg, mode, shard):
    for s in range(4) if mode == 'quant-cell' else [shard]:
        path = OLD/'samples'/cfg['name']/f'shard{s}.jsonl.gz'
        expected = json.loads((ROOT/'verified_inputs.json').read_text())['models'][cfg['name']]['samples'][path.name]
        assert sha256(path) == expected
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if mode == 'quant-cell' and not in_cell(row): continue
                if mode == 'quant-full' and in_cell(row): continue
                if mode == 'large-mid' and row['benchmark'] != 'longeval': continue
                yield row

def destination(mode, cfg, shard):
    return ROOT/'results'/mode/cfg['name']/('cell' if mode=='quant-cell' else f'shard{shard}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-index', type=int, required=True)
    ap.add_argument('--shard', type=int, default=0, choices=range(4))
    ap.add_argument('--mode', choices=['quant-cell','quant-full','large-final','large-mid'], required=True)
    args=ap.parse_args();cfg=MODELS[args.model_index];out=destination(args.mode,cfg,args.shard)
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists(): print('Already complete, no duplicate.');return
    verify_code(); frozen=verify_model(cfg)
    if args.mode=='quant-full':
        gate=json.loads((destination('quant-cell',cfg,0)/'complete.json').read_text())
        assert gate['samples']==100 and gate['h16_parity_passed']
    seed_all();tok=tokenizer(cfg);model=load_model(cfg)
    large=args.mode.startswith('large-')
    adapter=(ROOT/'training'/cfg['name']/('adapter-step4000.pt' if args.mode=='large-mid' else 'adapter-final.pt')) if large else OLD/'training'/cfg['name']/'adapter-final.pt'
    if large and args.mode=='large-final':assert json.loads((adapter.parent/'complete.json').read_text())['steps']==8000
    ckpt=torch.load(adapter,map_location='cpu',weights_only=False)
    step=4000 if args.mode=='large-mid' or not large else 8000
    rank=128 if large else 32
    assert ckpt['step']==step and ckpt['rank']==rank and ckpt['alpha']==rank
    reader=HybridReader(model,cfg['j']);reader.attach(rank=rank,alpha=rank)
    load_state(reader,ckpt,cfg);del ckpt;model.requires_grad_(False)
    binding=f"{cfg['name']}:{cfg['revision']}:j{cfg['j']}:adapter:{sha256(adapter)}"
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos if eos is not None else tok.eos_token_id])
    old=old_predictions(cfg) if not large else None
    bits=[16,8,4] if args.mode=='quant-cell' else [8,4]
    arms=[f'cache_h{b}' for b in bits] if not large else [f'cache_r128_s{step}_h16']
    protocol=dict(**metadata(cfg),mode=args.mode,shard=args.shard,arms=arms,binding=binding,
                  adapter_sha256=sha256(adapter),rank=rank,alpha=rank,step=step,
                  protocol_id='newbackbones-prompt-pack-hprecision-v1',
                  inference='unchanged native chat/no-thinking; unchanged selected indices and token IDs',
                  store='all source chunks packed; native sink; query and online cache unquantized',
                  boundary='Source excludes only final 512-token chunk; may contain part of question. Not independent-document protocol.',
                  store_measurement='logical exact bytes per entry; concurrent precision entries share one quality worker; no per-arm peak or formal latency claim',
                  stop_ids=sorted(eos),native_dtype='bfloat16',group_size=64,
                  model_class=type(model).__name__,core_class=type(reader.core).__name__,
                  request_cache='transformers.cache_utils.DynamicCache; native attention and DeltaNet states')
    if (out/'protocol.json').exists():
        prev=json.loads((out/'protocol.json').read_text())
        for key in ('mode','shard','arms','binding','protocol_id'):assert prev[key]==protocol[key]
    dump(out/'protocol.json',protocol)
    seen={};file=out/'predictions.jsonl'
    if file.exists():
        for line in file.open(encoding='utf-8'):
            rec=json.loads(line);key=(rec['id'],rec['arm']);assert key not in seen;seen[key]=rec
    samples=0;expected_keys=set();parity=sum(r.get('h16_parity_passed',False) for r in seen.values());started=time.monotonic();audit_cells=set()
    with file.open('a',encoding='utf-8') as outputs:
        for row in rows_for(cfg,args.mode,args.shard):
            samples+=1;digest=inference_sha(row)
            for arm in arms:expected_keys.add((row['id'],arm))
            missing=[arm for arm in arms if (row['id'],arm) not in seen]
            if not missing:
                if args.mode=='quant-cell':assert seen[(row['id'],'cache_h16')]['h16_parity_passed']
                continue
            source,query,selected=split_fixture(row)
            cell=tuple(row[k] for k in ('benchmark','task','length'))
            audit=args.mode=='quant-cell' or cell not in audit_cells
            entries={};encode_error=None;t0=time.monotonic()
            try:
                if not large:entries=encode_entries(reader,source,row['sink'],[b for b in bits if f'cache_h{b}' in missing],binding)
            except torch.OutOfMemoryError:
                encode_error='OOM during full-source Write; no truncation/offload'
                gc.collect();torch.cuda.empty_cache()
            write_s=time.monotonic()-t0
            try:
                for arm in missing:
                    t0=time.monotonic();bit=int(arm[7:]) if not large else 16
                    rec={k:row[k] for k in ('id','benchmark','task','length','index','answers','extra','budget')}
                    rec.update(arm=arm,bits=bit,rank=rank,step=step,shard=row['sequence']%4,
                               inference_sha256=digest,selected_indices=selected,
                               source_tokens=row['source_tokens'],query_tokens=row['query_tokens'],binding=binding)
                    try:
                        if encode_error:raise torch.OutOfMemoryError(encode_error)
                        if large:
                            with torch.no_grad():ids=reader.generate(row['segments'],'cache',row['budget'],eos)
                        else:
                            entry=entries[bit];before=entry.digest() if audit else None
                            rec['store']=entry.inventory()
                            if bit==16:
                                # Compare actual BF16 values, not merely aggregate scores.
                                with torch.no_grad():
                                    native=torch.cat([reader.write(s) for s in row['segments'][:-1]],dim=1)
                                    restored=materialize(entry,selected,binding)
                                    assert torch.equal(native,restored),'H16 hidden mismatch'
                                    del native,restored
                            ids=generate(reader,entry,selected,query,row['budget'],eos,binding)
                            if audit:
                                assert before==entry.digest(),'Persistent entry hash changed'
                                rec['entry_hash_before']=before;rec['entry_hash_after']=before
                            if cell not in audit_cells:
                                repeated=generate(reader,entry,selected,query,row['budget'],eos,binding)
                                assert repeated==ids and before==entry.digest()
                                rec['same_entry_repeated_read_equal']=True
                            if bit==16:
                                previous=old[row['id']]
                                assert previous['status']=='ok' and ids==previous['generated_ids'],f"H16 native-output mismatch: {row['id']}"
                                rec['h16_parity_passed']=True;parity+=1
                            rec['shared_full_source_write_s_nonbenchmark']=write_s
                        text=tok.decode(ids,skip_special_tokens=True)
                        rec.update(status='ok',generated_ids=ids,text=text,**scores(row,text),
                                   reached_cap_without_stop=len(ids)==row['budget'] and ids[-1] not in eos)
                    except torch.OutOfMemoryError as exc:
                        rec.update(status='OOM',generated_ids=None,text=None,score=None,failure=str(exc)[:300])
                        gc.collect();torch.cuda.empty_cache()
                    rec['elapsed_s_nonbenchmark']=time.monotonic()-t0
                    outputs.write(json.dumps(rec,ensure_ascii=False)+'\n');outputs.flush();seen[(row['id'],arm)]=rec
                    dump(out/'progress.json',dict(phase='evaluation',records=len(seen),samples=samples,last_id=row['id'],last_arm=arm,elapsed_s=time.monotonic()-started))
            finally:
                for entry in entries.values():
                    entry.release();assert entry.released and not entry.chunks and entry.sink is None
                entries.clear()
            audit_cells.add(cell)
            if samples%5==0:print(json.dumps(dict(samples=samples,records=len(seen))),flush=True)
    assert set(seen)==expected_keys,(len(seen),len(expected_keys))
    if args.mode=='quant-cell':assert samples==100 and parity==100
    if args.mode=='quant-full':assert samples==1784
    if args.mode=='large-final':assert samples==1809
    if args.mode=='large-mid':assert samples==125
    dump(out/'complete.json',dict(generation_complete=True,samples=samples,records=len(seen),
        mode=args.mode,model=cfg,arms=arms,h16_parity_passed=parity==100 if args.mode=='quant-cell' else None,
        failures=dict(collections.Counter(r['status'] for r in seen.values())),judge_pending=any(r['benchmark']=='locomo' for r in seen.values()),
        note='Worker process return code and Slurm exit status must also succeed.'))

if __name__=='__main__':
    try: main()
    except BaseException:
        # Preserve explicit failures separately from scored negative results.
        print(traceback.format_exc(),flush=True);raise
