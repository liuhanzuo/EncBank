"""8,000-step rank-128 PG-19 suffix distillation; fixed final checkpoint, no eval tuning."""
import argparse, json, math, os, random, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common import *
from hybrid_reader import HybridReader

def tokens(tok, out):
    dest = out / 'pg19_tokens.u32'
    meta = out / 'tokens.json'
    if dest.exists():
        assert meta.exists()
        m = json.loads(meta.read_text())
        assert dest.stat().st_size == m['tokens'] * 4 and m['documents'] == 64
    else:
        docs, count = 0, 0
        tmp = dest.with_suffix('.tmp')
        with open(TRAIN_DATA, encoding='utf-8') as f, tmp.open('wb') as dst:
            for line in f:
                if not line.strip(): continue
                ids = tok.encode(json.loads(line)['text'], add_special_tokens=False)
                np.asarray(ids, dtype='<u4').tofile(dst)
                docs += 1; count += len(ids)
        assert docs == 64 and count > WINDOW
        tmp.replace(dest)
        dump(meta, dict(documents=docs, tokens=count, source=TRAIN_DATA,
                       separators=False, ordering='sequential cyclic'))
    return np.memmap(dest, dtype='<u4', mode='r')

def loss_for(reader, window, sink):
    # Chunked LM heads implement the same query-token mean with bounded memory.
    with reader.adapter(False), torch.no_grad():
        th = reader.core.norm(reader.full_hidden([sink] + window)[:, -CHUNK:])
        targets = []
        for part in th.split(32, dim=1):
            logits = reader.model.lm_head(part).float()
            vals, ids = logits.topk(64, dim=-1)
            targets.append((vals, ids))
        del th, logits
    segments = [[sink]] + [window[i:i+CHUNK] for i in range(0, WINDOW, CHUNK)]
    sh = reader.core.norm(reader.cache_hidden(segments, grad=True)[:, -CHUNK:])
    losses = []
    for part, (vals, ids) in zip(sh.split(32, dim=1), targets):
        def head_loss(h, v=vals, ix=ids):
            logp = F.log_softmax(v.float(), dim=-1)
            logq = F.log_softmax(reader.model.lm_head(h).float().gather(-1, ix), dim=-1)
            return (.6 * (logp.exp() * (logp-logq)).sum(-1) +
                    .4 * (logq.exp() * (logq-logp)).sum(-1)).mean()
        losses.append(checkpoint(head_loss, part, use_reentrant=False) * part.shape[1] / CHUNK)
    return torch.stack(losses).sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-index', type=int, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-after', type=int)
    args = ap.parse_args()
    cfg = MODELS[args.model_index]
    out = ROOT / 'training' / cfg['name']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print('Training already complete; no duplicate run.', flush=True); return
    assert args.resume or not (out/'last.pt').exists(), 'Existing training: explicit resume required'
    assert args.resume or not (out/'training.jsonl').exists(), 'Existing attempt: inspect before resume'
    try:
        from binding import verify_model, seed_training_tokens
        verify_model(cfg)
        seed_training_tokens(cfg, out)
        seed_all(); tok = tokenizer(cfg)
        corpus = tokens(tok, out)
        model = load_model(cfg); reader = HybridReader(model, cfg['j'])
        dump(out/'correctness_base.json', reader.validate(tok))
        params = reader.attach(rank=128, alpha=128)
        assert params and all(p.dtype == torch.float32 for p in params)
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in params}
        # Zero-initialized LoRA must preserve the native forward and split continuation.
        dump(out/'correctness_zero_lora.json', reader.validate(tok))
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.95), weight_decay=0., foreach=False)
        recipe = dict(steps=STEPS, window=WINDOW, chunk=CHUNK, context_chunks=7,
                      rank=128, alpha=128, dropout=0, seed=42, initial_skip_windows=8,
                      lr=1e-4, warmup=50, schedule='cosine', betas=[.9,.95],
                      topk=64, kl_lambda=.6, grad_accum=1, weight_decay=0,
                      targets='all suffix nn.Linear, including Gated DeltaNet projections',
                      precision='bf16 backbone, fp32 adapter master, bf16 autocast',
                      training_tokens=STEPS*WINDOW, checkpoint_selection='fixed final step 8000; step4000 diagnostic only',
                      depth_selection='preceding frozen exploratory knee screen; fixed across benchmarks')
        dump(out/'protocol.json', {**metadata(cfg), 'recipe':recipe,
             'trainable_parameters':sum(p.numel() for p in params)})
        sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        completed, cursor = 0, 8 * WINDOW
        if args.resume:
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            assert saved['recipe'] == recipe
            load_state(reader, saved, cfg); opt.load_state_dict(saved['optimizer'])
            completed, cursor = saved['step'], saved['cursor']
            torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state(saved['rng_cuda'])
            random.setstate(saved['rng_python'])
        started = time.monotonic()
        for step in range(completed, min(STEPS, args.stop_after or STEPS)):
            lr = 1e-4 * ((step+1)/50 if step < 50 else
                         .5*(1+math.cos(math.pi*(step-50)/(STEPS-50))))
            for group in opt.param_groups: group['lr'] = lr
            window = np.asarray(corpus[(np.arange(WINDOW)+cursor) % len(corpus)], dtype=np.int64).tolist()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_for(reader, window, sink)
            assert torch.isfinite(loss), 'Nonfinite distillation loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=False)
            if step == 0:
                assert all(p.grad is not None for p in params), 'Disconnected suffix adapter'
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                assert float(norm) > 0
            opt.step(); cursor += WINDOW; completed = step+1
            record = dict(phase='training', step=completed, target_steps=STEPS,
                          loss=float(loss.detach()), grad_norm=float(norm), lr=lr,
                          elapsed_s=time.monotonic()-started, cursor=cursor,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'training.jsonl').open('a') as log: log.write(json.dumps(record)+'\n')
            dump(out/'progress.json', record)
            if completed == 1 or completed % 10 == 0: print(json.dumps(record), flush=True)
            if completed % 250 == 0 or completed == (args.stop_after or STEPS):
                adapter = dict(j=cfg['j'], model=cfg, rank=128, alpha=128, step=completed,
                               recipe=recipe, modules=flat_state(reader))
                if completed in (4000, 8000):
                    save(out/f'adapter-step{completed}.pt', adapter)
                save(out/'last.pt', {**adapter, 'optimizer':opt.state_dict(), 'cursor':cursor,
                     'rng_cpu':torch.get_rng_state(), 'rng_cuda':torch.cuda.get_rng_state(),
                     'rng_python':random.getstate()})
        if completed == STEPS:
            dump(out/'correctness_adapted.json', reader.validate(tok))
            save(out/'adapter-final.pt', dict(j=cfg['j'], model=cfg, rank=128, alpha=128,
                 step=completed, recipe=recipe, modules=flat_state(reader)))
            from binding import sha256
            dump(out/'complete.json', dict(adapter_sha256=sha256(out/'adapter-final.pt'), complete=True, steps=completed,
                 adapter=str(out/'adapter-final.pt'), model=cfg, training_tokens=completed*WINDOW))
            dump(out/'progress.json', dict(phase='complete', step=completed, target_steps=STEPS))
    except BaseException:
        dump(out/('failure-'+os.environ.get('SLURM_JOB_ID','local')+'.json'),
             dict(traceback=traceback.format_exc(), time=time.time()))
        raise

if __name__ == '__main__': main()
