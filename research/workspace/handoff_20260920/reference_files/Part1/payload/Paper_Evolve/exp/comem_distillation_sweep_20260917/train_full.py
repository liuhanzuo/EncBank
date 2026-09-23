"""Full-parameter self-distillation: fixed original teacher, unchanged packed student and KL.

BF16 student computation/gradients, FP32 master parameters and Adam moments.
This preserves the original BF16 forward path including RMSNorm/RoPE dtypes.
"""
import argparse, json, os, random, time
from pathlib import Path
import config
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import train_support as T


class FullReader(T.PublishedReader):
    def write(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.first).reshape(1, -1)
        return self.layers(self.model.model.embed_tokens(ids), 0, self.j, use_checkpoint=True).to(self.first)


def full_loss(student, teacher, window, bos, args):
    with torch.no_grad():
        teacher_h = teacher.hidden(window, bos, args.chunk, args.n_ctx, teacher=True)
        targets = []
        for part in teacher_h.split(args.logit_chunk, dim=1):
            logits = teacher.model.lm_head(part).float()
            top = logits.topk(min(args.topk, logits.shape[-1]), dim=-1)
            targets.append((top.indices, top.values))
            del logits
        del teacher_h
    student_h = student.hidden(window, bos, args.chunk, args.n_ctx)
    losses = []
    for part, (idx, val) in zip(student_h.split(args.logit_chunk, dim=1), targets):
        def head_loss(h, indices=idx, values=val):
            return T.support_loss(student.model.lm_head(h), indices, values, args.lam, args.loss)
        loss = checkpoint(head_loss, part, use_reentrant=False) if torch.is_grad_enabled() else head_loss(part)
        losses.append(loss * part.shape[1] / args.chunk)
    return torch.stack(losses).sum()


class MasterAdam:
    def __init__(self, model, lr):
        self.named = list(model.named_parameters())
        assert all(p.requires_grad for _, p in self.named)
        self.master = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.named]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=(.9, .95), weight_decay=0, foreach=False)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)
        for _, p in self.named:
            p.grad = None

    def step(self, lr):
        for group in self.opt.param_groups:
            group['lr'] = lr
        for (name, p), master in zip(self.named, self.master):
            if p.grad is None:
                raise AssertionError(f'Full-parameter path has no gradient for {name}')
            master.grad = p.grad.detach().float()
        norm = torch.nn.utils.clip_grad_norm_(self.master, 1., error_if_nonfinite=True, foreach=False)
        self.opt.step()
        with torch.no_grad():
            for (_, p), master in zip(self.named, self.master):
                p.copy_(master)
        self.zero_grad()
        return float(norm)

    def state_dict(self):
        return dict(names=[n for n, _ in self.named], master=[p.detach() for p in self.master], optimizer=self.opt.state_dict())

    def load_state_dict(self, saved):
        assert saved['names'] == [n for n, _ in self.named]
        with torch.no_grad():
            for p, value in zip(self.master, saved['master']):
                p.copy_(value)
            for (_, p), master in zip(self.named, self.master):
                p.copy_(master)
        self.opt.load_state_dict(saved['optimizer'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', type=int, required=True)
    args0 = ap.parse_args()
    arm = config.ARMS[args0.arm]
    assert arm['mode'] == 'full'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    out = config.ROOT/'runs'/arm['name']/'training'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['complete']:
        return
    args = argparse.Namespace(chunk=512, n_ctx=7, topk=64, lam=.6, loss='published', logit_chunk=32,
                              steps=arm['steps'], lr=arm['lr'], warmup=50, j=12, seed=42, skip_windows=8)
    torch.set_num_threads(2)
    torch.manual_seed(42); random.seed(42)
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    tok = AutoTokenizer.from_pretrained(config.MODEL, local_files_only=True)
    data_meta = T.prepare_tokens(config.DATA, tok, config.TOKEN_CACHE, config.MODEL)
    def load():
        return AutoModelForCausalLM.from_pretrained(config.MODEL, dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    teacher_model = load().requires_grad_(False)
    student_model = load().requires_grad_(True)
    assert not any(p.requires_grad for p in teacher_model.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(student_model.parameters(), teacher_model.parameters()))
    assert len(student_model.model.layers) == 36 and student_model.config.attention_dropout == 0
    optim = MasterAdam(student_model, args.lr)
    teacher, student = T.PublishedReader(teacher_model, 12), FullReader(student_model, 12)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    assert bos == 151645
    tokens = np.memmap(config.TOKEN_CACHE, dtype='<u4', mode='r')
    stream = T.TokenStream(tokens, 4096, 8*4096)
    metadata = dict(arm=arm, recipe=vars(args), data=data_meta, model=config.MODEL, training_bos=bos,
        teacher='independent frozen original full causal model; never updated',
        student='independent chunk/query Write followed by causal packed Read; all parameters trainable',
        precision='BF16 student/teacher forwards and student gradients; FP32 master parameters and Adam moments',
        loss='unchanged teacher-top64 support-normalized .6 forward KL + .4 reverse KL, query512 only',
        optimizer=dict(betas=[.9,.95], weight_decay=0, clip=1, warmup=50, schedule='cosine to planned steps'),
        trainable_parameters=sum(p.numel() for p in student_model.parameters()),
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        source_order='cyclic PG19 token concatenation, no EOS insertion, skip first8 windows',
        checkpoint_selection='final fixed training budget only; no LongEval selection')
    completed = 0
    if (out/'last.pt').exists():
        saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        for key in ('arm','recipe','data','model','training_bos','precision'):
            assert saved['metadata'][key] == metadata[key], key
        student_model.load_state_dict(saved['model'])
        optim.load_state_dict(saved['master_optimizer'])
        completed, stream.cursor = saved['step'], saved['cursor']
        T.restore_rng(saved['rng'], [torch.device('cuda:0')])
        del saved
    T.atomic_json(out/'metadata.json', metadata)
    started = time.monotonic()
    teacher_probe = next(teacher_model.parameters()).detach().reshape(-1)[:4096].clone()
    def save(step):
        T.atomic_save(out/'last.pt', dict(step=step, cursor=stream.cursor, model=student_model.state_dict(),
            master_optimizer=optim.state_dict(), rng=T.rng_state([torch.device('cuda:0')]), metadata=metadata))
    row = dict(step=completed, target_steps=args.steps, complete=False)
    with (out/'train.jsonl').open('a', encoding='utf-8') as log:
        for step in range(completed, args.steps):
            optim.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = full_loss(student, teacher, stream.take(), bos, args)
            assert torch.isfinite(loss), f'Nonfinite loss at {step+1}'
            loss.backward()
            if step == completed:
                gradient_check = {}
                for name, p in student_model.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    if name in ('model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight',
                                'model.layers.11.mlp.down_proj.weight','model.layers.35.mlp.down_proj.weight','lm_head.weight'):
                        gradient_check[name] = float(p.grad.float().abs().sum())
                        assert gradient_check[name] > 0, name
                assert all(p.grad is None for p in teacher_model.parameters())
                T.atomic_json(out/'gradient_check.json',dict(teacher_frozen=True, all_student_gradients_finite=True,
                                                            nonzero_gradients=gradient_check, at_step=step+1))
            norm = optim.step(T.lr_at(step, args))
            row = dict(step=step+1, target_steps=args.steps, loss=float(loss.detach()), grad_norm=norm,
                lr=T.lr_at(step,args), token_cursor=stream.cursor, elapsed_s=time.monotonic()-started,
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, complete=False)
            log.write(json.dumps(row)+'\n'); log.flush()
            T.atomic_json(out/'status.json', row)
            if step == completed or (step+1)%10 == 0:
                print(json.dumps(row), flush=True)
            if (step+1)%1000 == 0 or step+1 == args.steps:
                assert all(p.grad is None for p in teacher_model.parameters())
                assert torch.equal(teacher_probe, next(teacher_model.parameters()).detach().reshape(-1)[:4096])
                save(step+1)
    student_model.save_pretrained(out/'final', safe_serialization=True, max_shard_size='4GB')
    tok.save_pretrained(out/'final')
    T.atomic_json(out/'status.json', dict(row, complete=True, exported=str(out/'final')))


if __name__ == '__main__':
    main()
