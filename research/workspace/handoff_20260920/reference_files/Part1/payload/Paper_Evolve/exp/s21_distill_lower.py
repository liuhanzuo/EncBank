"""
S21 -- Self-distillation for BOTH read paths, so the base paper's adapter and our repair
can be compared and composed on the same backbone.

WHY A LOCAL TRAINER
-------------------
The repo's own trainer (COMem/train/distill.py) uses peft, whose LoRA dispatcher calls
is_torchao_available(); the torchao 0.13 in this machine's user site-packages raises there,
and disabling user site-packages hides `packaging`, which transformers needs. Rather than
install into a shared environment, this file trains the same recipe with the hand-rolled
LoRA of exp/s11_distill.py and saves a plain state_dict that exp/s15_ruler_lower.py can
load (--adapter-pt), so the adapter is exercised by exactly the forward the evaluation uses.

RECIPE -- taken from COMem/train/distill.py and the base paper, not invented here
    teacher   the SAME pack read from j=0 with the adapter OFF, no grad. Not the full
              document: the retrieval upper bound the base paper distils against.
    student   the pack read from depth j with LoRA attached to layers[j:].
    loss      bidirectional KL on the teacher's top-k support,
              lam*KL(p||q) + (1-lam)*KL(q||p), lam=0.6, k=64  (distill.py:81-96)
    LoRA      r=32, alpha=32 on q,k,v,o,gate,up,down in layers[j:] only (COMem/train/
              README.md defaults; scale = alpha/r = 1.0).
    optim     AdamW lr 1e-4, 50 warmup steps, cosine, grad clip 1.0, 1000 total steps
              -- COMem/train/README.md's documented run.
    data      PG19 windows of (n_ctx+1) chunks: n_ctx context chunks + 1 query chunk.

THE TWO PATHS (--path)
    base   student reads as published CoMem does: chunks written alone to h_j, query
           written alone to h_j, pack recomputed from j. This reproduces the base paper's
           adapter, whose job is to teach the upper band to read a blind lower band.
    lower  student reads with the repair: the chunks' lower-band K/V are cached and the
           query's own [0:j) attends to them before the pack is recomputed from j. An
           adapter trained here is asked to improve a reader that can already see the
           memory, which is the composition question.
The teacher is identical in both cases, so the two adapters are comparable.

Nothing about COMem/comem/model.py changes; the LoRA is attached to a model this script
loaded and is written to its own file.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "COMem"))
sys.path.insert(0, str(ROOT / "exp"))

from comem import CoMem                                        # noqa: E402
from s11_distill import LoRALinear, attach_lora, windows       # noqa: E402
from s15_ruler_lower import CoMemLower                         # noqa: E402


def flat_state(model):
    """{"layers.{i}.{proj}.A"/".B"} -- the form exp/s15_ruler_lower.py --adapter-pt merges."""
    named = {}
    for i, layer in enumerate(model.model.layers):
        pairs = [(layer.self_attn, nm) for nm in ("q_proj", "k_proj", "v_proj", "o_proj")]
        pairs += [(layer.mlp, nm) for nm in ("gate_proj", "up_proj", "down_proj")]
        for parent, nm in pairs:
            mod = getattr(parent, nm, None)
            if isinstance(mod, LoRALinear):
                named[f"layers.{i}.{nm}.A"] = mod.A.detach().float().cpu()
                named[f"layers.{i}.{nm}.B"] = mod.B.detach().float().cpu()
    return named


def set_lora(model, on: bool, scale: float):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale = scale if on else 0.0


def teacher_logits(cm0, sink_id, ctx_chunks, query_ids, n_score):
    """j=0 replay of [sink; ctx; query] with the adapter off -- the distillation target."""
    sink_hj = cm0.write_chunk([sink_id])
    sel = cm0.write_chunks(list(ctx_chunks))
    q_hj = cm0.write_chunk(query_ids)
    return cm0.read_core(sink_hj, sel, q_hj, logits_tail=n_score)


def student_logits(cm, sink_id, ctx_chunks, query_ids, n_score, lower):
    """Student read at depth j, either the published path or the repaired one."""
    if lower:
        sink_hj, sel = cm.build_bottom(sink_id, list(ctx_chunks))
        q_hj = cm.write_prefill(query_ids)[0]
        out = cm.read_core(sink_hj, sel, q_hj, logits_tail=n_score)
        cm._bottom = None          # write_prefill appended the query's K/V to this cache
        return out
    sink_hj = cm.write_chunk([sink_id])
    sel = cm.write_chunks(list(ctx_chunks))
    q_hj = cm.write_chunk(query_ids)
    return cm.read_core(sink_hj, sel, q_hj, logits_tail=n_score)


def distill_loss(student, t_idx, t_val, lam=0.6):
    """Bidirectional KL on the teacher's top-k support (COMem/train/distill.py:81-96)."""
    log_q = torch.log_softmax(student.float(), -1).gather(-1, t_idx)
    log_p = torch.log_softmax(t_val.float(), -1)
    p, q = log_p.exp(), log_q.exp()
    return (lam * (p * (log_p - log_q)).sum(-1)
            + (1 - lam) * (q * (log_q - log_p)).sum(-1)).mean()


@torch.no_grad()
def evaluate(cm, cm0, sink_id, batches, n_score, lower):
    """Mean top-1 agreement with the teacher and mean per-token KL, adapter as attached."""
    agree, kls = [], []
    for ctx, q in batches:
        t = teacher_logits(cm0, sink_id, ctx, q, n_score)
        s = student_logits(cm, sink_id, ctx, q, n_score, lower)
        agree.append(float((s.argmax(-1) == t.argmax(-1)).float().mean()))
        lp = torch.log_softmax(t.float(), -1)
        kls.append(float((lp.exp() * (lp - torch.log_softmax(s.float(), -1))).sum(-1).mean()))
    return sum(agree) / len(agree), sum(kls) / len(kls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-1.7B")
    ap.add_argument("--path", default="base", choices=["base", "lower"])
    ap.add_argument("--j", type=int, default=9)
    ap.add_argument("--data", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=7)
    ap.add_argument("--score", type=int, default=512,
                    help="query positions scored per step; 512 = the whole query chunk, "
                         "which is COMem/train/distill.py's --query_loss_tokens 0")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)   # rank 32 / alpha 32 = scale 1.0
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=10.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="checkpoint the read layer loop (CoMem.grad_checkpoint)")
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()
    lower = args.path == "lower"
    out = Path(args.out or f"exp/results/s21_lora_{args.path}_j{args.j}.pt")

    if args.cpu_smoke:
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model)
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(cfg).float().eval()
        j, chunk, n_ctx, score, steps = 2, 16, 2, 8, 4
        args.eval_every, args.n_eval = 2, 2
        out = Path(f"exp/results/s21_smoke_{args.path}.pt")
    else:
        from gpu_gate import acquire_gpu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb,
                    idle_slack_gb=args.idle_slack_gb, tag=f"s21_distill_{args.path}")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        j, chunk, n_ctx, score, steps = args.j, args.chunk, args.n_ctx, args.score, args.steps
    torch.manual_seed(args.seed)

    for p in model.parameters():
        p.requires_grad_(False)
    targets = set(t for t in args.targets.split(",") if t)
    params = attach_lora(model, j, args.rank, args.alpha, targets)
    n_train = sum(p.numel() for p in params)

    # teacher: j=0, adapter forced off; student: depth j, adapter on
    cm0 = CoMem(model, resume_j=0, tokenizer=tok)
    cm = (CoMemLower(model, j, tok, lower_layers=None) if lower
          else CoMem(model, resume_j=j, tokenizer=tok))
    if not lower:
        cm.write_sink = False          # the published write, which is what the base adapter fixes
    cm.grad_checkpoint = bool(args.grad_ckpt)
    sink_id = int(tok.bos_token_id or tok.eos_token_id)
    base_scale = args.alpha / args.rank

    dev = next(model.parameters()).device

    def split_window(w):
        """windows() yields one flat (n_ctx+1)*chunk tensor: n_ctx context chunks + query."""
        pieces = list(torch.as_tensor(w).split(chunk))
        return [c.to(dev) for c in pieces[:n_ctx]], pieces[n_ctx].to(dev)

    stream = windows(args.data, tok, chunk, n_ctx, seed=args.seed)
    held = [split_window(next(stream)) for _ in range(args.n_eval)]   # never trained on

    print(f"{args.model}: path={args.path} j={j} L={model.config.num_hidden_layers} "
          f"trainable={n_train} steps={steps} score={score} n_ctx={n_ctx}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist = []

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    t0 = time.time()
    for step in range(steps + 1):
        if step % args.eval_every == 0 or step == steps:
            set_lora(model, True, base_scale)
            a, k = evaluate(cm, cm0, sink_id, held, score, lower)
            set_lora(model, False, base_scale)
            a0, k0 = evaluate(cm, cm0, sink_id, held, score, lower)
            set_lora(model, True, base_scale)
            hist.append({"step": step, "top1": a, "kl": k, "top1_noadapter": a0,
                         "kl_noadapter": k0, "elapsed_s": time.time() - t0})
            print(f"  step {step:5d}  top1 {a:.4f} (no-adapter {a0:.4f})  "
                  f"KL {k:.4f} (no-adapter {k0:.4f})  {time.time()-t0:.0f}s", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"args": vars(args), "j": j, "path": args.path, "rank": args.rank,
                        "alpha": args.alpha, "targets": sorted(targets), "hist": hist,
                        "named": flat_state(model)}, out)
        if step == steps:
            break
        ctx, q = split_window(next(stream))
        set_lora(model, False, base_scale)
        with torch.no_grad():
            t = teacher_logits(cm0, sink_id, ctx, q, score)
            tk = torch.topk(t.float(), min(args.topk, t.shape[-1]), -1)
        set_lora(model, True, base_scale)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        s = student_logits(cm, sink_id, ctx, q, score, lower)
        loss = distill_loss(s, tk.indices, tk.values, args.lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"    step {step:5d} loss {float(loss):.4f} lr {lr_at(step):.2e}", flush=True)

    # save in the flat form s15_ruler_lower.py --adapter-pt expects
    named = {}
    idx = 0
    for i, layer in enumerate(model.model.layers):
        for parent, nm in [(layer.self_attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
                          [(layer.mlp, n) for n in ("gate_proj", "up_proj", "down_proj")]:
            mod = getattr(parent, nm, None)
            if isinstance(mod, LoRALinear):
                named[f"layers.{i}.{nm}.A"] = mod.A.detach().cpu()
                named[f"layers.{i}.{nm}.B"] = mod.B.detach().cpu()
                idx += 1
    torch.save({"args": vars(args), "j": j, "path": args.path, "rank": args.rank,
                "alpha": args.alpha, "targets": sorted(targets), "hist": hist,
                "named": named}, out)
    print(f"wrote {out}  ({idx} wrapped linears, {n_train} trainable params)")
    if not args.cpu_smoke:
        print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
