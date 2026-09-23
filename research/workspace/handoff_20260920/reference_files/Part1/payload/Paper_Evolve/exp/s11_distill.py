"""
S11 -- Self-distillation of the Encbank read path, on our own verified forward.

RECIPE -- taken from the paper and from Encbank/train/distill.py, not invented here
    teacher   the SAME pack read from j=0, adapters off, no grad.  Not the full
              document: the retrieval upper bound, which is what the paper distils
              against ("a student that reads from j=12 is trained to match a j=0
              teacher (which sees the full recompute) by KL divergence on their
              next-token distributions", 04_methodology.tex).
    student   the pack read from depth j: chunks written alone through the FROZEN
              layers[0:j], then layers[j:] recomputed with LoRA attached.
    loss      bidirectional KL on the teacher's top-k support,
              lam*KL(p||q) + (1-lam)*KL(q||p), lam=0.6, k=64  (distill.py:81-96)
    data      PG19 windows of (n_ctx+1) chunks; n_ctx context chunks + 1 query chunk.
    LoRA      r=32, alpha=64 on q,k,v,o,gate,up,down in layers[j:] only.
              NOTE the paper says alpha=64 while distill.py defaults to 32; we follow
              the paper and record the discrepancy.
    optim     AdamW lr 1e-4, betas (0.9,0.95), warmup 50, cosine, grad clip 1.0.
    j         9 for Qwen3-1.7B -- encbank/model_registry.py pins resume_j = round(0.33*L)
              and lists Qwen3-1.7B (L=28) -> 9 explicitly.  Earlier experiments here
              swept even j and never measured the canonical point.

WHY NOT peft
------------
peft's LoRA dispatcher calls is_torchao_available(), which RAISES on the torchao 0.13
in this machine's user site-packages, and the newer torchao needs torch >= 2.8.  Rather
than disturb a shared environment, LoRA is implemented directly here -- it is a rank-r
product added to a frozen linear, roughly thirty lines.

The better reason: training must use the SAME read path the measurements use.  Our
forward is verified against the stock HF forward to KL 2.5e-11; training through a
different implementation and then measuring through this one would reintroduce exactly
the train/inference mismatch the adapter exists to remove.

METRICS -- the paper's, not ours
    gap    Encbank-readout perplexity / teacher perplexity on the query tokens.
           1.0 is exact; the paper calls this the multiplicative LM tax
           (tab_hy3_distill.tex: "gap is the multiplicative LM tax
           (Encbank-readout perplexity / full-context perplexity; 1.0 is exact)").
    top1   argmax agreement with the teacher, same tokens.
    frac   our own normalised KL, kept so this run is comparable to S2-S10.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import wait_for_gpu
from s2_functional_kl import embed_to, forward_from


# --------------------------------------------------------------------------------------
# minimal LoRA on a frozen nn.Linear
# --------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B(A x).  W frozen; A, B trainable; B starts at zero so the
    adapter is exactly identity at step 0."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # follow the frozen weight's device and dtype: attach_lora runs after .to(cuda),
        # so defaulting to CPU here would silently split the model across devices
        dt, dv = base.weight.dtype, base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=dt, device=dv))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def attach_lora(model, j: int, r: int, alpha: int, targets):
    """Wrap the named linears in layers[j:] and return the trainable parameters."""
    params = []
    for i, layer in enumerate(model.model.layers):
        if i < j:
            continue
        for parent, name in ((layer.self_attn, n) for n in
                             ("q_proj", "k_proj", "v_proj", "o_proj")):
            if name in targets:
                mod = LoRALinear(getattr(parent, name), r, alpha)
                setattr(parent, name, mod)
                params += [mod.A, mod.B]
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in targets:
                mod = LoRALinear(getattr(layer.mlp, name), r, alpha)
                setattr(layer.mlp, name, mod)
                params += [mod.A, mod.B]
    return params


def set_lora(model, on: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale = m.scale if on else 0.0


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def windows(path, tok, chunk, n_ctx, seed=42):
    """Stream a JSONL/raw corpus into (n_ctx+1)*chunk-token windows, looping forever."""
    need = (n_ctx + 1) * chunk
    buf = []
    while True:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0] in "{[":
                    try:
                        o = json.loads(line)
                        line = o.get("text") or o.get("content") or ""
                    except json.JSONDecodeError:
                        pass
                if not line:
                    continue
                buf.extend(tok.encode(line, add_special_tokens=False))
                while len(buf) >= need:
                    w, buf = buf[:need], buf[need:]
                    yield torch.tensor(w, dtype=torch.long)


# --------------------------------------------------------------------------------------


def read_pack(model, sink, chunks, query, j, dev, n_score, ckpt=False):
    """The Encbank read: chunks written alone to depth j, packed, layers[j:] recomputed."""
    pack_len = 1 + sum(c.shape[1] for c in chunks) + query.shape[1]
    pp = torch.arange(pack_len, device=dev).unsqueeze(0)
    if j == 0:
        ids = torch.cat([sink] + chunks + [query], 1)
        return forward_from(model, embed_to(model, ids, pp, 0), pp, 0,
                            final=n_score, ckpt=ckpt)
    parts, off = [], 1
    for ch in chunks:
        cid = torch.cat([sink, ch], 1)
        cpos = torch.arange(off - 1, off - 1 + cid.shape[1], device=dev).unsqueeze(0)
        parts.append(embed_to(model, cid, cpos, j)[:, 1:, :])
        off += ch.shape[1]
    qid = torch.cat([sink, query], 1)
    qpos = torch.arange(qid.shape[1], device=dev).unsqueeze(0)
    hq = embed_to(model, qid, qpos, j)[:, 1:, :]
    hs = embed_to(model, sink, torch.zeros(1, 1, dtype=torch.long, device=dev), j)
    return forward_from(model, torch.cat([hs] + parts + [hq], 1), pp, j,
                        final=n_score, ckpt=ckpt)


def distill_loss(student_logits, t_idx, t_val, lam=0.6):
    """Bidirectional KL on the teacher's top-k support (distill.py:81-96)."""
    p = torch.softmax(t_val, -1)
    log_p = torch.log_softmax(t_val, -1)
    log_q = torch.log_softmax(torch.gather(student_logits, -1, t_idx), -1)
    q = log_q.exp()
    return (lam * (p * (log_p - log_q)).sum(-1)
            + (1 - lam) * (q * (log_q - log_p)).sum(-1)).mean()


@torch.no_grad()
def evaluate(model, stream, sink, j, dev, n_ctx, n_score, n_eval):
    """Paper metrics: gap = ppl_student/ppl_teacher (1.0 exact), top1 agreement. Plus frac."""
    gaps, tops, fracs = [], [], []
    for _ in range(n_eval):
        w = next(stream).to(dev)
        cs = list(w.split(w.shape[0] // (n_ctx + 1)))
        chunks = [c.unsqueeze(0) for c in cs[:n_ctx]]
        q = cs[n_ctx].unsqueeze(0)
        tgt = q[0, -n_score:]
        set_lora(model, False)
        t = read_pack(model, sink, chunks, q, 0, dev, n_score + 1)[:, :-1, :]
        no_mem = read_pack(model, sink, [], q, 0, dev, n_score + 1)[:, :-1, :]
        set_lora(model, True)
        s = read_pack(model, sink, chunks, q, j, dev, n_score + 1)[:, :-1, :]
        ce_t = F.cross_entropy(t[0].float(), tgt)
        ce_s = F.cross_entropy(s[0].float(), tgt)
        gaps.append(float(torch.exp(ce_s - ce_t)))
        tops.append(float((s[0].argmax(-1) == t[0].argmax(-1)).float().mean()))
        lp = F.log_softmax(t[0].float(), -1)
        kl_s = float((lp.exp() * (lp - F.log_softmax(s[0].float(), -1))).sum(-1).mean())
        kl_0 = float((lp.exp() * (lp - F.log_softmax(no_mem[0].float(), -1))).sum(-1).mean())
        fracs.append(kl_s / max(kl_0, 1e-6))
    n = len(gaps)
    return sum(gaps) / n, sum(tops) / n, sum(fracs) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot dir")
    ap.add_argument("--data", default="/f/qencbank/data/pg19_train_64.jsonl")
    ap.add_argument("--j", type=int, default=9, help="registry value for Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=3,
                   help="paper/distill.py use 7; reduced to fit a shared GPU")
    ap.add_argument("--score", type=int, default=511,
                   help="query tokens scored. Score the WHOLE query chunk, as "
                        "distill.py does (query_loss_tokens=0). Scoring only the tail "
                        "measures the region where the memory barely matters: the LM "
                        "tax at j=9 reads 1.114 over 512 tokens but 0.943 over the last "
                        "16, i.e. it inverts, because late query tokens have enough "
                        "local context of their own.")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64, help="paper says 64; code says 32")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--out", default="exp/results/s11_distill.json")
    ap.add_argument("--cap-gb", type=float, default=12.0)
    ap.add_argument("--need-gb", type=float, default=14.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    torch.manual_seed(42)
    random.seed(42)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    L = model.config.num_hidden_layers
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    params = attach_lora(model, args.j, args.rank, args.alpha, targets)
    n_par = sum(p.numel() for p in params)
    print(f"L={L}  j={args.j}  LoRA r={args.rank} alpha={args.alpha} on layers[{args.j}:{L}]"
          f"  -> {n_par/1e6:.2f}M trainable")

    sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
    train = windows(args.data, tok, args.chunk, args.n_ctx)
    ev = windows(args.data, tok, args.chunk, args.n_ctx, seed=7)
    for _ in range(64):          # hold out the first windows for eval
        next(ev)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist = []

    g, t, fr = evaluate(model, ev, sink, args.j, dev, args.n_ctx, args.score, args.n_eval)
    print(f"step   0  |  gap {g:.4f}  top1 {t:.4f}  frac {fr:.4f}   (before training)")
    hist.append({"step": 0, "gap": g, "top1": t, "frac": fr})

    for step in range(1, args.steps + 1):
        lr = (args.lr * step / args.warmup if step < args.warmup else
              0.5 * args.lr * (1 + math.cos(math.pi * (step - args.warmup) /
                                            max(1, args.steps - args.warmup))))
        for gp in opt.param_groups:
            gp["lr"] = lr
        w = next(train).to(dev)
        cs = list(w.split(w.shape[0] // (args.n_ctx + 1)))
        chunks = [c.unsqueeze(0) for c in cs[:args.n_ctx]]
        q = cs[args.n_ctx].unsqueeze(0)

        with torch.no_grad():
            set_lora(model, False)
            t_log = read_pack(model, sink, chunks, q, 0, dev, args.score).float()
            tk = torch.topk(t_log, min(args.topk, t_log.shape[-1]), -1)
        set_lora(model, True)
        s_log = read_pack(model, sink, chunks, q, args.j, dev, args.score,
                          ckpt=True).float()
        loss = distill_loss(s_log[0], tk.indices[0], tk.values[0], args.lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps:
            g, t, fr = evaluate(model, ev, sink, args.j, dev, args.n_ctx,
                                args.score, args.n_eval)
            print(f"step {step:3d}  |  gap {g:.4f}  top1 {t:.4f}  frac {fr:.4f}  "
                  f"|  loss {float(loss):.4f}  lr {lr:.2e}", flush=True)
            hist.append({"step": step, "gap": g, "top1": t, "frac": fr,
                         "loss": float(loss)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "trainable": n_par, "hist": hist}, indent=1))
    torch.save({k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k},
               str(out.with_suffix(".pt")))
    print(f"\nwrote {out} and {out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
