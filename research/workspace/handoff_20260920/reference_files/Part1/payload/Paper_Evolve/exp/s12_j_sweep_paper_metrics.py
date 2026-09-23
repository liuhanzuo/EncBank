"""
S12 -- Sweep the split depth j in the PAPER's metrics, and test its flat-plateau claim.

TWO THINGS THIS SETTLES
-----------------------
1. The canonical operating point was never measured.  `comem/model_registry.py` pins
   `resume_j = round(0.33 * L)` and lists Qwen3-1.7B (L=28) -> **9** explicitly, but
   S2/S9 swept even j only and skipped it.

2. A live conflict with the paper.  `03_motivation.tex:29` reports a "flat plateau over
   j in [8,32] (top-1 agreement ~0.90, KL ~0.21), placing the cacheable ceiling near
   0.4L" on the 80-layer Hy3, whereas our `frac` curve rises steeply and monotonically.
   That could be a real disagreement or an artefact of using different metrics and a
   different reference, and until it is resolved none of our j-dependent statements can
   be trusted.

WHY THIS SETTING IS THE COMPARABLE ONE
--------------------------------------
The paper's `gap` is "CoMem-readout perplexity / **full-context** perplexity"
(tab_hy3_distill.tex:19-21).  Our S2-S9 reference was the RETRIEVED PACK, which is a
strict subset of the document -- not the same object.  In this PG19 language-modelling
setting there is no retrieval: every one of the n_ctx context chunks goes into the pack,
so **pack == full context** and the two metrics coincide.  That makes this the one
setting in which our numbers can be laid next to the paper's.

METRICS -- all three, so the old and new work stay linked
    gap    exp(CE_student - CE_teacher) on the query tokens.  1.0 = exact.
           The paper's "multiplicative LM tax"; +14.6% means gap 1.146.
    top1   fraction of query positions where student and teacher argmax agree.
    frac   KL(teacher||student) / KL(teacher||no_mem), our own normalisation, kept
           only so this run can be compared with S2-S11.

The teacher (j=0) and the no-memory floor do not depend on j, so each evaluation window
is forwarded once for those and once per j for the student.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import wait_for_gpu
from s11_distill import read_pack, windows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=r"/srv/encbank/legacy_workspace\data\pg19_train_64.jsonl")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=7, help="paper/distill.py default")
    ap.add_argument("--score", type=int, default=511, help="whole query chunk")
    ap.add_argument("--n", type=int, default=24, help="evaluation windows")
    ap.add_argument("--js", default="", help="default: every j in [0, L)")
    ap.add_argument("--out", default="exp/results/s12_j_sweep.json")
    ap.add_argument("--cap-gb", type=float, default=16.0)
    ap.add_argument("--need-gb", type=float, default=20.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(dev).eval()
    L = model.config.num_hidden_layers
    js = ([int(x) for x in args.js.split(",")] if args.js else list(range(0, L)))
    sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
    gen = windows(args.data, tok, args.chunk, args.n_ctx)
    n_score = args.score
    print(f"L={L}  registry j = round(0.33*L) = {round(0.33 * L)}  "
          f"n_ctx={args.n_ctx} (pack == full context, no retrieval)  n={args.n}")

    acc = {j: {"gap": [], "top1": [], "frac": []} for j in js}
    for wi in range(args.n):
        w = next(gen).to(dev)
        cs = list(w.split(w.shape[0] // (args.n_ctx + 1)))
        chunks = [c.unsqueeze(0) for c in cs[: args.n_ctx]]
        q = cs[args.n_ctx].unsqueeze(0)
        tgt = q[0, -n_score:]

        t = read_pack(model, sink, chunks, q, 0, dev, n_score + 1)[:, :-1, :]
        z = read_pack(model, sink, [], q, 0, dev, n_score + 1)[:, :-1, :]
        ce_t = float(F.cross_entropy(t[0].float(), tgt))
        lp = F.log_softmax(t[0].float(), -1)
        kl0 = float((lp.exp() * (lp - F.log_softmax(z[0].float(), -1))).sum(-1).mean())
        t_arg = t[0].argmax(-1)

        for j in js:
            s = t if j == 0 else read_pack(model, sink, chunks, q, j, dev, n_score + 1)[:, :-1, :]
            ce_s = float(F.cross_entropy(s[0].float(), tgt))
            acc[j]["gap"].append(float(torch.exp(torch.tensor(ce_s - ce_t))))
            acc[j]["top1"].append(float((s[0].argmax(-1) == t_arg).float().mean()))
            kls = float((lp.exp() * (lp - F.log_softmax(s[0].float(), -1))).sum(-1).mean())
            acc[j]["frac"].append(kls / max(kl0, 1e-9))
        if wi % 6 == 0:
            print(f"  window {wi}: CE_teacher {ce_t:.4f}, KL(t||no_mem) {kl0:.4f}", flush=True)

    m = lambda j, k: sum(acc[j][k]) / len(acc[j][k])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "args": vars(args), "L": L,
         "rows": [{"j": j, "gap": m(j, "gap"), "top1": m(j, "top1"), "frac": m(j, "frac")}
                  for j in js]}, indent=1))

    print("\n  j | j/L  |   gap    LM tax |  top1  |  frac")
    print("----+------+-----------------+--------+-------")
    for j in js:
        star = "  <- registry" if j == round(0.33 * L) else ""
        print(f"{j:3d} | {j/L:.2f} | {m(j,'gap'):.4f}  {100*(m(j,'gap')-1):+6.1f}% | "
              f"{m(j,'top1'):.4f} | {m(j,'frac'):.3f}{star}")
    print("\npaper (Hy3, L=80): flat plateau over j in [8,32] i.e. j/L in [0.10,0.40], "
          "top1 ~ 0.90, KL ~ 0.21, ceiling near 0.4L")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
