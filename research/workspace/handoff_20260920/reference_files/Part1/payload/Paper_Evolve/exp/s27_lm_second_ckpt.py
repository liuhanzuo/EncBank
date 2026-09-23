"""S27 -- the S14 LM fidelity protocol on a SECOND checkpoint.

WHY THIS EXISTS AND IS NOT JUST `s14_deployable_fix.py --model ...`
-------------------------------------------------------------------
S14 hardcodes two arms, FL_g3w and FL_g3p, as dictionaries keyed by the 8B's depth grid:

    G3_WIKI = {6: [0,3,4], 9: [1,4,8], 12: [1,9,11], ...}

Those are the |S|=3 layer sets that a greedy search selected *on Qwen3-8B*. Two problems on a
second checkpoint: the lookup raises KeyError at any depth outside that grid, and even where
it does not raise, a set greedily selected on the 8B is not a meaningful arm on the 1.7B --
we already know from the RULER sweeps that the single-layer profile does not transfer between
these two checkpoints. So this script runs S14's protocol with the arms that ARE meaningful
across checkpoints and drops the two that are not, rather than editing a shared script that
other sessions run.

Everything else is imported from S14 unchanged -- `prepare`, `eval_arms`, `samples_from`,
`pg19_text`, `forward_from`, `embed_to`, `rotate`, `_qkv_pre`, `kl`, `even_layers` -- so the
estimand, the sampling, the two self-checks and the arm construction are literally the same
code. Arms produced: A_off, A_on, A_chunk (built inside eval_arms), FL_empty, FL_1, FL_all,
C_bos and the KVb_m references.

WHAT IT IS FOR. The paper has two checkpoints at the task level but only one for its
mechanism claims: C1 (the zero-shot depth loss is entirely query-side) and C2 (lower-band
visibility removes most of it) rest on Qwen3-8B alone. This measures both on the 1.7B.

DEPTHS ARE MATCHED BY FRACTION, NOT INDEX. The 8B used j in {6,9,12,15,18,24} of L=36, i.e.
0.17-0.67 L. On L=28 the same fractions are {5,7,9,12,14,19}; j=9 = 0.32 L is the operating
point every 1.7B task result already uses. The full-depth KV references are matched the same
way: 14 of 36 -> 11 of 28, and 2 and 4 of 36 -> 2 and 3 of 28.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "Encbank"))

from s14_deployable_fix import (embed_to, eval_arms, forward_from, kl, pg19_text,  # noqa: E402
                                prepare, rotate, samples_from, _qkv_pre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-1.7B")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--pg19", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--pg19-skip", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-sel", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--js", default="5,7,9,12,14,19")
    ap.add_argument("--kv-sizes", default="2,3,11")
    ap.add_argument("--out", default="")
    ap.add_argument("--cap-gb", type=float, default=30.0)
    ap.add_argument("--need-gb", type=float, default=10.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    args = ap.parse_args()

    from gpu_gate import acquire_gpu
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from s1_binding_curve import fetch_text
    acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb,
                idle_slack_gb=args.idle_slack_gb, tag="s27_lm_second_ckpt")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    L = model.config.num_hidden_layers
    js = [int(x) for x in args.js.split(",")]
    assert max(js) < L, f"js {js} exceed L={L}"
    sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
    c, nq, k, n_doc, n_eval = args.chunk, args.query, args.k, args.n_doc, args.n_eval
    kv_sizes = [int(x) for x in args.kv_sizes.split(",")]

    if args.corpus == "wikitext":
        eval_ids = tok(fetch_text("main", Path("exp/data")),
                       return_tensors="pt").input_ids[0].to(dev)
    else:
        need = (n_doc * c + nq) * (args.n_sel + n_eval) * 5
        ids = tok(pg19_text(args.pg19, need, args.pg19_skip),
                  return_tensors="pt").input_ids[0].to(dev)
        eval_ids = ids[(n_doc * c + nq) * args.n_sel:]          # the S13 eval split
    tag = Path(args.model.rstrip("/\\")).name.lower()
    out = Path(args.out or f"exp/results/s27_deployable_{tag}_{args.corpus}.json")

    eval_samples = samples_from(eval_ids, c, n_doc, nq, k, n_eval)
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    kb_layer = 4 * model.config.num_key_value_heads * hd / 1024
    kb_resid = 2 * model.config.hidden_size / 1024
    print(f"{args.model}: L={L}, js={js} (fractions "
          + "/".join(f"{j/L:.2f}" for j in js) + f"), {len(eval_samples)} eval samples, "
          f"resid {kb_resid:.0f} KB, KV {kb_layer:.0f} KB/layer/token", flush=True)

    # the two self-checks S14 runs, unchanged
    probe = eval_ids[: c + nq].unsqueeze(0)
    pos = torch.arange(probe.shape[1], device=dev).unsqueeze(0)
    ours = forward_from(model, embed_to(model, probe, pos, 0), pos, 0, nq)
    theirs = model(input_ids=probe, position_ids=pos, use_cache=False).logits[:, -nq:, :].float()
    d = kl(theirs, ours)
    print(f"manual-forward check: KL(lib || ours) = {d:.2e} nats", flush=True)
    assert d < 3e-3
    x = model.model.embed_tokens(probe)
    lyr = model.model.layers[0]
    pe = model.model.rotary_emb(x, pos)
    _, k_direct, _, k_pre = _qkv_pre(lyr, lyr.input_layernorm(x), pe)
    print("rotation check: max|K_rot - K_direct| = "
          f"{(rotate(model, k_pre, pos) - k_direct).abs().max().item():.2e}", flush=True)

    rows = []
    for si, (chunks, qi) in enumerate(eval_samples):
        P = prepare(model, sink, chunks, qi, js, L)
        row = {"sample": si, "k": P["kk"], "kl_no_mem": P["base"]}
        for j in js:
            # the two greedy sets of S14 are 8B artefacts and are deliberately absent here
            sets = {"FL_empty": [], "FL_1": [1] if j > 1 else [0], "FL_all": list(range(j))}
            res = eval_arms(model, sink, P, j, sets, kv_sizes if j == js[0] else None, L)
            for arm, r in res.items():
                key = arm if arm == "C_bos" or arm.startswith("KVb") else f"{arm}_j{j}"
                for mname, v in r.items():
                    row[f"{key}_{mname}"] = v
        rows.append(row)
        jm = js[len(js) // 2]
        print(f"  eval {si}: base {P['base']:.3f}  j={jm}: A_on {row[f'A_on_j{jm}_frac']:.3f}  "
              f"A_chunk {row[f'A_chunk_j{jm}_frac']:.3f}  "
              f"FL_all {row[f'FL_all_j{jm}_frac']:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"model": args.model, "L": L, "args": vars(args), "js": js,
             "kb_resid": kb_resid, "kb_layer": kb_layer,
             "note": "S14 protocol on a second checkpoint; FL_g3w/FL_g3p omitted because "
                     "those layer sets were greedily selected on Qwen3-8B and do not "
                     "transfer between these checkpoints.",
             "rows": rows}, indent=1))
    print(f"wrote {out}")
    print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
