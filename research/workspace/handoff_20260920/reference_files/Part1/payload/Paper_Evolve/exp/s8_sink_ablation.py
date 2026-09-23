"""
S8 -- How load-bearing is the attention sink when independently-cached chunks are merged?

THE OBSERVATION THIS TESTS
--------------------------
In S7, merging the full-depth K/V of 4 independently cached chunks and letting the query
attend to the result scored frac 1.355 -- WORSE than having no memory at all -- until one
BOS sink was kept at the head of the merged cache, at which point it scored 0.041.  One
token, a 33x swing.  That was found while checking an implementation, not by design, so
it needs to be established properly before it is claimed.

Two questions decide whether it is a real, general result or a quirk of one setting:

  1. Does the damage GROW WITH THE NUMBER OF MERGED CHUNKS?  If a sinkless merge gets
     worse as k rises, the mechanism is "attention mass has nowhere to go and the amount
     of misplaced mass scales with the cache", which is the attention-sink story and
     applies to the whole chunk-KV family (Prompt Cache, Block-Attention, TurboRAG,
     EPIC, CacheBlend).  If it is flat in k, it is something narrower.
  2. Is ONE sink enough, or does it need to scale?  If one suffices at every k, the fix
     is genuinely free and there is nothing to tune.

ARMS
----
  n_sink in {0, 1, 2, 4} tokens kept at the head of the merged cache
  k      in {1, 2, 4, 8} independently cached chunks, scattered over the document

k=1 is the control: with a single chunk at its own positions the merge is a true prefix,
so causality makes the cache exact and the sink cannot matter for coherence.  A large
sink effect at k=1 would mean the mechanism is NOT about merging, and would falsify the
reading above.

METRIC -- unchanged from S2/S3/S4/S6/S7
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
full-vocabulary KL in nats at every query position, averaged over positions then samples.
frac > 1 means the memory left the model worse off than having no memory.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl
from s6_cacheblend_arm import chunk_kv_all_layers, forward_with_cache


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--sinks", default="0,1,2,4")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="exp/results/s8_sink_ablation.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()

    bos = tok.bos_token_id or tok.eos_token_id
    sink1 = torch.tensor([bos], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
    c, nq = args.chunk, args.query
    ks = [int(x) for x in args.ks.split(",")]
    n_sinks = [int(x) for x in args.sinks.split(",")]
    doc_len = args.n_doc * c

    rows = []
    for si in range(args.n):
        s = si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        row = {"sample": si}

        for k in ks:
            picks = sorted({round(i * (args.n_doc - 1) / max(k - 1, 1)) for i in range(k)})
            sel = [chunks[p] for p in picks]
            kk = len(sel)

            # reference and floor for THIS k
            pack = torch.cat([sink1.unsqueeze(0)] + sel + [qi], 1)
            pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
            ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
            short = torch.cat([sink1.unsqueeze(0), qi], 1)
            sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
            base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))
            row[f"base_k{kk}"] = base

            # cache each chunk alone, with its own write sink, at its pack positions
            caches, off = [], 1
            for ch in sel:
                cid = torch.cat([sink1.unsqueeze(0), ch], 1)
                cpos = torch.arange(off - 1, off - 1 + cid.shape[1], device=dev).unsqueeze(0)
                cc = chunk_kv_all_layers(model, cid, cpos)
                caches.append({i: (v[0][:, :, 1:, :], v[1][:, :, 1:, :]) for i, v in cc.items()})
                off += c
            q_pos = torch.arange(off, off + nq, device=dev).unsqueeze(0)

            for ns in n_sinks:
                if ns == 0:
                    head = []
                else:
                    sids = torch.full((1, ns), bos, device=dev, dtype=torch.long)
                    spos = torch.arange(ns, device=dev).unsqueeze(0)
                    sk = chunk_kv_all_layers(model, sids, spos)
                    head = [sk]
                merged = {i: (torch.cat([h[i][0] for h in head] + [cv[i][0] for cv in caches], 2),
                              torch.cat([h[i][1] for h in head] + [cv[i][1] for cv in caches], 2))
                          for i in caches[0]}
                row[f"k{kk}_s{ns}"] = kl(
                    ref, forward_with_cache(model, qi, q_pos, merged, nq)) / base
        rows.append(row)
        if si % 4 == 0:
            print(f"  sample {si} done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args), "rows": rows}, indent=1))

    n = len(rows)
    m = lambda key: sum(r[key] for r in rows if key in r) / max(
        1, len([r for r in rows if key in r]))
    ks_seen = sorted({int(kx[1:].split("_")[0]) for kx in rows[0] if kx.startswith("k")})
    print(f"\nfrac by (chunks merged) x (sink tokens kept), {n} samples")
    print("frac > 1.000 means the memory made the model WORSE than no memory at all\n")
    print("  k | cached tokens | " + " | ".join(f"{ns} sink" for ns in n_sinks))
    print("----+---------------+" + "+".join(["--------"] * len(n_sinks)))
    for k in ks_seen:
        cells = " | ".join(f"{m(f'k{k}_s{ns}'):6.3f} " for ns in n_sinks)
        print(f"{k:3d} |     {k*c:6d}    | {cells}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
