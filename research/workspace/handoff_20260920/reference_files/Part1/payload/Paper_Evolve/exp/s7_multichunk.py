"""
S7 -- Redo every arm with MULTIPLE chunks, and decompose the loss.

THE DEFECT THIS FIXES
---------------------
S2/S3/S4/S6 all used ONE chunk, placed in the read pack at exactly the positions it was
cached at.  In that configuration a cached chunk state is exact by causality: the chunk
cannot see the query anyway, so encoding `[sink; chunk]` alone and encoding it as a
prefix of `[sink; chunk; query]` are the same computation.  Measured at layer 12, the
cached chunk state differed from ground truth by relative L2 0.0138 (bf16 kernel noise)
while the cached QUERY state differed by 0.7266.  Arm C (a full-depth prefix KV cache)
came out at frac 0.000 -- an identity, not a result.

So every number so far reflects the QUERY-side effect only: how many layers the query
spends with the chunk visible.  Chunk staleness was identically zero and could not have
been measured.  That is not what deployed Encbank or CacheBlend look like: they retrieve
several chunks from scattered parts of a document, each cached without the others, then
concatenate them at fresh adjacent positions.  Cross-chunk attention is what breaks, and
it is exactly what CacheBlend's selective recompute exists to repair.

DESIGN
------
A document of `--n-doc` chunks; `--k` of them are retrieved from scattered positions.

  ref       pack [sink; c_a; c_b; ...; query] forwarded from layer 0.
            This is the RETRIEVAL upper bound, i.e. C2's matched j=0 reference and the
            same object distill.py uses as its teacher.  It is not the full document;
            `full_doc` is reported separately so the cost of retrieval itself is visible.
  no_mem    [sink; query] -- the floor.
  A         Encbank: every chunk written alone at chunk-local positions to h_j, query
            written alone, packed at fresh contiguous positions, layers [j,L) recomputed.
  C         CacheBlend without the repair: every chunk's FULL per-layer K/V cached from
            an isolated encoding at the positions it will occupy, concatenated, query
            attends to all of it.  Now non-trivial, because chunk b's cache was built
            without chunk a.

DECOMPOSITION -- the point of this script
-----------------------------------------
Arm A is run three ways so the two error sources separate:

  A_both    chunks isolated + query isolated            (deployed Encbank)
  A_chunk   chunks isolated, query's h_j TAKEN FROM ref (query side made perfect)
            -> what remains is CHUNK STALENESS alone
  A_query   chunks' h_j taken from ref, query isolated
            -> what remains is the QUERY-SIDE effect alone

If A_chunk is near zero the whole story is query-side and "pack-dependence of the stored
chunk" is not a real cost; if it is large, cross-chunk attention loss is real and the
earlier single-chunk numbers understated the problem.

METRIC -- unchanged, so everything stays comparable to S2/S3/S4/S6
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
full-vocabulary KL in nats at every query position, averaged over positions then samples.
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
    ap.add_argument("--k", type=int, default=4, help="retrieved chunks per pack")
    ap.add_argument("--n-doc", type=int, default=12, help="chunks in the document")
    ap.add_argument("--n", type=int, default=12, help="samples")
    ap.add_argument("--layers", default="", help="comma list of j; default every 4")
    ap.add_argument("--write-sink", action=argparse.BooleanOptionalAction, default=True,
                    help="prepend a BOS at write time (S4: helps for j/L<0.57)")
    ap.add_argument("--out", default="exp/results/s7_multichunk.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    L = model.config.num_hidden_layers
    js = ([int(x) for x in args.layers.split(",")] if args.layers
          else list(range(2, L, 4)))

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
    c, nq, k = args.chunk, args.query, args.k
    doc_len = args.n_doc * c

    def write(ids_1d, local_len):
        """Encode a span alone, chunk-local positions, optional write sink; return ids/pos."""
        if args.write_sink:
            x = torch.cat([sink.unsqueeze(0), ids_1d], 1)
            return x, torch.arange(x.shape[1], device=dev).unsqueeze(0), 1
        return ids_1d, torch.arange(local_len, device=dev).unsqueeze(0), 0

    rows = []
    for si in range(args.n):
        s = si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        # scattered retrieval: evenly spread over the document, not a contiguous block
        picks = [round(i * (args.n_doc - 1) / max(k - 1, 1)) for i in range(k)]
        sel = [chunks[p] for p in sorted(set(picks))]
        kk = len(sel)

        pack = torch.cat([sink.unsqueeze(0)] + sel + [qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)

        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))

        full = torch.cat([sink.unsqueeze(0), doc.unsqueeze(0), qi], 1)
        fp = torch.arange(full.shape[1], device=dev).unsqueeze(0)
        full_doc = kl(ref, forward_from(model, embed_to(model, full, fp, 0), fp, 0, final=nq))

        row = {"sample": si, "kl_no_mem": base, "kl_full_doc": full_doc, "k": kk}

        # ---- arm C: full-depth chunk KV, each chunk cached WITHOUT the others ----
        caches, off = [], 1
        for ch in sel:
            cid, cpos_local, drop = write(ch, c)
            # position repair: give the cache the positions the chunk occupies in the pack
            cpos = torch.arange(off - drop, off - drop + cid.shape[1], device=dev).unsqueeze(0)
            cc = chunk_kv_all_layers(model, cid, cpos)
            caches.append({i: (v[0][:, :, drop:, :], v[1][:, :, drop:, :])
                           for i, v in cc.items()})
            off += c
        # Keep ONE sink at the head of the merged cache.  Dropping every chunk's write
        # sink leaves the query attending to kk*512 tokens with no sink anywhere, which
        # is the degenerate regime S1/S4 measured -- an artefact of the merge, not a
        # property of CacheBlend, which packs a sink like everyone else.
        sink_kv = chunk_kv_all_layers(
            model, sink.unsqueeze(0), torch.zeros(1, 1, dtype=torch.long, device=dev))
        merged = {i: (torch.cat([sink_kv[i][0]] + [cv[i][0] for cv in caches], 2),
                      torch.cat([sink_kv[i][1]] + [cv[i][1] for cv in caches], 2))
                  for i in caches[0]}
        q_pos = torch.arange(off, off + nq, device=dev).unsqueeze(0)
        row["C"] = kl(ref, forward_with_cache(model, qi, q_pos, merged, nq)) / base

        # ---- arm A, three ways ----
        for j in js:
            h_ref = embed_to(model, pack, pp, j)
            h_s = h_ref[:, :1, :]
            hc_ref = h_ref[:, 1:1 + kk * c, :]
            hq_ref = h_ref[:, 1 + kk * c:, :]

            parts = []
            for ch in sel:
                cid, cpos, drop = write(ch, c)
                parts.append(embed_to(model, cid, cpos, j)[:, drop:, :])
            hc_iso = torch.cat(parts, 1)
            qid, qpos, qdrop = write(qi, nq)
            hq_iso = embed_to(model, qid, qpos, j)[:, qdrop:, :]

            for name, hc, hq in (("A_both", hc_iso, hq_iso),
                                 ("A_chunk", hc_iso, hq_ref),
                                 ("A_query", hc_ref, hq_iso)):
                hp = torch.cat([h_s, hc, hq], 1)
                row[f"{name}_j{j}"] = kl(ref, forward_from(model, hp, pp, j, final=nq)) / base
            # how stale is the cached chunk state itself, in the same units as S1
            row[f"stale_j{j}"] = float(
                ((hc_iso - hc_ref).norm(dim=-1) / hc_ref.norm(dim=-1).clamp_min(1e-6)).mean())
        rows.append(row)
        print(f"  sample {si}: base {base:.3f} nats, k={kk}, arm C {row['C']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args),
                               "layers": js, "rows": rows}, indent=1))

    n = len(rows)
    m = lambda key: sum(r[key] for r in rows) / n
    print(f"\nKL(ref || no_mem) = {m('kl_no_mem'):.3f} nats over {n} samples, k={rows[0]['k']}")
    print(f"full document vs the retrieved pack: frac {m('kl_full_doc'):.3f} "
          f"(what retrieval itself costs; ref IS the pack, so 0 would mean retrieval is free)")
    print(f"\narm C  CacheBlend, full-depth chunk KV, no repair, {L*4} KB/token: "
          f"{m('C'):.3f}")
    print("\n  j | A_both | A_chunk | A_query | chunk staleness (relL2)")
    print("----+--------+---------+---------+------------------------")
    for j in js:
        print(f"{j:3d} | {m(f'A_both_j{j}'):.3f}  |  {m(f'A_chunk_j{j}'):.3f}  |  "
              f"{m(f'A_query_j{j}'):.3f}  |        {m(f'stale_j{j}'):.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
