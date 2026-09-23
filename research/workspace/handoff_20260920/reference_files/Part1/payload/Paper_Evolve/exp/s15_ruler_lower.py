"""
S15 -- The query-side repair on the paper's own RULER evaluation (Qwen3-8B, zero-shot).

WHAT IS TESTED
--------------
S12-S14 showed on an LM-continuation metric that Encbank's zero-shot depth loss is query-side
and is removed by letting the query's lower band [0:j) attend to the retrieved chunks'
cached lower-layer K/V.  The paper's own evidence for j=12 is task recall: zero-shot j=12
collapses on RULER (niah_single 14, multikey 8/3/1) and needs a 4000-step LoRA to recover.
This script asks whether the repair recovers task recall WITHOUT training, on the paper's
RULER driver (eval/ruler.py sample synthesis, BM25 top-12, string_match_all).

ARMS (same backbone, same samples, paired)
------------------------------------------
  pub       Encbank as published: j=12, chunk-local write, no sink, query bottom band blind
  pub_sink  same with a BOS prepended at write (model.py write_sink=True)
  fix_all   EncbankLower: chunks written once locally (with BOS), lower-layer K stored
            pre-RoPE and rotated to pack positions at read; the query's bottom band and
            every decode step attend to [sink; chunk K/V at layers 0..j-1]; then Encbank's
            top-band read/decode unchanged.  Storage 8 + 4j KB/token.  Nothing trained.
  fix_S     same with the cache visible only at layers S (chunk entries masked elsewhere;
            the sink entry is always visible)
  j0        Encbank(resume_j=0): the RAG upper bound (full recompute of the retrieved pack)

IMPLEMENTATION NOTES
--------------------
The bottom band uses the stock HF decoder layers with a DynamicCache pre-populated per
layer with [sink; chunk_1; ...; chunk_k] K/V (post-RoPE at pack positions); the query is
processed at its pack positions (1 + sum chunk lens + t), and a per-layer boolean mask
lets query rows see the sink, the chunk entries (only at layers in S) and themselves
causally.  With S = {} and no sink entry this reduces to the published bottom band at
shifted positions (RoPE-invariant), which `--check` verifies against the base class.

Sample synthesis follows eval/ruler.py exactly (its _build_sample / _render / scorer are
imported); PYTHONHASHSEED must be fixed for its per-(task,length) seed to be reproducible.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Encbank"))
sys.path.insert(0, str(ROOT / "exp"))

from encbank import Encbank                                   # noqa: E402
from encbank import selectors as _sel                       # noqa: E402
from eval import ruler as R                               # noqa: E402
from eval._common import load_backbone                    # noqa: E402
from transformers.cache_utils import DynamicCache         # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402
import transformers.integrations.sdpa_attention as _sdpa                    # noqa: E402

# This Windows torch build has no flash-attention kernel.  With attention_mask=None HF passes
# enable_gqa=True to SDPA, which the memory-efficient/cuDNN kernels do not support, so every
# 6.7k-token top-band read silently falls back to the O(T^2) MATH kernel (12.7 GiB of scores
# per layer -> OOM).  Forcing the repeat_kv path keeps the efficient kernel (0.24 GiB).
# Numerically identical attention; only the kernel choice changes.
_sdpa.use_gqa_in_sdpa = lambda *a, **k: False


class EncbankLower(Encbank):
    """Encbank whose query bottom band sees the retrieved chunks' lower-layer K/V."""

    def __init__(self, model, resume_j, tokenizer=None, lower_layers=None,
                 sink_in_cache=True, chunk_write_sink=True):
        super().__init__(model, resume_j=resume_j, tokenizer=tokenizer)
        self.lower_layers = (set(range(resume_j)) if lower_layers is None
                             else set(int(l) for l in lower_layers))
        self.sink_in_cache = bool(sink_in_cache)
        self.write_sink = bool(chunk_write_sink)
        self._bottom = None

    # ---------------- capture lower-layer K/V during a chunk-local write -------------
    @torch.no_grad()
    def _capture_lower(self, ids):
        """Run layers[0:j] on `ids` alone (positions 0..T-1).  Returns h_j and
        {l: (k_pre [1,n_kv,T,hd], v [1,n_kv,T,hd])} with k post-k_norm, pre-RoPE."""
        ids = self._as_ids(ids)
        T = ids.shape[1]
        store, handles = {}, []
        for l in range(self.resume_j):
            attn = self.layers[l].self_attn

            def k_hook(_m, _i, out, l=l):
                store.setdefault(l, {})["k"] = out.detach()      # [1,T,n_kv,hd]

            def v_hook(_m, _i, out, l=l):
                store.setdefault(l, {})["v"] = out.detach()      # [1,T,n_kv*hd]

            handles.append(attn.k_norm.register_forward_hook(k_hook))
            handles.append(attn.v_proj.register_forward_hook(v_hook))
        try:
            emb = self.embed_tokens(ids)
            pos = torch.arange(T, device=self.device).unsqueeze(0)
            mask, pe = self._make_mask_and_rope(emb, pos)
            h_j = self._run_layers(emb, slice(0, self.resume_j), mask, pos, pe)
        finally:
            for h in handles:
                h.remove()
        n_kv = int(self.config.num_key_value_heads)
        hd = int(getattr(self.config, "head_dim",
                         self.config.hidden_size // self.config.num_attention_heads))
        kv = {}
        for l, d in store.items():
            k = d["k"].view(1, T, n_kv, hd).transpose(1, 2)
            v = d["v"].view(1, T, n_kv, hd).transpose(1, 2)
            kv[l] = (k, v)
        return h_j, kv

    def _rotate(self, k_pre, positions):
        cos, sin = self.rotary_emb(k_pre, position_ids=positions)
        return apply_rotary_pos_emb(k_pre, k_pre, cos, sin)[1]

    @torch.no_grad()
    def build_bottom(self, sink_id, chunk_ids_list):
        """Write the sink and the selected chunks locally, capture their lower K/V, rotate
        the keys to their pack positions, and pre-populate a DynamicCache.
        Returns (sink_hj, selected_hj_list) and stores the cache on self._bottom."""
        # sink alone at position 0 (no prefix)
        saved = self.write_sink
        self.write_sink = False
        try:
            sink_hj, sink_kv = self._capture_lower([sink_id])
        finally:
            self.write_sink = saved
        sink_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        pieces_k = {l: [self._rotate(sink_kv[l][0], sink_pos)] for l in sink_kv}
        pieces_v = {l: [sink_kv[l][1]] for l in sink_kv}
        sel_hj = []
        off = 1
        pre = int(self._sink_prefix_id())
        for ch in chunk_ids_list:
            ids = self._as_ids(ch)
            drop = 0
            if self.write_sink:
                ids = torch.cat([torch.tensor([[pre]], device=ids.device, dtype=ids.dtype),
                                 ids], 1)
                drop = 1
            h_j, kv = self._capture_lower(ids)
            h_j = h_j[:, drop:, :]
            n = h_j.shape[1]
            pos = torch.arange(off, off + n, device=self.device).unsqueeze(0)
            for l in kv:
                pieces_k[l].append(self._rotate(kv[l][0][:, :, drop:, :], pos))
                pieces_v[l].append(kv[l][1][:, :, drop:, :])
            sel_hj.append(h_j)
            off += n
        cache = DynamicCache(config=self.config)
        if self.sink_in_cache:
            for l in range(self.resume_j):
                cache.update(torch.cat(pieces_k[l], 2), torch.cat(pieces_v[l], 2), l)
            M = off
        else:
            for l in range(self.resume_j):
                cache.update(torch.cat(pieces_k[l][1:], 2), torch.cat(pieces_v[l][1:], 2), l)
            M = off - 1
        self._bottom = {"cache": cache, "M": M, "q_off": off}
        return sink_hj, sel_hj

    # ---------------- bottom band with per-layer masks ----------------
    def _bottom_mask(self, l, T, kv_len):
        """[1,1,T,kv_len] bool: sink col always, chunk cols iff l in S, query cols causal."""
        M = self._bottom["M"]
        m = torch.zeros(T, kv_len, dtype=torch.bool, device=self.device)
        if self.sink_in_cache:
            m[:, 0] = True
            if l in self.lower_layers:
                m[:, 1:M] = True
        elif l in self.lower_layers:
            m[:, :M] = True
        nq = kv_len - M                      # query tokens so far incl. the current T
        rows = torch.arange(T, device=self.device).view(T, 1) + (nq - T)
        cols = torch.arange(nq, device=self.device).view(1, nq)
        m[:, M:] = cols <= rows
        return m.view(1, 1, T, kv_len)

    @torch.no_grad()
    def _bottom_forward(self, emb, positions):
        cache = self._bottom["cache"]
        pe = self.rotary_emb(emb, position_ids=positions)
        T = emb.shape[1]
        hidden = emb
        for l in range(self.resume_j):
            kv_len = cache.get_seq_length(l) + T
            mask = self._bottom_mask(l, T, kv_len)
            out = self.layers[l](hidden, attention_mask=mask, position_ids=positions,
                                 position_embeddings=pe, past_key_values=cache,
                                 use_cache=True)
            hidden = self._layer_out_hidden(out)
        return hidden

    @torch.no_grad()
    def write_prefill(self, token_ids):
        if self._bottom is None:
            return super().write_prefill(token_ids)
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        q_off = self._bottom["q_off"]
        positions = torch.arange(q_off, q_off + T, device=self.device).unsqueeze(0)
        h_j = self._bottom_forward(self.embed_tokens(ids), positions)
        return h_j, self._bottom["cache"], q_off + T

    @torch.no_grad()
    def decode_step(self, token_id, bottom_cache, top_cache, q_local_pos, pack_pos):
        if self._bottom is None:
            return super().decode_step(token_id, bottom_cache, top_cache, q_local_pos, pack_pos)
        ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        emb = self.embed_tokens(ids)
        b_pos = torch.tensor([[int(q_local_pos)]], device=self.device)
        new_hj = self._bottom_forward(emb, b_pos)
        t_pos = torch.tensor([[int(pack_pos)]], device=self.device)
        t_pe = self.rotary_emb(new_hj, position_ids=t_pos)
        t_mask = self._decode_attn_mask(int(pack_pos) + 1)
        hidden = self._run_layers(new_hj, slice(self.resume_j, self.num_layers),
                                  t_mask, t_pos, t_pe, past_key_values=top_cache,
                                  use_cache=True)
        return self.lm_head(self.norm(hidden))

    @torch.no_grad()
    def generate_from_ids(self, input_ids, *, chunk_size=512, max_new_tokens=20,
                          selector="bm25", topk=12, sink_tokens="bos",
                          needle_chunk_set=None, bare_question_ids=None,
                          no_retrieval=False, stats=None, iter_rounds=0, iter_hop_topk=2,
                          iter_score="meanpool", iter_conf_ratio=0.3, iter_max_chunks=64,
                          dense_retriever=None, use_kv_cache=True, tokenizer=None):
        tok = tokenizer if tokenizer is not None else self.tokenizer
        tokens = input_ids[0]
        chunks = list(tokens.split(chunk_size))
        context_chunks = chunks[:-1]
        query_chunk = chunks[-1]
        bos_id, eos_id = self._bos_eos(tok, fallback_first=int(tokens[0].item()))
        if no_retrieval:
            sel_idx = list(range(len(context_chunks)))
        else:
            sel_idx = _sel.select_context_chunk_indices(
                selector, context_chunks, bare_question_ids or [], topk,
                needle_chunk_set, context_hj=None, query_hj=None,
                iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk,
                iter_score=iter_score, iter_conf_ratio=iter_conf_ratio,
                iter_max_chunks=iter_max_chunks, dense_retriever=dense_retriever,
                dense_tokenizer=tok)
        try:
            sink_hj, selected_hj = self.build_bottom(bos_id, [context_chunks[i] for i in sel_idx])
            if sink_tokens != "bos":
                sink_hj = None
            generated = self._decode_from_pack(
                sink_hj, selected_hj, query_chunk.tolist(), eos_id, max_new_tokens,
                use_kv_cache, stats=stats, n_context_chunks=len(context_chunks))
        finally:
            self._bottom = None
        if tok is not None:
            return tok.decode(generated, skip_special_tokens=True).strip()
        return generated


# --------------------------------------------------------------------------------------



def apply_lora_pt(model, path, fp32_accum=False):
    """Merge a flat LoRA state dict written by exp/s21_distill_lower.py into the weights.

    That trainer saves {"named": {"layers.{i}.{proj}.A"/".B": tensor}, "rank", "alpha"};
    merging W <- W + (alpha/r) B A costs nothing at run time.

    IT IS NOT EXACT.  The product is formed in fp32, but the accumulation into a bf16
    weight is a bf16 add, so any entry whose delta is below half a ULP of its weight is
    dropped.  Measured on the s21 adapters against the real Qwen3-1.7B weights: 39-42% of
    entries receive no delta at all, and roughly a third of ||dW||_F is replaced by
    rounding, even though the delta's overall magnitude survives (||W'-W||/||dW|| ~ 1.03).
    ||dW||/||W|| is 0.004-0.012 there, the same order as one bf16 ULP, so the effect on
    outputs is expected to be small -- but the evaluated model is not bit-for-bit the
    trained model, and an adapter with a smaller update loses proportionally more of it.
    Pass fp32_accum=True to accumulate in fp32 and round once at the end, which removes
    the dropped-entry effect at the cost of a transient fp32 copy of each weight."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    named = blob["named"]
    scale = float(blob["alpha"]) / float(blob["rank"])
    n = 0
    for i, layer in enumerate(model.model.layers):
        pairs = [(layer.self_attn, nm) for nm in ("q_proj", "k_proj", "v_proj", "o_proj")]
        pairs += [(layer.mlp, nm) for nm in ("gate_proj", "up_proj", "down_proj")]
        for parent, nm in pairs:
            A = named.get(f"layers.{i}.{nm}.A")
            B = named.get(f"layers.{i}.{nm}.B")
            if A is None or B is None:
                continue
            w = getattr(parent, nm).weight
            delta = (B.float() @ A.float()) * scale
            if tuple(delta.shape) != tuple(w.shape):
                raise ValueError(f"layers.{i}.{nm}: LoRA delta {tuple(delta.shape)} "
                                 f"does not match weight {tuple(w.shape)}")
            if fp32_accum:
                w.data.copy_((w.data.float() + delta.to(w.device)).to(w.dtype))
            else:
                w.data += delta.to(dtype=w.dtype, device=w.device)
            n += 1
    if n == 0:
        raise ValueError(f"{path} merged into 0 linears -- wrong checkpoint or model")
    return n, {k: blob.get(k) for k in ("j", "path", "rank", "alpha", "targets")}


def build_arms(model, tok, j, names, fix_layers):
    arms = {}
    for n in names:
        if n == "pub":
            cm = Encbank(model, resume_j=j, tokenizer=tok); cm.write_sink = False
        elif n == "pub_sink":
            cm = Encbank(model, resume_j=j, tokenizer=tok); cm.write_sink = True
        elif n == "fix_all":
            cm = EncbankLower(model, j, tok, lower_layers=None)
        elif n == "fix_S":
            cm = EncbankLower(model, j, tok, lower_layers=fix_layers)
        elif n == "fix_none":
            # control: query at pack positions with the sink entry but NO chunk visibility
            cm = EncbankLower(model, j, tok, lower_layers=[])
        elif n == "cbos":
            # full-depth isolated chunk K/V visible at ALL layers, no upper recompute
            # (resume_j = L): the BOS-corrected CacheBlend-style reference of s14, on RULER
            cm = EncbankLower(model, int(model.config.num_hidden_layers), tok, lower_layers=None)
        elif n == "fix_nosink":
            cm = EncbankLower(model, j, tok, lower_layers=None, chunk_write_sink=False)
        elif n == "j0":
            cm = Encbank(model, resume_j=0, tokenizer=tok)
        else:
            raise ValueError(n)
        arms[n] = cm
    return arms


@torch.no_grad()
def identity_check(model, tok, j, input_ids, bare_q, k, mnt):
    """EncbankLower with S={} and no sink entry must reproduce the base Encbank output."""
    base = Encbank(model, resume_j=j, tokenizer=tok); base.write_sink = False
    low = EncbankLower(model, j, tok, lower_layers=[], sink_in_cache=False, chunk_write_sink=False)
    kw = dict(chunk_size=512, max_new_tokens=mnt, selector="bm25", topk=k,
              bare_question_ids=bare_q)
    sb, sl = {"capture_step_logits": True}, {"capture_step_logits": True}
    ob = base.generate_from_ids(input_ids, stats=sb, **kw)
    ol = low.generate_from_ids(input_ids, stats=sl, **kw)
    _a, _b = sb["step_logits"][0], sl["step_logits"][0]
    _f = torch.isfinite(_a) & torch.isfinite(_b)   # step 0 masks EOS to -inf in both
    d0 = float((_a[_f] - _b[_f]).abs().max())
    return ob, ol, d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--arms", default="pub,pub_sink,fix_all,j0")
    ap.add_argument("--fix-layers", default="1,9,11")
    ap.add_argument("--tasks", default="niah_multikey_1,niah_single_2")
    ap.add_argument("--lengths", default="16k")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--adapter", default="",
                    help="path to a Encbank-distill LoRA adapter dir; empty = stock backbone")
    ap.add_argument("--adapter-pt", default="",
                    help="path to a flat LoRA .pt from exp/s21_distill_lower.py, merged "
                         "into the weights; empty = stock backbone")
    ap.add_argument("--lora-fp32-accum", action="store_true",
                    help="accumulate the merged LoRA delta in fp32 (see apply_lora_pt); "
                         "off by default so earlier result files stay reproducible")
    ap.add_argument("--selector", default="auto",
                    help="auto = eval/ruler.py routing: variable_tracking -> iter_bm25, niah -> bm25")
    ap.add_argument("--iter-hop-topk", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--essay", default=str(ROOT / "exp/data/pg19_essay.txt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", type=int, default=2, help="identity-check samples (0 = skip)")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    args = ap.parse_args()

    from gpu_gate import acquire_gpu
    acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb, idle_slack_gb=args.idle_slack_gb,
                tag="s15_ruler_lower")
    torch.manual_seed(args.seed)
    R._ESSAY_PATH = args.essay
    # --adapter is opt-in and defaults to "" (no adapter), so every earlier result file
    # was produced with the stock backbone.  load_backbone applies the LoRA through peft
    # and hands back base_model.model, which is what Encbank/EncbankLower read the layers off,
    # so both the published arms and the repaired arms see the same adapted weights.
    model, tok = load_backbone(args.model, "bfloat16", "sdpa", "cuda:0", args.adapter)
    lora_meta = None
    if args.adapter_pt:
        n_merged, lora_meta = apply_lora_pt(model, args.adapter_pt,
                                            fp32_accum=args.lora_fp32_accum)
        print(f"merged LoRA from {args.adapter_pt} into {n_merged} linears: {lora_meta}",
              flush=True)
    device = torch.device("cuda:0")
    names = [a for a in args.arms.split(",") if a]
    fix_layers = [int(x) for x in args.fix_layers.split(",") if x]
    arms = build_arms(model, tok, args.j, names, fix_layers)
    tasks = [R._resolve_task(t) for t in args.tasks.split(",")]
    lengths = args.lengths.split(",")
    out = Path(args.out or f"exp/results/s15_ruler_j{args.j}_{'-'.join(lengths)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: j={args.j} arms={names} fix_layers={fix_layers} tasks={tasks} selector={args.selector} "
          f"lengths={lengths} n={args.n} topk={args.topk} essay={os.path.exists(args.essay)} "
          f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}", flush=True)

    rows = []
    summary = {}
    for task in tasks:
        for length in lengths:
            target = R._LENGTH_TOKENS[length]
            base_seed = args.seed + (hash((task, length)) % 100000)
            vt_icl = (R._make_vt_icl(random.Random(base_seed + 777), 4)
                      if task == "variable_tracking" else None)
            mnt = args.max_new_tokens if task != "variable_tracking" else max(args.max_new_tokens, 60)
            sel = args.selector if args.selector != "auto" else ("iter_bm25" if task == "variable_tracking" else "bm25")
            sums = {a: 0.0 for a in names}
            t0 = time.time()
            for i in range(args.n):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold = R._build_sample(task, target, tok, rng, vt_icl)
                ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                bare_q = tok.encode(R._bare_question(prompt), add_special_tokens=False)
                if i < args.check:
                    ob, ol, d0 = identity_check(model, tok, args.j, input_ids, bare_q,
                                                args.topk, mnt)
                    print(f"  [check {i}] base={ob!r} lower(S={{}},nosink)={ol!r} "
                          f"same={ob == ol} max|dlogit0|={d0:.3e}", flush=True)
                row = {"task": task, "length": length, "i": i, "answers": answers,
                       "n_tokens": int(input_ids.shape[1])}
                for a in names:
                    o = arms[a].generate_from_ids(
                        input_ids, chunk_size=512, max_new_tokens=mnt, selector=sel,
                        iter_hop_topk=args.iter_hop_topk,
                        topk=args.topk, sink_tokens="bos", bare_question_ids=bare_q)
                    rec = R._string_match_all_one(o, answers)
                    row[f"{a}_out"] = o
                    row[f"{a}_recall"] = rec
                    sums[a] += rec
                rows.append(row)
                if (i + 1) % 5 == 0 or i + 1 == args.n:
                    print(f"  {task}/{length} {i+1}/{args.n}  " +
                          "  ".join(f"{a}={100*sums[a]/(i+1):.1f}" for a in names) +
                          f"  ({(time.time()-t0)/(i+1):.1f} s/sample)", flush=True)
                    out.write_text(json.dumps({"args": vars(args), "arms": names,
                                               "fix_layers": fix_layers, "lora": lora_meta, "rows": rows,
                                               "summary": summary}, indent=1))
            summary[f"{task}/{length}"] = {a: round(100 * sums[a] / args.n, 2) for a in names}
            print(f"[S15] {task}/{length} n={args.n}: " +
                  "  ".join(f"{a}={v}" for a, v in summary[f'{task}/{length}'].items()), flush=True)
            out.write_text(json.dumps({"args": vars(args), "arms": names, "fix_layers": fix_layers, "lora": lora_meta,
                                       "rows": rows, "summary": summary}, indent=1))
    print("peak GPU memory allocated: %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
