"""
S1 -- Layer-wise context-binding curve: residual stream vs attention K/V.

QUESTION
--------
Encbank (C2) stores the residual stream h_j of a chunk, computed in isolation with
chunk-local positions, then injects it into a freshly-assembled read pack and
recomputes layers [j, L).  Its five-arm contrast found that at 16k essentially the
whole accuracy loss is carried by depth itself (A .52 -> E .16), not by chunk
independence (-.02) and not by position relocation (0 discordant at the endpoint).

The proposed alternative (Family B / Memorizing-Transformer style) stores K,V at a
small set of layers instead, and never recomputes the context.  The stated hope is
that attention over stored K,V is more forgiving than injecting a residual, because
a K/V pair only has to be *read from*, whereas a residual has to *keep computing*
inside a context it never saw.

That hope is an untested assumption.  This script tests it directly and cheaply:
for the same chunk, how much do h_l and K_l/V_l each move when the surrounding
context is taken away?  If K/V moves just as much as h, Family B's premise is dead
before any of it is built.

CONDITIONS (all on the same target chunk of c tokens)
-----------------------------------------------------
  CTX      chunk sits at offset p inside the real document, true positions.
           This is ground truth -- what the states "should" be.
  ISO_POS  chunk forwarded alone, but keeping its true positions p..p+c-1.
           Difference from CTX isolates the loss of *attention context*.
  ISO      chunk forwarded alone with chunk-local positions 0..c-1.
           This is exactly Encbank's write condition.
           Difference from ISO_POS is the write-side *position* effect, which is an
           EXACT ZERO -- see the note below.  It is a plumbing check, not a finding.
  CTX_ALT  chunk sits at offset p but preceded by a DIFFERENT document.
           This is the null scale: how far do the states legitimately move when
           the surrounding context is swapped for other real text?  A distance is
           only "large" relative to this.

  So:  d(ISO_POS, CTX)  = attention-context binding  <- the quantity of interest
       d(ISO, ISO_POS)  = write-side position effect <- exactly zero, a check
       d(ISO, CTX)      = total, what Encbank actually incurs
       d(CTX_ALT, CTX)  = null scale

  Why the position pair is exactly zero: RoPE is translation invariant.  For a chunk
  forwarded alone, adding a constant to every position leaves every relative offset
  unchanged, so ISO and ISO_POS are bit-identical.  Encbank's use of chunk-local
  positions at WRITE time is therefore provably a no-op, and the entire write-side
  effect is attention context.  cos exactly 1.000 here means the position plumbing is
  right; anything else is a bug.  (The position question that is NOT trivial is
  read-side re-packing, where chunks far apart in the document become adjacent in the
  read pack so the relative geometry genuinely changes.  Separate experiment.)

METRICS (per layer, per tensor kind)
------------------------------------
  cos        mean per-token cosine similarity
  cos_c      same, after removing the per-dimension mean across tokens.
             LLM residual streams have a few enormous-norm "rogue" dimensions and
             attention-sink structure that make raw cosine look near-1.0 for
             everything; the centered version is the honest one.
  relL2      ||x_a - x_b|| / ||x_b||, mean over tokens
  cka        linear CKA between the two token x dim matrices -- does the geometry
             survive even if individual vectors moved?

TENSORS CAPTURED
----------------
  h_l   hidden_states[l] = input to layer l (l = 0 is the embedding output)
  K_l   output of layer l's k_proj, after q/k RMSNorm if the model has one,
        and BEFORE RoPE.  Pre-RoPE is the right object: the design stores K
        un-rotated so that position can be re-assigned or dropped at read time.
  V_l   output of layer l's v_proj.

Nothing here is trained and nothing is generated; this is pure representation
measurement, a few forward passes.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------------
# text source
# --------------------------------------------------------------------------------------

def _wikitext_shards() -> list[Path]:
    """Locally cached wikitext-2-raw parquet shards. This machine's egress is flaky, so
    everything here reads from the HF cache rather than the network."""
    root = Path.home() / ".cache/huggingface/hub"
    hits = sorted(root.glob("datasets--wikitext/snapshots/*/wikitext-2-raw-v1/*.parquet"))
    if not hits:
        raise SystemExit("wikitext-2-raw parquet not found in the local HF cache")
    return hits


def fetch_text(which: str, _cache_dir: Path) -> str:
    """`which` in {"main", "alt"} -- two disjoint halves of the corpus.

    CTX_ALT needs text that is real and in-distribution but shares no content with the
    target document, so that swapping it measures "different context" and not
    "different genre".  Two halves of the same corpus give exactly that.
    """
    import pyarrow.parquet as pq

    parts = []
    for shard in _wikitext_shards():
        col = pq.read_table(shard, columns=["text"])["text"].to_pylist()
        parts.append("".join(t for t in col if t))
    blob = "\n".join(parts)
    half = len(blob) // 2
    piece = blob[:half] if which == "main" else blob[half:]
    # Cap before tokenising.  We only ever consume `ctx` tokens (16k), and English
    # runs ~4 chars/token, so 400k chars is >5x headroom; feeding the tokenizer the
    # whole multi-million-character corpus just to slice off the front is minutes of
    # pure waste.
    return piece[:400_000]


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------


class Capture:
    """Grabs the residual stream h_l and the pre-RoPE K, V at every layer.

    Everything is sliced to the target chunk and moved to CPU *inside* the hook.
    Using output_hidden_states=True instead would materialise L+1 full-length copies
    on the GPU -- at 16k x 4096 x bf16 x 37 that is ~5 GB for nothing, since we only
    ever look at 512 of those positions.
    """

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.h, self.k, self.v = {}, {}, {}
        self.handles = []
        self.sl = slice(None)  # token slice to keep, set per forward

    def _proj(self, store, idx, norm):
        def hook(_mod, _inp, out):
            t = out.detach()
            if norm is not None:
                # Qwen3 applies per-head RMSNorm to K before RoPE; replicate it so the
                # captured tensor is exactly what would be stored.
                b, s, _ = t.shape
                hd = norm.weight.shape[-1]
                t = norm(t.view(b, s, -1, hd)).view(b, s, -1)
            store[idx] = t[0, self.sl].to(torch.float32).cpu()

        return hook

    def _resid(self, idx):
        # pre-hook: args[0] (or kwargs["hidden_states"]) is the input to this layer,
        # i.e. h_idx by definition.
        def hook(_mod, args, kwargs):
            t = kwargs.get("hidden_states", args[0] if args else None)
            self.h[idx] = t.detach()[0, self.sl].to(torch.float32).cpu()
            return None

        return hook

    def attach(self):
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k_norm = getattr(attn, "k_norm", None)
            self.handles += [
                layer.register_forward_pre_hook(self._resid(i), with_kwargs=True),
                attn.k_proj.register_forward_hook(self._proj(self.k, i, k_norm)),
                attn.v_proj.register_forward_hook(self._proj(self.v, i, None)),
            ]
        # h_L: the input to the final norm, after the last layer
        self.handles.append(
            self.model.model.norm.register_forward_pre_hook(
                self._resid(len(self.layers)), with_kwargs=True
            )
        )

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        # in place: the k/v hooks close over these dict objects at attach time, so
        # rebinding them here would leave the hooks writing into orphaned dicts.
        self.h.clear()
        self.k.clear()
        self.v.clear()


@torch.no_grad()
def run(model, cap, ids, position_ids, keep: slice):
    """One forward. Returns {'h': {l: [c,d]}, 'k': {...}, 'v': {...}} for the kept slice."""
    cap.clear()
    cap.sl = keep
    # Two allocations have to be suppressed or this OOMs on a shared card:
    #  1. logits_to_keep=1 kills the full-sequence logits tensor (s x 151936 x bf16 --
    #     ~5 GB at 16k).  We never read logits here.  This is the same avoidable
    #     prefill term that inflates C2's own full-context memory baseline.
    #  2. the sdpa_kernel guard keeps SDPA off the math backend, which materialises
    #     an s x s score matrix per layer (16 heads x 8192^2 x bf16 = 2.1 GB, twice
    #     over for the softmax).  Passing explicit position_ids is enough to make
    #     transformers hand SDPA a dense mask and lose is_causal, so we force the
    #     memory-efficient kernels rather than hope for them.
    #     On this box neither fused kernel is available -- an RTX 5090 is sm_120 and
    #     torch 2.7.1 ships no flash/mem-efficient kernel for it, so SDPA silently
    #     falls back to math.  We try the fused path and accept math if it is refused,
    #     which is why ctx is kept modest (4k is also C2's own decision cell).
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            model(input_ids=ids, position_ids=position_ids, use_cache=False, logits_to_keep=1)
    except RuntimeError:
        model(input_ids=ids, position_ids=position_ids, use_cache=False, logits_to_keep=1)
    return {"h": dict(cap.h), "k": dict(cap.k), "v": dict(cap.v)}


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def cka(x, y):
    """Linear CKA between two [n, d] matrices."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = (x.T @ y).norm(p="fro") ** 2
    xx = (x.T @ x).norm(p="fro") ** 2
    yy = (y.T @ y).norm(p="fro") ** 2
    return (xty / (xx.sqrt() * yy.sqrt() + 1e-12)).item()


def compare(a, b):
    """a vs reference b, both [n, d]."""
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    ac, bc = a - a.mean(0, keepdim=True), b - b.mean(0, keepdim=True)
    cos_c = F.cosine_similarity(ac, bc, dim=-1).mean().item()
    rel = ((a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-9)).mean().item()
    return {"cos": cos, "cos_c": cos_c, "relL2": rel, "cka": cka(a, b)}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def wait_for_gpu(need_gb: float, cap_gb: float, poll: int = 60, max_wait: int = 21600,
                 idle_slack_gb: float = 3.0):
    """Block until the GPU is IDLE, take the cross-process GPU lock, then hard-cap this
    process at `cap_gb`.  Thin wrapper over exp/gpu_gate.acquire_gpu -- read that file
    for the three guards and the 2026-09-05 incident that made them mandatory.  Never
    launch two GPU chains concurrently; chain jobs in ONE sequential shell."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from gpu_gate import acquire_gpu
    acquire_gpu(need_gb=need_gb, cap_gb=cap_gb, idle_slack_gb=idle_slack_gb,
                poll=min(poll, 30), max_wait=max_wait)


def main():
    ap = argparse.ArgumentParser()
    # 1.7B by default: it is the bed we can also afford to TRAIN a memory gate on.
    # Note the GQA ratio differs from the 8B bed C2 used -- 1.7B is 16/8 = 2:1, so one
    # residual (2d = 4 KB) buys exactly ONE layer of KV (4*d_kv = 4 KB), where on the
    # 8B (4:1) it buys two.  The matched-bytes contrast is therefore h_j vs KV@1 here.
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512, help="chunk size c")
    ap.add_argument("--ctx", type=int, default=16384, help="document prefix length")
    ap.add_argument("--n-chunks", type=int, default=8, help="target chunks sampled")
    ap.add_argument("--out", default="exp/results/s1_binding.json")
    ap.add_argument("--cache", default="exp/data")
    ap.add_argument("--cap-gb", type=float, default=9.0, help="hard cap on our GPU use")
    ap.add_argument("--need-gb", type=float, default=10.0, help="free VRAM required to start")
    ap.add_argument("--wait", type=int, default=7200, help="seconds to wait for free VRAM")
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb, max_wait=args.wait)
    tok = AutoTokenizer.from_pretrained(args.model)
    # sdpa, not eager: eager materialises an s x s attention matrix per layer, which at
    # s=16384 x 32 heads is ~17 GB.  The hooks are on k_proj/v_proj and on the layer
    # inputs, so they are independent of the attention kernel.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()

    cfg = model.config
    print(
        f"{args.model}: L={cfg.num_hidden_layers} d={cfg.hidden_size} "
        f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
        f"gqa={cfg.num_attention_heads // cfg.num_key_value_heads}:1"
    )

    cache = Path(args.cache)
    main_ids = tok(fetch_text("main", cache), return_tensors="pt").input_ids[0][: args.ctx]
    alt_ids = tok(fetch_text("alt", cache), return_tensors="pt").input_ids[0][: args.ctx]
    print(f"document tokens: {len(main_ids)} (alt {len(alt_ids)})")

    cap = Capture(model)
    cap.attach()

    c = args.chunk
    # sample target chunks spread across the document, skipping the first chunk
    # (it has no left context, so isolation is a no-op there)
    n_slots = len(main_ids) // c
    picks = [
        (i * (n_slots - 1)) // max(args.n_chunks - 1, 1) + 1
        for i in range(args.n_chunks)
    ]
    picks = sorted(set(p for p in picks if p < n_slots))
    print(f"target chunk indices: {picks}")

    acc = {}  # (kind, pair, layer) -> list of metric dicts

    for ci in picks:
        p = ci * c
        keep_full = slice(p, p + c)
        keep_iso = slice(0, c)
        chunk_ids = main_ids[p : p + c]

        pos_true = torch.arange(p, p + c, device=dev).unsqueeze(0)
        pos_local = torch.arange(0, c, device=dev).unsqueeze(0)

        # CTX: chunk inside the real document, true positions
        full = main_ids[: p + c].unsqueeze(0).to(dev)
        r_ctx = run(model, cap, full, torch.arange(p + c, device=dev).unsqueeze(0), keep_full)

        # CTX_ALT: same offset, but preceded by unrelated text -> null scale
        alt = torch.cat([alt_ids[:p], chunk_ids]).unsqueeze(0).to(dev)
        r_alt = run(model, cap, alt, torch.arange(p + c, device=dev).unsqueeze(0), keep_full)

        # ISO_POS: chunk alone, true positions
        solo = chunk_ids.unsqueeze(0).to(dev)
        r_isop = run(model, cap, solo, pos_true, keep_iso)

        # ISO: chunk alone, chunk-local positions (Encbank's write condition)
        r_iso = run(model, cap, solo, pos_local, keep_iso)

        # How much left context does a chunk need in order to be written correctly?
        # Encbank's write pass forwards the chunk with nothing in front of it, which is
        # the regime where attention has no sink to park mass in.  Its read pack does
        # prepend a sink, but by then the stored state was already computed without
        # one.  These arms price that: one sink token, then short real prefixes.
        prefix_arms = {}
        for name, npre in (("sink1", 0), ("pre32", 32), ("pre128", 128)):
            if npre == 0:
                pre = torch.tensor([tok.bos_token_id or tok.eos_token_id])
            else:
                pre = main_ids[max(p - npre, 0) : p]
            seq = torch.cat([pre, chunk_ids]).unsqueeze(0).to(dev)
            npre_actual = seq.shape[1] - c
            prefix_arms[name] = run(
                model, cap, seq,
                torch.arange(npre_actual + c, device=dev).unsqueeze(0),
                slice(npre_actual, npre_actual + c),
            )

        pairs = {
            "attn_ctx": (r_isop, r_ctx),   # loss of attention context
            "position": (r_iso, r_isop),   # position relocation only
            "total": (r_iso, r_ctx),       # what Encbank incurs
            "null": (r_alt, r_ctx),        # swap the surrounding document
            **{k: (v, r_ctx) for k, v in prefix_arms.items()},
        }
        for pair, (ra, rb) in pairs.items():
            for kind in ("h", "k", "v"):
                for l in ra[kind]:
                    acc.setdefault((kind, pair, l), []).append(compare(ra[kind][l], rb[kind][l]))
        print(f"  chunk {ci} @tok {p} done")

    cap.detach()

    # average over target chunks
    rows = []
    for (kind, pair, l), ms in sorted(acc.items()):
        row = {"kind": kind, "pair": pair, "layer": l, "n": len(ms)}
        for m in ("cos", "cos_c", "relL2", "cka"):
            row[m] = sum(x[m] for x in ms) / len(ms)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "config": {
                    "L": cfg.num_hidden_layers,
                    "d": cfg.hidden_size,
                    "n_heads": cfg.num_attention_heads,
                    "n_kv_heads": cfg.num_key_value_heads,
                    "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                },
                "args": vars(args),
                "chunks": picks,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"\nwrote {out}  ({len(rows)} rows)")

    # console summary: centered cosine, the honest metric
    print("\ncentered cosine vs ground-truth CTX (mean over chunks)")
    print("layer |   h:attn  h:pos  h:tot  h:null |   k:attn  k:pos  k:tot  k:null |   v:tot  v:null")
    L = cfg.num_hidden_layers
    for l in range(0, L + 1, max(1, L // 12)):
        def g(kind, pair):
            hit = [r for r in rows if r["kind"] == kind and r["pair"] == pair and r["layer"] == l]
            return hit[0]["cos_c"] if hit else float("nan")
        print(
            f"{l:5d} | {g('h','attn_ctx'):7.3f}{g('h','position'):7.3f}{g('h','total'):7.3f}{g('h','null'):7.3f} "
            f"| {g('k','attn_ctx'):8.3f}{g('k','position'):7.3f}{g('k','total'):7.3f}{g('k','null'):7.3f} "
            f"| {g('v','total'):8.3f}{g('v','null'):7.3f}"
        )


if __name__ == "__main__":
    main()
