"""Does the dial's arithmetic saving become a wall-clock saving? Measured, not derived.

sec:exp-dial observes that from j=12 to j=33 the upper band recomputed at read falls from
24/36 to 3/36 of the stack -- eight times less arithmetic per query. That is a ratio of layer
counts. Latency in this paper had only ever been measured at j=12, so the sentence paired a
measured recall claim with a derived compute claim and invited the reader to take both as
measured. S39 measures it.

Why it might not translate, and this is the paper's own roofline argument (S14/S29): the dial
trades arithmetic for BYTES. The repaired read streams 8+4j KB/token of store, so j=12 -> j=33
is 56 -> 140 KB/token, 2.5x more bytes for 8x less arithmetic, on hardware whose ridge S29
puts at 117 FLOP/byte. A FLOP saving on the bandwidth-bound side of the ridge buys nothing.

Protocol, inherited from S16 unchanged: Qwen3-8B, niah_multikey_1, 32k, pack 6.7k, one
process, one card, median of `reps` timed repetitions after warm-up, torch.cuda.synchronize()
around every phase. Arms pub / fix_all / j0; fix_S is absent because {1,9,11} is a j=12 layer
set and is undefined at j=24 and j=33.

WHAT THIS CANNOT SHOW, carried from S16 and not weakened: one card, one prompt shape, a
prototype that caches all lower layers and masks the unused ones and rebuilds its per-layer
mask in Python at every decode step, no batching, no concurrency, no store I/O, no retrieval.
These are UPPER BOUNDS on the repaired arms. The comparison that is meaningful here is
BETWEEN DEPTHS under one identical harness, not against a serving system.
"""
import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
DEPTHS = [(12, "s39_latency_j12.json"), (24, "s39_latency_j24.json"),
          (33, "s39_latency_j33.json")]
ARMS = ("pub", "fix_all", "j0")


def load(name):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def agg(d, arm, field):
    """Median over prompts of the per-prompt value the harness already reduced over reps."""
    vals = [r[field] for r in d["rows"] if r.get("arm") == arm and field in r]
    return st.median(vals) if vals else None


print("Qwen3-8B, niah_multikey_1, 32k, one card, one process, median of 3 after warm-up.")
print("Repaired-arm store = 8+4j KB/token. Upper band recomputed at read = (36-j)/36.\n")
print(f"{'j':>3s} {'KB/tok':>7s} {'recomp':>7s} | {'arm':8s} {'write':>9s} {'prefill':>9s} "
      f"{'decode':>12s} {'peak GiB':>9s}")
rows = {}
for j, name in DEPTHS:
    d = load(name)
    if d is None:
        print(f"{j:3d}  missing {name}")
        continue
    for arm in ARMS:
        # the harness stores seconds; convert at the point of use so the printed units are
        # the ones the paper quotes (ms for phases, ms/token for decode)
        w = agg(d, arm, "write_s")
        pf = agg(d, arm, "prefill_s")
        dec = agg(d, arm, "decode_s_per_tok")
        w = None if w is None else 1000 * w
        pf = None if pf is None else 1000 * pf
        dec = None if dec is None else 1000 * dec
        pk, kb = agg(d, arm, "peak_gib"), agg(d, arm, "kb_per_token")
        if dec is None:
            continue
        rows[(j, arm)] = dict(write=w, prefill=pf, decode=dec, peak=pk, kb=kb)
        # the lead column describes the DEPTH, so it must carry the repaired arm's store
        # (8+4j), not whichever arm happens to print first -- pub stores only h_j and
        # would make every row read 8.0 KB/tok.
        lead = f"{j:3d} {8 + 4 * j:7.1f} {(36-j)}/36" if arm == ARMS[0] else " " * 19
        print(f"{lead:>19s} | {arm:8s} {w if w is not None else -1:8.0f}m "
              f"{pf if pf is not None else -1:8.0f}m {dec:9.1f} ms/tok "
              f"{pk if pk is not None else -1:8.2f}")
    print()

print("=" * 78)
print("THE QUESTION: does 8x less arithmetic (24/36 -> 3/36) show up in the clock?\n")
base = rows.get((12, "fix_all"))
if base:
    for j in (12, 24, 33):
        r = rows.get((j, "fix_all"))
        if not r:
            continue
        print(f"  fix_all at j={j:2d} ({r['kb']:.0f} KB/tok, recompute {(36-j)}/36): "
              f"prefill {r['prefill']:.0f} ms ({r['prefill']/base['prefill']:.2f}x), "
              f"decode {r['decode']:.1f} ms/tok ({r['decode']/base['decode']:.2f}x), "
              f"write {r['write']:.0f} ms ({r['write']/base['write']:.2f}x)")
    print()
    print("  Read the three columns separately -- they do not move together:")
    print("   * prefill is where the arithmetic saving lands (it is the phase that runs the")
    print("     upper band over the whole pack);")
    print("   * decode streams the store every token, so it tracks BYTES, not layers;")
    print("   * write grows with j because more lower-band K/V is captured and stored.")
for j in (12, 24, 33):
    r, b = rows.get((j, "fix_all")), rows.get((j, "j0"))
    if r and b:
        print(f"\n  j={j:2d}: fix_all decode {r['decode']:.1f} vs full-replay j0 {b['decode']:.1f} "
              f"ms/tok ({r['decode']-b['decode']:+.1f}); prefill {r['prefill']:.0f} vs "
              f"{b['prefill']:.0f} ms; peak {r['peak']:.2f} vs {b['peak']:.2f} GiB")
