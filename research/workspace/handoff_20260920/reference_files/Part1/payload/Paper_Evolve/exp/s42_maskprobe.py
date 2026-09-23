"""Does `_bottom_mask` actually open the chunk columns for a sparse layer set at large j?

S41 found `fix_S` bit-identical to `pub` on 50/50 samples in every cell at j=33, which is the
signature of an arm that is not doing what its name says (the E54 failure mode). Reading the
code did not settle it: `_capture_lower` hooks every layer in range(resume_j) regardless of
the set, and `_bottom_mask` gates chunk visibility on `l in self.lower_layers`, both of which
look right. This exercises the mask itself, with no model and no GPU, so one link in the
chain stops being an inference.

The mask is called as a bound method, so we borrow it onto a stub carrying only the four
attributes it touches: `_bottom["M"]`, `sink_in_cache`, `lower_layers`, `device`. If the mask
is correct, then for a layer IN the set the chunk columns 1..M-1 must be open, and for a
layer OUT of the set they must be closed, at every j -- and in particular at j=33, where the
suspicious result came from.

WHAT THIS CANNOT SHOW: that the rest of the read path uses the mask it is given. It isolates
the mask, nothing more. If the mask is correct the fault (if any) is downstream.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s15_ruler_lower import CoMemLower  # noqa: E402


class Stub:
    """Carries exactly the attributes _bottom_mask touches."""
    def __init__(self, layers, M, sink_in_cache=True):
        self._bottom = {"M": M}
        self.sink_in_cache = sink_in_cache
        self.lower_layers = set(layers)
        self.device = "cpu"


def probe(layers, M=13, T=4, nq=3, sink_in_cache=True):
    s = Stub(layers, M, sink_in_cache)
    kv_len = M + nq
    out = {}
    for l in sorted(set(layers) | {max(layers) + 1 if layers else 0}):
        m = CoMemLower._bottom_mask(s, l, T, kv_len)[0, 0]
        chunk_open = bool(m[:, 1:M].any()) if sink_in_cache else bool(m[:, :M].any())
        sink_open = bool(m[:, 0].all())
        out[l] = (chunk_open, sink_open, int(m[:, 1:M].sum()))
    return out


print("Unit-probing CoMemLower._bottom_mask with no model. M=13 cache cols (1 sink + 12"
      " chunk), T=4 query rows, 3 query cols.\n")
ok = True
for name, layers, j in (("the j=12 set that works in the paper", [1, 9, 11], 12),
                        ("S41's sparse set at j=33", [2, 8, 14, 20, 26, 32], 33),
                        ("S42's dense set at j=33", list(range(2, 33, 2)), 33)):
    print(f"{name}: lower_layers={sorted(layers)[:6]}{'...' if len(layers) > 6 else ''} "
          f"({len(layers)} layers, j={j})")
    res = probe(layers)
    for l, (chunk_open, sink_open, ncols) in sorted(res.items()):
        inset = l in layers
        verdict = "OK" if chunk_open == inset else "*** WRONG ***"
        if chunk_open != inset:
            ok = False
        print(f"    layer {l:2d} {'IN ' if inset else 'OUT'} set: chunk cols open="
              f"{chunk_open} ({ncols} cells), sink col open={sink_open}   {verdict}")
    print()

print("=" * 72)
if ok:
    print("MASK IS CORRECT at every layer tested, including both j=33 sets.")
    print("So the sparse arm's chunk columns ARE opened at the layers in its set, and the")
    print("S41 result is not a masking bug. If fix_S still equals pub at j=33, the cause is")
    print("downstream of the mask -- or the effect is real and both arms sit at the floor.")
else:
    print("MASK IS WRONG -- the S41 result is an artefact and E64 must be withdrawn.")
sys.exit(0 if ok else 1)
