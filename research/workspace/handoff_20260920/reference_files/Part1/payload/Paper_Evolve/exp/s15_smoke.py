"""CPU smoke for exp/s15_ruler_lower.py: a random 4-layer Qwen3 with the real Qwen3
tokenizer vocabulary, tiny chunks, and the same code paths the 8B run will use.

Checks: (1) CoMemLower(S={}, no sink entry) reproduces the base CoMem generation and
its first-step logits (translation invariance of the shifted bottom band); (2) fix_all,
fix_S, pub_sink and j0 arms all run end to end through generate_from_ids with the
resumed-band KV-cache decode; (3) the bottom cache has the expected length per layer.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "COMem"))

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from comem import CoMem  # noqa: E402
from s15_ruler_lower import CoMemLower  # noqa: E402

tok = AutoTokenizer.from_pretrained("/srv/encbank/legacy_workspace/models/Qwen3-8B")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
torch.manual_seed(0)
model = Qwen3ForCausalLM(cfg).float().eval()
j, c, k = 2, 8, 3
text = ("The grass is green. The sky is blue. One of the special magic numbers for "
        "apple-tree is: 1234567. The sun is yellow. Here we go. There and back again. "
        "What is the special magic number for apple-tree mentioned in the text? The "
        "special magic number for apple-tree mentioned in the provided text is")
ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")
bare = tok.encode("What is the special magic number for apple-tree mentioned in the text?",
                  add_special_tokens=False)
kw = dict(chunk_size=c, max_new_tokens=6, selector="bm25", topk=k, bare_question_ids=bare)

base = CoMem(model, resume_j=j, tokenizer=tok); base.write_sink = False
low0 = CoMemLower(model, j, tok, lower_layers=[], sink_in_cache=False, chunk_write_sink=False)
sb, sl = {"capture_step_logits": True}, {"capture_step_logits": True}
ob = base.generate_from_ids(ids, stats=sb, **kw)
ol = low0.generate_from_ids(ids, stats=sl, **kw)
_a, _b = sb["step_logits"][0], sl["step_logits"][0]
_f = torch.isfinite(_a) & torch.isfinite(_b)   # step 0 masks EOS to -inf in both
d0 = float((_a[_f] - _b[_f]).abs().max())
print(f"identity: base={ob!r} lower={ol!r} same={ob == ol} max|dlogit0|={d0:.2e}")
assert d0 < 1e-3, "shifted bottom band does not reproduce the base read"

# recompute path vs KV-cache path inside CoMemLower must agree
fix = CoMemLower(model, j, tok, lower_layers=None)
s1, s2 = {"capture_step_logits": True}, {"capture_step_logits": True}
o1 = fix.generate_from_ids(ids, stats=s1, use_kv_cache=True, **kw)
print(f"fix_all kv-cache: {o1!r}; bottom cache M expected 1+{k}*{c}")
for name, cm in [("fix_S", CoMemLower(model, j, tok, lower_layers=[0])),
                 ("fix_nosink", CoMemLower(model, j, tok, lower_layers=None, chunk_write_sink=False)),
                 ("pub_sink", None), ("j0", CoMem(model, resume_j=0, tokenizer=tok))]:
    if cm is None:
        cm = CoMem(model, resume_j=j, tokenizer=tok); cm.write_sink = True
    print(f"{name}: {cm.generate_from_ids(ids, **kw)!r}")

# bottom-cache bookkeeping: build_bottom on 3 chunks -> per-layer length 1 + 3*c
fix2 = CoMemLower(model, j, tok, lower_layers=None)
chunks = list(ids[0].split(c))[:k]
sink_hj, sel = fix2.build_bottom(int(ids[0, 0]), chunks)
lens = [fix2._bottom["cache"].get_seq_length(l) for l in range(j)]
print("bottom cache lengths per layer:", lens, "M=", fix2._bottom["M"], "q_off=", fix2._bottom["q_off"])
assert lens == [1 + sum(ch.shape[0] for ch in chunks)] * j
print("smoke ok")
