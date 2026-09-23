"""Two corrections the literature survey forced, plus the bib entry one of them needs.

(1) Section 2's "Position of this work" states the novelty as a bare universal: "no existing
    method gives the reader's lower band a view of a retrieved memory while storing less than
    full-depth KV on a stock, untrained model." The survey found two methods that miss that
    conjunction on exactly ONE term each -- xRAG (all layers see a one-vector memory, but a
    modality bridge is trained) and Memory Inception (training-free, stock, sub-full-depth,
    injected into the reader's own layers, but no retrieval). Section 3 now names both and
    states the claim as a four-term conjunction; Section 2 still states it loosely, which is
    an internal inconsistency I introduced. Narrowed here to match, with the near misses
    pointed at rather than re-argued.

(2) Section 4 introduces the pre-RoPE store plus rotate-to-pack-position without attribution.
    It does not claim the format as novel -- it presents it as construction and checks it is
    exact -- but LazyAttention (arXiv:2606.04302, ICML 2026) does the same thing under the name
    "deferred positional encoding", at full depth and for whole chunks. Verified on the arXiv
    abs page 2026-09-07: "kernelizes deferred positional encoding to enable zero-copy,
    position-agnostic KV reuse". One clause of attribution, and the claim narrows to the part
    that is ours: applying it to the LOWER band and to the reader's own query path.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REL = ROOT / "sections/02_related.tex"
s = REL.read_text(encoding="utf-8")
old = """Read together they give the point of
this paper: no existing method gives the reader's lower band a view of a
\\emph{retrieved} memory while storing less than full-depth KV on a stock,
untrained model. Table~\\ref{tab:priorart} in Appendix~\\ref{app:priorart} lays
the seven prior lines out on both axes."""
new = """Read together they give the point of
this paper, which is a conjunction of four terms and not a single claim: no
existing method lets the reader's \\emph{own} lower band see a memory that is
\\emph{retrieved}, from a store below full-depth KV, on a \\emph{stock} model.
Dropping any one term admits prior work --- \\S\\ref{sec:mot:visible} names the two
methods that miss it by one term each --- and the four together admit none.
Table~\\ref{tab:priorart} in Appendix~\\ref{app:priorart} lays
the seven prior lines out on both axes."""
if s.count(old) != 1:
    sys.exit(f"ABORT related: {s.count(old)} matches")
REL.write_text(s.replace(old, new), encoding="utf-8")
print("02_related: novelty claim narrowed to the four-term conjunction")

MET = ROOT / "sections/04_methodology.tex"
m = MET.read_text(encoding="utf-8")
old = """are stored as they are; the key is stored \\emph{before} its rotation because
the rotation depends on where the chunk will sit in the read pack, which is
not known at write time."""
new = """are stored as they are; the key is stored \\emph{before} its rotation because
the rotation depends on where the chunk will sit in the read pack, which is
not known at write time. Storing keys position-agnostically and applying the
rotation at read is not new --- LazyAttention \\citep{lazyattention} defers
positional encoding this way to reuse a single physical KV copy at arbitrary
positions --- and what is ours is the band it is applied to: the \\emph{lower}
layers, read by the query's own lower band rather than by the layers above the
split."""
if m.count(old) != 1:
    sys.exit(f"ABORT methodology: {m.count(old)} matches")
MET.write_text(m.replace(old, new), encoding="utf-8")
print("04_methodology: pre-RoPE format attributed, claim narrowed to the band")

BIB = ROOT / "qcmem.bib"
b = BIB.read_text(encoding="utf-8")
if re.search(r"@\w+\s*\{\s*lazyattention\s*,", b):
    sys.exit("ABORT: lazyattention already in bib")
entry = """
@inproceedings{lazyattention,
  title     = {{LazyAttention}: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding},
  author    = {Xia, Haocheng and Pamnani, Mihir and Fang, Hanxi and Chockchowwat, Supawit and Park, Yongjoo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  eprint    = {2606.04302},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2606.04302}
}
"""
BIB.write_text(b.rstrip() + "\n" + entry, encoding="utf-8")
print("qcmem.bib: +lazyattention")
