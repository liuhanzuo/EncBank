"""Encbank — Comprehension Memory.

Fixed-size mid-depth-resume long-context memory over a plain (un-patched) decoder
LLM. See :class:`encbank.model.Encbank` for the full contract.

Quick start
-----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from encbank import Encbank

    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path).cuda().eval()

    model = Encbank(lm, resume_j=12, tokenizer=tok)
    model.encode(long_document)                 # comprehend once (cache h_j)
    answer = model.generate("What is X?", selector="bm25", topk=12)
"""
from .model import Encbank
from .moe import EncbankMoE, load_moe_encbank
from . import selectors

__all__ = ["Encbank", "EncbankMoE", "load_moe_encbank", "selectors"]
