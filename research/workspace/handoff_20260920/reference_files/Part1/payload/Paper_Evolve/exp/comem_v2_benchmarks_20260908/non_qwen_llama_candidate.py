"""Isolated dense-Llama candidate; deliberately not connected to any benchmark.

Uses the stock CoMem upper reader, but does not import/patch s15_ruler_lower or
reader_adapter.  Llama has no k_norm: capture k_proj before native Llama RoPE.
Only full-head default RoPE, batch one, dense Llama and eager/SDPA are admitted.
The payload is an explicit CPU document; every read creates new runtime caches.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM, LlamaAttention, LlamaDecoderLayer, apply_rotary_pos_emb,
)

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "COMem"))
from comem.model import CoMem


class DenseLlamaCandidate(CoMem):
    """Explicit selected-chunk V2 reader; no tokenizer/retriever/weight loader."""

    def __init__(self, model, resume_j, *, model_binding, chunk_write_sink=True,
                 lower_layers=None):
        if type(model) is not LlamaForCausalLM or model.config.model_type != "llama":
            raise TypeError("candidate admits only stock dense LlamaForCausalLM")
        if model.training:
            raise ValueError("inference candidate requires model.eval()")
        super().__init__(model, resume_j)
        if getattr(self.config, "_attn_implementation", None) not in ("sdpa", "eager"):
            raise ValueError("only eager/SDPA have been adapted")
        params = self.config.rope_parameters or {}
        if params.get("rope_type", "default") != "default":
            raise ValueError("scaled/dynamic/partial RoPE remains unvalidated")
        if getattr(self.config, "partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("partial RoPE is outside this candidate")
        if getattr(self.config, "sliding_window", None) is not None:
            raise ValueError("sliding/hybrid caches are outside this candidate")
        self.nkv = int(self.config.num_key_value_heads)
        self.head_dim = int(self.config.head_dim)
        for layer in self.layers:
            if type(layer) is not LlamaDecoderLayer or type(layer.self_attn) is not LlamaAttention:
                raise TypeError("only native dense Llama layers are admitted")
            a = layer.self_attn
            if hasattr(a, "k_norm") or a.k_proj.out_features != self.nkv * self.head_dim:
                raise ValueError("unverified K layout/normalization")
            if a.v_proj.out_features != self.nkv * self.head_dim:
                raise ValueError("unverified V layout")
        if not isinstance(model_binding, str) or not model_binding:
            raise ValueError("explicit model binding is required for reusable tensors")
        self.model_binding = model_binding
        self.write_sink = bool(chunk_write_sink)
        self.lower_layers = set(range(self.resume_j)) if lower_layers is None else set(lower_layers)
        if not self.lower_layers.issubset(set(range(self.resume_j))):
            raise ValueError("lower layer selection lies outside [0,j)")
        self._bottom = None
        self.capture_calls = 0

    def _sink_prefix_id(self):
        bos = self.config.bos_token_id
        if bos is None:
            raise ValueError("explicit native BOS is required; no implicit token-zero fallback")
        return int(bos)

    def _signature(self):
        return {"schema": "dense-llama-candidate-v1", "model_binding": self.model_binding,
                "resume_j": self.resume_j, "chunk_write_sink": self.write_sink,
                "layers": self.num_layers, "hidden_size": self.hidden_size,
                "kv_heads": self.nkv, "head_dim": self.head_dim, "dtype": str(self.dtype),
                "bos_token_id": self._sink_prefix_id(), "rope_parameters": self.config.rope_parameters,
                "max_position_embeddings": int(self.config.max_position_embeddings)}

    @torch.no_grad()
    def capture_lower(self, token_ids):
        """K before RoPE, V and h_j in local chunk coordinates; no K norm."""
        ids = self._as_ids(token_ids)
        if ids.shape[1] < 1:
            raise ValueError("empty capture")
        self._check_positions(int(ids.shape[1]))
        captured, handles = {}, []
        self.capture_calls += 1
        def save(layer_id, kind):
            def hook(_module, _args, value):
                captured[layer_id, kind] = value.detach().clone()
            return hook
        try:
            for i in range(self.resume_j):
                a = self.layers[i].self_attn
                handles.append(a.k_proj.register_forward_hook(save(i, "k")))
                handles.append(a.v_proj.register_forward_hook(save(i, "v")))
            emb = self.embed_tokens(ids)
            pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
            mask, rope = self._make_mask_and_rope(emb, pos)
            hj = self._run_layers(emb, slice(0, self.resume_j), mask, pos, rope)
        finally:
            for handle in handles:
                handle.remove()
        def heads(t):
            return t.reshape(1, ids.shape[1], self.nkv, self.head_dim).transpose(1, 2).contiguous()
        return hj.detach(), {i: (heads(captured[i, "k"]), heads(captured[i, "v"]))
                             for i in range(self.resume_j)}

    @torch.no_grad()
    def rotate_key(self, k_pre, positions):
        if k_pre.shape != (1, self.nkv, positions.shape[1], self.head_dim):
            raise ValueError("K/position layout mismatch")
        cos, sin = self.rotary_emb(k_pre, position_ids=positions)
        return apply_rotary_pos_emb(k_pre, k_pre, cos, sin)[1]

    def _check_positions(self, count):
        if count > int(self.config.max_position_embeddings):
            raise ValueError("pack exceeds explicitly admitted native context limit")

    @torch.no_grad()
    def write_document(self, chunks):
        if not chunks or any(len(c) == 0 for c in chunks):
            raise ValueError("nonempty document chunks required")
        def payload(ids, drop):
            h, kv = self.capture_lower(ids)
            # Clone after removing local BOS, so dropped storage is not retained.
            return {"h": h[:, drop:].detach().cpu().contiguous().clone(),
                    "kv": {i: (k[:, :, drop:].detach().cpu().contiguous().clone(),
                               v[:, :, drop:].detach().cpu().contiguous().clone())
                           for i, (k, v) in kv.items()}}
        sink = payload([self._sink_prefix_id()], 0)
        entries = []
        for chunk in chunks:
            ids = list(map(int, chunk))
            prefix = [self._sink_prefix_id()] if self.write_sink else []
            entries.append({"ids": ids, **payload(prefix + ids, len(prefix))})
        return {"signature": self._signature(), "sink": sink, "chunks": entries}

    def _validate_document(self, document):
        if document.get("signature") != self._signature():
            raise ValueError("document was produced by a different model/split/write protocol")
        for entry in [document["sink"]] + document["chunks"]:
            h = entry["h"]
            n = h.shape[1]
            if h.shape != (1, n, self.hidden_size) or h.device.type != "cpu" or h.dtype != self.dtype:
                raise ValueError("invalid CPU hidden payload")
            if "ids" in entry and len(entry["ids"]) != n:
                raise ValueError("stored token count mismatch")
            if set(entry["kv"]) != set(range(self.resume_j)):
                raise ValueError("stored lower-layer coverage mismatch")
            for k, v in entry["kv"].values():
                for x in (k, v):
                    if x.shape != (1, self.nkv, n, self.head_dim) or x.device.type != "cpu" or x.dtype != self.dtype:
                        raise ValueError("invalid CPU K/V payload")
        if document["sink"]["h"].shape[1] != 1:
            raise ValueError("one standalone sink required")

    @torch.no_grad()
    def prepare_read(self, document, selected):
        """Stage CPU tensors without recapture; the lower runtime cache is fresh."""
        if self._bottom is not None:
            raise RuntimeError("finish/clear the previous read before starting another")
        self._validate_document(document)
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(document["chunks"]) for i in selected):
            raise ValueError("invalid explicit retrieval indices")
        entries = [document["sink"]] + [document["chunks"][i] for i in selected]
        cache, keys, values = DynamicCache(config=self.config), {}, {}
        off = 0
        hidden = []
        for entry in entries:
            h = entry["h"].to(self.device)
            n = h.shape[1]
            pos = torch.arange(off, off + n, device=self.device).unsqueeze(0)
            self._check_positions(off + n)
            hidden.append(h)
            for i, (k, v) in entry["kv"].items():
                keys.setdefault(i, []).append(self.rotate_key(k.to(self.device), pos))
                values.setdefault(i, []).append(v.to(self.device))
            off += n
        for i in range(self.resume_j):
            # Concatenation creates private mutable runtime storage even on CPU.
            cache.update(torch.cat(keys[i], dim=2), torch.cat(values[i], dim=2), i)
        self._bottom = {"cache": cache, "M": off}
        return hidden[0], hidden[1:]

    def clear_read(self):
        self._bottom = None

    def lower_visibility(self, layer_id, query_tokens, kv_len):
        m = self._bottom["M"]
        q_seen = kv_len - m
        if q_seen < query_tokens:
            raise ValueError("cache shorter than current query")
        allow = torch.zeros((1, 1, query_tokens, kv_len), dtype=torch.bool, device=self.device)
        allow[..., 0] = True
        if layer_id in self.lower_layers:
            allow[..., 1:m] = True
        rows = torch.arange(q_seen - query_tokens, q_seen, device=self.device)[:, None]
        cols = torch.arange(q_seen, device=self.device)[None, :]
        allow[..., m:] = cols <= rows
        return allow

    @torch.no_grad()
    def _lower_forward(self, hidden, positions):
        cache = self._bottom["cache"]
        pe = self.rotary_emb(hidden, position_ids=positions)
        for i in range(self.resume_j):
            n = cache.get_seq_length(i) + hidden.shape[1]
            allow = self.lower_visibility(i, hidden.shape[1], n)
            mask = allow
            if self.config._attn_implementation == "eager":
                mask = torch.zeros(allow.shape, dtype=self.dtype, device=self.device)
                mask.masked_fill_(~allow, torch.finfo(self.dtype).min)
            hidden = self._layer_out_hidden(self.layers[i](
                hidden, attention_mask=mask, position_ids=positions,
                position_embeddings=pe, past_key_values=cache, use_cache=True))
        return hidden

    @torch.no_grad()
    def write_prefill(self, token_ids):
        if self._bottom is None:
            raise RuntimeError("prepare_read must precede query prefill")
        ids = self._as_ids(token_ids)
        if ids.shape[1] < 1:
            raise ValueError("query must be nonempty")
        m = self._bottom["M"]
        self._check_positions(m + ids.shape[1])
        pos = torch.arange(m, m + ids.shape[1], device=self.device).unsqueeze(0)
        h = self._lower_forward(self.embed_tokens(ids), pos)
        return h, self._bottom["cache"], m + ids.shape[1]

    @torch.no_grad()
    def decode_step(self, token_id, bottom_cache, top_cache, q_local_pos, pack_pos):
        if self._bottom is None or bottom_cache is not self._bottom["cache"]:
            raise ValueError("decode cache does not belong to active read")
        if q_local_pos != pack_pos:
            raise ValueError("lower and upper query positions must share final pack coordinates")
        self._check_positions(pack_pos + 1)
        ids = self._as_ids([int(token_id)])
        pos = torch.tensor([[pack_pos]], device=self.device)
        h = self._lower_forward(self.embed_tokens(ids), pos)
        pe = self.rotary_emb(h, position_ids=pos)
        h = self._run_layers(h, slice(self.resume_j, self.num_layers),
                             self._decode_attn_mask(pack_pos + 1), pos, pe,
                             past_key_values=top_cache, use_cache=True)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def query(self, document, selected, query_ids, *, max_new_tokens, eos_ids=None):
        if max_new_tokens < 1:
            raise ValueError("positive generation cap required")
        if eos_ids is None:
            native = self.config.eos_token_id
            eos_ids = [] if native is None else native if isinstance(native, list) else [native]
        try:
            sink, chunks = self.prepare_read(document, selected)
            qh, bottom, qpos = self.write_prefill(query_ids)
            logits, top, ppos = self.read_prefill(sink, chunks, qh)
            result = {"ids": [], "step_logits": [], "stopped_on_eos": False,
                      "pack_tokens": ppos, "decoder_forwards": 0}
            for step in range(max_new_tokens):
                result["step_logits"].append(logits.detach().cpu().clone())
                token = int(logits[0, -1].argmax())
                result["ids"].append(token)
                if token in eos_ids:
                    result["stopped_on_eos"] = True
                    break
                if step + 1 < max_new_tokens:
                    logits = self.decode_step(token, bottom, top, qpos, ppos)
                    qpos += 1
                    ppos += 1
                    result["decoder_forwards"] += 1
            result["lower_cache_lengths"] = [bottom.get_seq_length(i) for i in range(self.resume_j)]
            result["upper_cache_lengths"] = [top.get_seq_length(i) for i in range(self.resume_j, self.num_layers)]
            return result
        finally:
            self.clear_read()
