"""Align SDPA input heads with the native reader without expanding stored KV.

This inference candidate keeps decode_v2's graph and cache organization. Only
temporary SDPA K/V inputs repeat KV heads; enable_gqa=False lets PyTorch choose
a supported backend. It does not force or promise an efficient kernel.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from optimized_sparse_reader import OptimizedSparseEncbankReader
from sparse_reader import KVPair


class BackendAlignedSparseEncbankReader(OptimizedSparseEncbankReader):
    BACKEND_IMPLEMENTATION = "temporary-repeat-kv-enable-gqa-false-v1"

    def configuration(self):
        result = super().configuration()
        result["attention_backend_policy"] = self.BACKEND_IMPLEMENTATION
        result["prefill_implementation"] = "same-graph-with-temporary-kv-head-expansion"
        result["persistent_kv_heads"] = self.num_kv_heads
        result["sdpa_input_kv_heads"] = self.num_heads
        return result

    def _repeat_kv_for_sdpa(self, tensor):
        # Same grouped-head order as HF repeat_kv / repeat_interleave(dim=1).
        # This result is temporary and never assigned to query/doc/hot state.
        batch, heads, length, width = tensor.shape
        if heads != self.num_kv_heads:
            raise ValueError("Persistent/input KV must retain the model's KV-head count")
        groups = self.num_heads // self.num_kv_heads
        if groups == 1:
            return tensor
        return tensor[:, :, None, :, :].expand(
            batch, heads, groups, length, width).reshape(batch, heads * groups, length, width)

    def _sdpa_expanded(self, q, k, v, *, mask, dropout, causal, layer_index):
        attn = self.encbank.layers[layer_index].self_attn
        return F.scaled_dot_product_attention(
            q, self._repeat_kv_for_sdpa(k), self._repeat_kv_for_sdpa(v),
            attn_mask=mask, dropout_p=dropout, is_causal=causal,
            scale=float(getattr(attn, "scaling", self.head_dim ** -0.5)),
            enable_gqa=False)

    def _attention(self, q, own_k, own_v, prefix: KVPair, layer_index):
        # This is the reference rectangular/square causal construction verbatim.
        plen, t = prefix.length, int(q.shape[-2])
        if plen:
            k = torch.cat((prefix.k, own_k), dim=-2)
            v = torch.cat((prefix.v, own_v), dim=-2)
        else:
            k, v = own_k, own_v
        mask = None
        causal = plen == 0 and t > 1
        if plen and t > 1:
            rows = torch.arange(t, device=q.device).unsqueeze(-1) + plen
            cols = torch.arange(plen + t, device=q.device).unsqueeze(0)
            mask = (cols <= rows).unsqueeze(0).unsqueeze(0)
        attn = self.encbank.layers[layer_index].self_attn
        dropout = float(getattr(attn, "attention_dropout", 0.0)) if self.training else 0.0
        return self._sdpa_expanded(q, k, v, mask=mask, dropout=dropout,
                                   causal=causal, layer_index=layer_index)

    def _decode_layer(self, layer_index, hidden, positions, rope_cache,
                      own_past, document_kv=None):
        # Identical projection, RoPE, one-document-cat and residual organization
        # to decode_v2. Only the SDPA invocation below differs.
        layer = self.encbank.layers[layer_index]
        attn = layer.self_attn
        z = layer.input_layernorm(hidden)
        b, t = z.shape[:2]
        q = attn.q_proj(z).view(b, t, self.num_heads, self.head_dim)
        k = attn.k_proj(z).view(b, t, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(z).view(b, t, self.num_kv_heads, self.head_dim)
        q = self._decode_rotate(attn.q_norm(q).transpose(1, 2), positions, rope_cache)
        k = self._decode_rotate(attn.k_norm(k).transpose(1, 2), positions, rope_cache)
        v = v.transpose(1, 2)
        new_own = KVPair(k, v)
        parts = ([document_kv] if document_kv is not None else []) + [own_past, new_own]
        packed = (new_own if document_kv is None and not own_past.length
                  else self._cat_kv(parts))
        attention = self._sdpa_expanded(q, packed.k, packed.v, mask=None, dropout=0.0,
                                        causal=False, layer_index=layer_index)
        attention = attention.transpose(1, 2).reshape(b, t, -1)
        out = hidden + attn.o_proj(attention)
        out = out + layer.mlp(layer.post_attention_layernorm(out))
        return out, self._cat_kv([own_past, new_own])
