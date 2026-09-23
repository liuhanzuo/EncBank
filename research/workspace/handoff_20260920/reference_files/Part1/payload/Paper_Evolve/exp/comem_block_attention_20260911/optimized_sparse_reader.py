"""Eval-only one-token decode cleanup; the attention graph/backend stay unchanged.

Prefill, routing, teacher forcing, training and persistent HotBlock serialization
are inherited verbatim. This class references the supplied CoMem model; it does
not copy or merge weights. The original reader remains the numerical reference.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from sparse_reader import KVPair, SparseCoMemReader, SparseQueryState


class OptimizedSparseCoMemReader(SparseCoMemReader):
    DECODE_IMPLEMENTATION = "shared-rope-single-document-cat-v1"

    def configuration(self):
        result = super().configuration()
        result["decode_implementation"] = self.DECODE_IMPLEMENTATION
        result["prefill_implementation"] = "unchanged-sparse-reader"
        return result

    def _decode_rotate(self, x, positions, rope_cache):
        # Use the actual normalized projection dtype, as _rotate does. Under
        # autocast this can differ from the FP32 residual / LoRA master dtype.
        # Separate dtype entries also preserve behavior if q and k differ.
        key = (x.device, x.dtype)
        pair = rope_cache.get(key)
        if pair is None:
            cos, sin = self.comem.rotary_emb(x[:, 0], position_ids=positions)
            pair = (cos.unsqueeze(1), sin.unsqueeze(1))
            rope_cache[key] = pair
        cos, sin = pair
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def _decode_layer(self, layer_index, hidden, positions, rope_cache,
                      own_past, document_kv=None):
        """Return hidden and query-only KV; no probe/raw-K tensors are exported."""
        layer = self.comem.layers[layer_index]
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
        # A single cat contains document -> past query -> current query. In
        # particular, never materialize cat(document,past) and then cat again.
        parts = ([document_kv] if document_kv is not None else []) + [own_past, new_own]
        # Retain even empty document entries: cat's dtype promotion is part of
        # the reference behavior. Dropping an empty FP32 entry under BF16
        # autocast could silently change the attention input dtype.
        packed = (new_own if document_kv is None and not own_past.length
                  else self._cat_kv(parts))
        attention = F.scaled_dot_product_attention(
            q, packed.k, packed.v, attn_mask=None, dropout_p=0.0,
            is_causal=False,
            scale=float(getattr(attn, "scaling", self.head_dim ** -0.5)),
            enable_gqa=self.num_heads != self.num_kv_heads)
        attention = attention.transpose(1, 2).reshape(b, t, -1)
        out = hidden + attn.o_proj(attention)
        out = out + layer.mlp(layer.post_attention_layernorm(out))
        # Persistent query state deliberately excludes document KV.
        return out, self._cat_kv([own_past, new_own])

    @torch.no_grad()
    def decode_step(self, token_id, state: SparseQueryState):
        """Append one token with the same route, positions and SDPA policy."""
        if self.training:
            raise ValueError("Call reader.eval() before cached generation")
        token = self._ids([int(token_id)])
        hidden = self.comem.embed_tokens(token)
        # These positions are distinct even after documents are physically
        # pruned. Neither is derived from the reduced late cache length.
        lower_positions = self._positions(1, state.query_position)
        upper_positions = self._positions(1, state.pack_position)
        lower_rope, upper_rope = {}, {}
        for layer in range(self.L):
            lower = layer < self.j
            hidden, own = self._decode_layer(
                layer, hidden, lower_positions if lower else upper_positions,
                lower_rope if lower else upper_rope, state.query_kv[layer],
                None if lower else state.doc_kv[layer])
            state.query_kv[layer] = own
        state.query_position += 1
        state.pack_position += 1
        return self.comem.lm_head(self.comem.norm(hidden))
