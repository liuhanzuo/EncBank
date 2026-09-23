"""CacheBlend-style Qwen3 port with contextual layer-1 V selection.

The public CacheBlend release defaults to check_layers=[1], ratio=.16, and
summed squared V error with floor(context_tokens * ratio), preserving suffix.
This readable port fully forwards BOTH bootstrap layers, then crops hidden
states. That produces the same selected hidden states but does extra layer-1
attention/MLP work: report its own measured cost, never upstream performance.
Sources: github.com/YaoJiayi/CacheBlend, vllm_blend/vllm/{model_executor/models/
llama.py,attention/backends/xformers.py}. The historical local layer-0-K
ranking is preserved in COMem/comem/cacheblend.py and is not reused here.
"""
from __future__ import annotations

import math
import torch
from transformers.cache_utils import DynamicCache

from comem.cacheblend import CacheBlend, _SparseWriteCache


class ContextualCacheBlend(CacheBlend):
    @torch.no_grad()
    def read(self, pack_ids, merged_kv, sink_len, query_len, recompute_ratio, stats=None):
        if self.num_layers < 2:
            raise ValueError("Contextual layer-1 selection requires at least two layers")
        ratio = float(recompute_ratio)
        if not 0 <= ratio <= 1:
            raise ValueError("recompute_ratio must be in [0,1]")
        ids = self.cm._as_ids(pack_ids)
        length = ids.shape[1]
        if sink_len < 0 or query_len < 1 or sink_len + query_len > length:
            raise ValueError("Invalid explicit sink/context/query boundaries")
        embeds = self.cm.embed_tokens(ids)
        positions = torch.arange(length, device=self.device).unsqueeze(0)
        mask, pe = self.cm._make_mask_and_rope(embeds, positions)
        fresh = DynamicCache(config=self.config)
        hidden = self.cm._run_layers(embeds, slice(0, 2), mask, positions, pe,
                                     past_key_values=fresh, use_cache=True)
        context_end = length - int(query_len)
        context_len = context_end - int(sink_len)
        deviation = (fresh.layers[1].values.float() - merged_kv[1][1].float()).square().sum(dim=(1, 3))[0]
        n_recompute = min(context_len, int(math.floor(ratio * context_len)))
        selected = torch.zeros(length, dtype=torch.bool, device=self.device)
        selected[:sink_len] = True
        selected[context_end:] = True
        if n_recompute:
            top = torch.argsort(deviation[sink_len:context_end], descending=True, stable=True)[:n_recompute] + sink_len
            selected[top] = True
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        mixed = [(fresh.layers[layer].keys, fresh.layers[layer].values) for layer in range(2)]
        mixed += [(key.clone(), value.clone()) for key, value in merged_kv[2:]]
        hidden = hidden[:, indices, :]
        rope = positions[:, indices]
        pe_selected = self.cm.rotary_emb(hidden, position_ids=rope)
        causal = self._sparse_attn_mask(torch.arange(length, device=self.device).view(1, -1) <= indices.view(-1, 1))
        for layer in range(2, self.num_layers):
            cache = _SparseWriteCache(*mixed[layer], indices)
            out = self.cm.layers[layer](hidden, attention_mask=causal, position_ids=rope,
                position_embeddings=pe_selected, past_key_values=cache, use_cache=True)
            hidden = self.cm._layer_out_hidden(out)
            mixed[layer] = cache.keys, cache.values
        logits = self.cm.lm_head(self.cm.norm(hidden))
        if stats is not None:
            stats.update(cacheblend_variant="qwen3_layer1_v_full_two_layer_bootstrap",
                recompute_ratio=ratio, n_recompute_ctx=n_recompute, n_context_tokens=context_len,
                pack_len=length, selected_positions=indices.tolist(),
                context_v_error_max=float(deviation[sink_len:context_end].max()) if context_len else 0.0,
                bootstrap_full_layers=2, cacheblend_kv_bytes_per_tok=self.kv_bytes_per_tok())
        return logits, indices, mixed

    @torch.no_grad()
    def generate_explicit(self, selected_chunks, query_ids, bos_id, eos_id, max_new_tokens,
                          stats=None, chunk_write_sink=False):
        """Use an externally fixed retrieval pack; no second retrieval or truncation.

        Default no write sink matches the original public CacheBlend-style local
        chunk prefill. Optional write sink is explicit and changes the cache.
        """
        pieces = [self.cm._as_ids([bos_id])]
        pieces.extend(self.cm._as_ids(ids) for ids in selected_chunks)
        pieces.append(self.cm._as_ids(query_ids))
        caches, offsets = [], []
        offset = 0
        for index, ids in enumerate(pieces):
            has_sink = chunk_write_sink and 0 < index < len(pieces) - 1
            local_ids = torch.cat([pieces[0], ids], 1) if has_sink else ids
            cached, _ = self.prefill_chunk_full(local_ids)
            if has_sink:
                cached = [(key[:, :, 1:, :], value[:, :, 1:, :]) for key, value in cached]
            caches.append(cached)
            offsets.append(offset - int(has_sink))
            offset += ids.shape[1]
        pack = torch.cat(pieces, 1)
        merged = self.concat_kv_reindex(caches, offsets)
        logits, indices, mixed = self.read(pack, merged, 1, pieces[-1].shape[1], self.recompute_ratio, stats)
        next_logits = logits[0, -1].float()
        if eos_id is not None:
            next_logits[eos_id] = -torch.inf
        tokens = [int(next_logits.argmax())]
        cache = self.decode_cache(mixed)
        for step in range(1, max_new_tokens):
            next_logits = self.decode_step(tokens[-1], cache, offset + step - 1)[0, -1]
            token = int(next_logits.argmax())
            if token == eos_id:
                break
            tokens.append(token)
        return tokens
