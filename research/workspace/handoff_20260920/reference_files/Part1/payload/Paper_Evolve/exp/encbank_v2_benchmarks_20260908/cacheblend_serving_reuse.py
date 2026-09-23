"""Independent write-once adapter for the existing contextual CacheBlend port.

This module adds no scheduler or GPU entry point. The separate
cacheblend_serving_driver.py applies the existing shared GPU gate before
constructing a GPU model through the original serving harness. CPU uses
are correctness tests only. Existing serving_reuse.py and completed attempts
are deliberately not modified.
"""
from __future__ import annotations

from pathlib import Path
import torch

from serving_reuse import ReusableReader, map_tensors, save_json, save_tensor_file, tensor_bytes, selectors
from cacheblend_contextual import ContextualCacheBlend


def kv_inventory(kv_layers):
    """Actual live KV tensors: logical elements and unique backing storages.

    Storage accounting counts a whole backing storage once, including unused
    space of a view. It is an inventory at this call, not a transient peak,
    allocator reservation or whole-GPU resident memory measurement.
    """
    tensors = [tensor for pair in kv_layers for tensor in pair]
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
    return {"tensor_count": len(tensors), "logical_tensor_bytes": tensor_bytes(tensors),
            "unique_storage_bytes": sum(storages.values()), "unique_storages": len(storages),
            "sequence_lengths": [int(pair[0].shape[-2]) for pair in kv_layers]}


class ReusableContextualCacheBlend(ReusableReader):
    """Full-depth stored KV with genuine 16% contextual selective recomputation."""
    def __init__(self, model, j, tokenizer=None, model_id="unspecified", recompute_ratio=.16):
        # This reuses the exact stock-backbone/device checks and native attention
        # policy already imported by serving_reuse, without changing that module.
        super().__init__(model, j, tokenizer, arm="pub", model_id=model_id)
        self.arm = "cacheblend16"
        self.cm.write_sink = False
        self.cb = ContextualCacheBlend(self.cm, recompute_ratio)

    def signature(self):
        signature = super().signature()
        signature.update(cacheblend_variant="qwen3_layer1_v_full_two_layer_bootstrap",
            recompute_ratio=self.cb.recompute_ratio, check_layer_zero_based=1,
            bootstrap_full_layers=2, context_selection="floor(ratio*n_context); stable descending V squared error",
            chunk_write_sink=False, persistent_key_space="post-k_norm/post-RoPE, chunk-local position zero",
            query_slots="zeros; every query position overwritten at every layer")
        return signature

    @torch.no_grad()
    def write_store(self, context_ids, path, chunk_size=512):
        path = Path(path)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if path.exists() and any(path.iterdir()):
            raise ValueError("Write requires a new empty store directory")
        path.mkdir(parents=True, exist_ok=True)
        if self.cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.cm.device)
            baseline = torch.cuda.memory_allocated(self.cm.device)
        else:
            baseline = 0
        started = self.clock()
        ids = torch.as_tensor(context_ids, dtype=torch.long, device="cpu").reshape(-1).clone()
        if ids.numel() == 0:
            raise ValueError("Document must be nonempty")
        chunks = [chunk.clone() for chunk in ids.split(chunk_size)]
        compute_s = copy_s = serialize_s = 0.0
        payload_bytes = 0
        t = self.clock()
        save_tensor_file(chunks, path / "tokens.pt")
        serialize_s += self.clock() - t
        entries = [("sink.pt", [self.cm._sink_prefix_id()])]
        entries += [(f"chunk_{i:06d}.pt", chunk) for i, chunk in enumerate(chunks)]
        for filename, chunk in entries:
            t = self.clock()
            kv, _ = self.cb.prefill_chunk_full(chunk)
            compute_s += self.clock() - t
            t = self.clock()
            payload = {"kv": map_tensors(kv, lambda x: x.detach().to("cpu").contiguous().clone())}
            copy_s += self.clock() - t
            payload_bytes += tensor_bytes(payload)
            t = self.clock()
            save_tensor_file(payload, path / filename)
            serialize_s += self.clock() - t
            del kv, payload
        save_json({"schema_version": 1, "signature": self.signature(), "n_tokens": ids.numel(),
            "n_chunks": len(chunks), "chunk_size": chunk_size,
            "payload_tensor_bytes": payload_bytes, "raw_token_bytes": tensor_bytes(chunks),
            "cache_key_space": "post-k_norm/post-RoPE K and unchanged V; full depth; chunk-local positions",
            "validity": "Reuse only with the exact checkpoint, tokenizer and recorded signature"}, path / "store.json")
        total_s = self.clock() - started
        peak = torch.cuda.max_memory_allocated(self.cm.device) if self.cm.device.type == "cuda" else 0
        return {"write_total_s": total_s, "write_compute_s": compute_s,
                "write_device_to_cpu_s": copy_s, "write_serialize_s": serialize_s,
                "serialized_bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                "payload_tensor_bytes": payload_bytes, "raw_token_bytes": tensor_bytes(chunks),
                "n_document_tokens": ids.numel(), "n_document_chunks": len(chunks),
                "capture_calls": len(entries), "write_peak_allocated_bytes": peak,
                "write_incremental_peak_bytes": max(0, peak - baseline)}

    def _prepare_pack(self, payloads, token_pieces, query):
        """Query KV is disposable: suffix positions are always recomputed.

        Unlike generate_explicit, this avoids a redundant isolated full-depth
        query prefill. Fresh bootstrap layers ignore the slots; higher layers
        overwrite every suffix slot before use. CPU tests compare exact logits
        and greedy sequences with generate_explicit's freshly prefetched query.
        """
        offsets, offset = [], 0
        caches = [payload["kv"] for payload in payloads]
        for piece in token_pieces:
            offsets.append(offset)
            offset += piece.numel()
        sample = caches[0][0][0]
        shape = (sample.shape[0], sample.shape[1], query.numel(), sample.shape[3])
        query_slots = [(torch.zeros(shape, dtype=sample.dtype, device=sample.device),
                        torch.zeros(shape, dtype=sample.dtype, device=sample.device))
                       for _ in range(self.cb.num_layers)]
        caches.append(query_slots)
        offsets.append(offset)
        pack = torch.cat([piece.reshape(1, -1) for piece in token_pieces] + [query.reshape(1, -1)], dim=1)
        merged = self.cb.concat_kv_reindex(caches, offsets)
        return pack, merged

    @torch.no_grad()
    def query_ids(self, query_ids, *, selected_indices=None, selector="bm25", topk=12,
                  bare_question_ids=None, max_new_tokens=16, force_length=False,
                  capture_logits=False, capture_inventory=False):
        if self.path is None or not self.chunks:
            raise ValueError("Open a complete document store before querying")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        query = torch.as_tensor(query_ids, dtype=torch.long, device="cpu").reshape(-1)
        if query.numel() == 0:
            raise ValueError("Query must be nonempty")
        cm = self.cm
        if cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cm.device)
            baseline = torch.cuda.memory_allocated(cm.device)
        else:
            baseline = 0
        start = self.clock()
        if selected_indices is None:
            if selector not in {"bm25", "recency"}:
                raise ValueError("Serving workload supports explicit indices, bm25 or recency")
            selected_indices = selectors.select_context_chunk_indices(
                selector, self.chunks, list(bare_question_ids) if bare_question_ids is not None else query.tolist(), topk)
        indices = [int(i) for i in selected_indices]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.chunks) for i in indices):
            raise ValueError("Invalid or duplicate selected indices")
        retrieval_end = self.clock()
        cpu_payloads, load_bytes = self._load_selected(indices)
        cpu_pieces = [torch.tensor([cm._sink_prefix_id()], dtype=torch.long)] + [self.chunks[i] for i in indices]
        load_end = self.clock()
        transfer_bytes = tensor_bytes(cpu_payloads) + tensor_bytes(cpu_pieces) + tensor_bytes(query)
        payloads = map_tensors(cpu_payloads, lambda x: x.to(cm.device))
        pieces = map_tensors(cpu_pieces, lambda x: x.to(cm.device))
        gpu_query = query.to(cm.device)
        transfer_end = self.clock()
        pack, merged = self._prepare_pack(payloads, pieces, gpu_query)
        prepare_end = self.clock()
        cb_stats, inventory = {}, {}
        if capture_inventory:
            inventory["staged_document_payload"] = kv_inventory([pair for p in payloads for pair in p["kv"]])
            inventory["merged_before_read"] = kv_inventory(merged)
        logits, positions, mixed = self.cb.read(pack, merged, 1, query.numel(), self.cb.recompute_ratio, cb_stats)
        if capture_inventory:
            inventory["blended_after_prefill"] = kv_inventory(mixed)
        cache = self.cb.decode_cache(mixed)
        pack_length = pack.shape[1]
        # Preserve persistent CPU payloads; online staging and stale merged KV
        # are no longer needed after the selected read has produced its cache.
        del payloads, pieces, gpu_query, pack, merged, mixed
        first_logits = logits[0, -1].float().clone()
        _, eos = cm._bos_eos(cm.tokenizer)
        if eos is not None:
            first_logits[eos] = -float("inf")
        step_logits = [first_logits.cpu().clone()] if capture_logits else []
        generated = [int(first_logits.argmax())]
        first_end = self.clock()
        decode_steps = 0
        for step in range(1, max_new_tokens):
            logits = self.cb.decode_step(generated[-1], cache, pack_length + step - 1)
            decode_steps += 1
            next_logits = logits[0, -1].float().clone()
            if force_length and eos is not None:
                next_logits[eos] = -float("inf")
            if capture_logits:
                step_logits.append(next_logits.cpu().clone())
            token = int(next_logits.argmax())
            if not force_length and eos is not None and token == eos:
                break
            generated.append(token)
        end = self.clock()
        peak = torch.cuda.max_memory_allocated(cm.device) if cm.device.type == "cuda" else 0
        if capture_inventory:
            inventory["decode_end"] = kv_inventory([(layer.keys, layer.values) for layer in cache.layers])
        stats = {"selected_indices": indices, "read_tokens": pack_length, "query_tokens": query.numel(),
            "retrieval_s": retrieval_end - start, "load_s": load_end - retrieval_end, "load_bytes": load_bytes,
            "transfer_s": transfer_end - load_end, "transfer_bytes": transfer_bytes,
            "rotate_prepare_s": prepare_end - transfer_end, "read_prefill_s": first_end - prepare_end,
            "ttft_s": first_end - start, "decode_s": end - first_end, "total_s": end - start,
            "generated_tokens": len(generated), "decode_steps": decode_steps,
            "fixed_generation_length": force_length, "peak_allocated_bytes": peak,
            "incremental_peak_bytes": max(0, peak - baseline), "capture_calls": 0,
            "document_capture_calls": 0, "query_capture_calls": 0,
            "cacheblend": cb_stats, "online_kv_inventory": inventory if capture_inventory else None,
            "instrumentation_diagnostic": bool(capture_logits or capture_inventory),
            "timing_eligible": bool(self.hardware['timing_eligible'] and not capture_logits and not capture_inventory)}
        if capture_logits:
            stats['step_logits'] = step_logits
        return generated, stats
