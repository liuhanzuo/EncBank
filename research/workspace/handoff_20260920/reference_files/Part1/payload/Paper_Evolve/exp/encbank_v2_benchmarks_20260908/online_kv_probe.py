"""Direct active-KV inventories around existing reader calls; no timed claims.

No generation loop or reader source is changed. Only scalar tensor metadata is
retained; duplicate backing storages are counted once at each named boundary.
"""
from contextlib import contextmanager, ExitStack


@contextmanager
def patched(obj, name, method):
    owned, old = name in obj.__dict__, obj.__dict__.get(name)
    setattr(obj, name, method)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def cache_entries(cache, band):
    if cache is None:
        return []
    out = []
    for index, layer in enumerate(cache.layers):
        key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
        if key is not None and value is not None and key.numel() and value.numel():
            out.append((band, index, key, value))
    return out


def inventory(entries):
    """Inspect actual tensors; export only metadata, never pointers or values."""
    layers, storages = [], {}
    for band, layer, key, value in entries:
        pair = []
        for kind, tensor in (("key", key), ("value", value)):
            storage = tensor.untyped_storage()
            identity = str(tensor.device), storage.data_ptr()
            if identity in storages and storages[identity] != storage.nbytes():
                raise ValueError("Conflicting storage sizes at a single boundary")
            storages[identity] = storage.nbytes()
            pair.append({"kind": kind, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "logical_bytes": tensor.numel()*tensor.element_size(),
                "backing_storage_bytes": storage.nbytes(), "storage_offset_elements": tensor.storage_offset(),
                "stride": list(tensor.stride())})
        layers.append({"band": band, "layer_index": layer, "sequence_length": int(key.shape[-2]),
                       "tensors": pair, "logical_bytes": sum(p["logical_bytes"] for p in pair)})
    return {"tensor_count": 2*len(layers), "populated_layer_entries": len(layers),
        "logical_tensor_bytes": sum(layer["logical_bytes"] for layer in layers),
        "unique_backing_storage_bytes": sum(storages.values()), "unique_backing_storages": len(storages),
        "layers": layers,
        "scope": "Active autoregressive K/V tensors at this boundary only; excludes weights, hidden states, attention intermediates, persistent CPU/disk cache, and temporary staged copies"}


def measure_query(reader, query_ids, *, selected_indices, max_new_tokens):
    """Return existing fixed-G output plus actual prefill/end-decode inventories."""
    cm = reader.cm
    points, lower = {}, {}
    steps = 0
    is_cb = hasattr(reader, "cb")
    with ExitStack() as stack:
        if is_cb:
            original_cache = reader.cb.decode_cache
            def make_cache(mixed):
                result = original_cache(mixed)
                points["prefill_complete"] = inventory(cache_entries(result, "full"))
                return result
            stack.enter_context(patched(reader.cb, "decode_cache", make_cache))
            original_step = reader.cb.decode_step
            def decode_step(token, cache, position):
                nonlocal steps
                result = original_step(token, cache, position)
                steps += 1
                if steps == max_new_tokens-1:
                    points["decode_complete"] = inventory(cache_entries(cache, "full"))
                return result
            stack.enter_context(patched(reader.cb, "decode_step", decode_step))
        else:
            original_lower = cm.write_prefill
            def write_prefill(ids):
                result = original_lower(ids)
                lower["cache"] = result[1]
                return result
            stack.enter_context(patched(cm, "write_prefill", write_prefill))
            original_upper = cm.read_prefill
            def read_prefill(*args, **kwargs):
                result = original_upper(*args, **kwargs)
                points["prefill_complete"] = inventory(cache_entries(lower["cache"], "lower") + cache_entries(result[1], "upper"))
                return result
            stack.enter_context(patched(cm, "read_prefill", read_prefill))
            original_step = cm.decode_step
            def decode_step(token, bottom_cache, top_cache, q_position, pack_position):
                nonlocal steps
                result = original_step(token, bottom_cache, top_cache, q_position, pack_position)
                steps += 1
                if steps == max_new_tokens-1:
                    points["decode_complete"] = inventory(cache_entries(bottom_cache, "lower") + cache_entries(top_cache, "upper"))
                return result
            stack.enter_context(patched(cm, "decode_step", decode_step))
        ids, stats = reader.query_ids(query_ids, selected_indices=selected_indices,
            max_new_tokens=max_new_tokens, force_length=True)
    lower.clear()
    if len(ids) != max_new_tokens or steps != max_new_tokens-1 or set(points) != {"prefill_complete", "decode_complete"}:
        raise ValueError("Expected complete fixed-G diagnostic with both cache boundaries")
    if getattr(cm, "_bottom", None) is not None:
        raise ValueError("Reader did not clear its query cache state")
    # Deliberately do not export the instrumented timing/peak fields as results.
    record = {"generated_ids": ids, "generated_tokens": len(ids), "actual_decode_forward_calls": steps,
        "selected_indices": stats["selected_indices"], "read_tokens": stats["read_tokens"],
        "query_tokens": stats["query_tokens"], "inventory": points, "timing_eligible": False,
        "measurement_kind": "direct live KV tensor inventory", "fixed_generation_length": True,
        "document_or_query_capture_calls": stats["capture_calls"], "state_reset_verified": True}
    validate_inventory_record(record, reader.arm, cm.resume_j, cm.num_layers)
    return record


def validate_inventory_record(record, arm, split, layers):
    for phase, extra in (("prefill_complete", 0), ("decode_complete", record["generated_tokens"]-1)):
        measured = record["inventory"][phase]
        entries = measured["layers"]
        if len(entries) != layers or len({row["layer_index"] for row in entries}) != layers:
            raise ValueError("Every model layer needs exactly one active KV pair")
        for row in entries:
            if arm in {"pub", "pub_sink", "pub_lora"} and row["layer_index"] < split:
                length = record["query_tokens"] + extra
            else:
                length = record["read_tokens"] + extra
            if row["sequence_length"] != length or any(t["shape"][-2] != length for t in row["tensors"]):
                raise ValueError("Actual KV sequence length differs from the named reader boundary")
        if measured["logical_tensor_bytes"] != sum(row["logical_bytes"] for row in entries):
            raise ValueError("Inventory byte arithmetic differs")
    if record["document_or_query_capture_calls"] != 0 or record["timing_eligible"]:
        raise ValueError("Inventory must reuse the document and remain timing-ineligible")
