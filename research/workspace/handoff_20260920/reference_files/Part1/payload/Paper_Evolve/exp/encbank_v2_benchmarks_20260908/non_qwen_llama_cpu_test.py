"""Remote CPU implementation evidence, never a pretrained benchmark score."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("run with CUDA_VISIBLE_DEVICES='' explicitly; CPU only")
if any(os.environ.get(k) != "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
    raise RuntimeError("OMP and MKL must both be 2")
if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
    raise RuntimeError("tokenizer parallelism must be false")

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from non_qwen_llama_candidate import DenseLlamaCandidate, WORKSPACE

CHUNKS = [[8, 11, 14, 17, 20], [9, 12, 15], [23, 26, 29, 32, 35, 38, 41]]
SELECTED, QUERY = [2, 0], [47, 50, 53, 56]
ATOL, RTOL = 2e-6, 2e-5
MEASUREMENTS = []


def equal(label, actual, expected):
    error = float((actual.float() - expected.float()).abs().max()) if actual.numel() else 0.0
    MEASUREMENTS.append({"name": label, "shape": list(actual.shape), "max_abs_error": error})
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=label)


def model(implementation="sdpa", *, eos=2, seed=1729):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=112,
                      num_hidden_layers=4, num_attention_heads=8,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                      bos_token_id=1, eos_token_id=eos, pad_token_id=0,
                      attention_dropout=0.0)
    cfg._attn_implementation = implementation
    result = LlamaForCausalLM(cfg).float().eval()
    assert next(result.parameters()).device.type == "cpu"
    return result


def candidate(m, j=2, **kwargs):
    return DenseLlamaCandidate(m, j, model_binding="tiny-random-seed1729-gqa8x2-h64-l4", **kwargs)


def native_capture(m, ids, j, offset=0):
    """Independent full stock decoder forward, with native projection/layer hooks."""
    seen, hooks = {}, []
    def save(name):
        def hook(_module, _args, value):
            seen[name] = value.detach().clone()
        return hook
    for i in range(j):
        a = m.model.layers[i].self_attn
        hooks.append(a.k_proj.register_forward_hook(save((i, "k"))))
        hooks.append(a.v_proj.register_forward_hook(save((i, "v"))))
    hooks.append((m.model.embed_tokens if j == 0 else m.model.layers[j - 1]).register_forward_hook(save("hj")))
    try:
        with torch.no_grad():
            output = m(input_ids=torch.tensor([ids]), use_cache=True,
                       position_ids=torch.arange(offset, offset + len(ids)).unsqueeze(0))
    finally:
        for hook in hooks:
            hook.remove()
    return seen, output.past_key_values


def native_greedy(m, ids, n):
    generated, logits = [], []
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids]), use_cache=True)
        cache = out.past_key_values
        for i in range(n):
            logits.append(out.logits[:, -1:].clone())
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            if i + 1 < n:
                out = m(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
    return generated, logits


def test_j0_native(m):
    cm = candidate(m, 0)
    doc = cm.write_document(CHUNKS)
    ids = [1] + [x for i in SELECTED for x in CHUNKS[i]] + QUERY
    with torch.no_grad():
        equal("j0.all_pack_logits", cm.resume_forward_ids(ids), m(input_ids=torch.tensor([ids])).logits)
    got = cm.query(doc, SELECTED, QUERY, max_new_tokens=8, eos_ids=[])
    expected_ids, expected_logits = native_greedy(m, ids, 8)
    assert got["ids"] == expected_ids
    assert got["decoder_forwards"] == 7
    for i, (a, b) in enumerate(zip(got["step_logits"], expected_logits)):
        equal(f"j0.greedy_step{i}", a, b)
    assert got["upper_cache_lengths"] == [len(ids) + 7] * 4
    return {"pack_tokens": len(ids), "generated_ids": got["ids"], "decoder_forwards": 7}


def test_capture_and_relocation(m):
    records = []
    for j in (2, 4):
        for bos in (False, True):
            cm = candidate(m, j, chunk_write_sink=bos)
            doc = cm.write_document(CHUNKS)
            for ci, raw in enumerate(CHUNKS):
                ids = ([1] if bos else []) + raw
                seen, cache = native_capture(m, ids, j)
                drop = int(bos)
                entry = doc["chunks"][ci]
                equal(f"j{j}.bos{bos}.chunk{ci}.hj", entry["h"], seen["hj"][:, drop:])
                for layer, (k, v) in entry["kv"].items():
                    shape = (1, len(ids), 2, 8)
                    nk = seen[layer, "k"].reshape(shape).transpose(1, 2)[:, :, drop:]
                    nv = seen[layer, "v"].reshape(shape).transpose(1, 2)[:, :, drop:]
                    equal(f"j{j}.bos{bos}.chunk{ci}.layer{layer}.preK", k, nk)
                    equal(f"j{j}.bos{bos}.chunk{ci}.layer{layer}.V", v, nv)
                    localpos = torch.arange(drop, len(ids)).unsqueeze(0)
                    equal("native.local_rotated_K", cm.rotate_key(k, localpos), cache.layers[layer].keys[:, :, drop:])
                    # Native decoder is shifted as a whole; the local write prefix
                    # shifts too. Native default RoPE preserves relative attention.
                    offset = 31 + ci * 17
                    shifted, nativecache = native_capture(m, ids, j, offset=offset - drop)
                    equal("native.shifted_preK", k, shifted[layer, "k"].reshape(shape).transpose(1, 2)[:, :, drop:])
                    finalpos = torch.arange(offset, offset + len(raw)).unsqueeze(0)
                    equal("native.relocated_K", cm.rotate_key(k, finalpos), nativecache.layers[layer].keys[:, :, drop:])
                    assert k.untyped_storage().nbytes() == k.numel() * k.element_size()
                records.append({"j": j, "chunk_local_bos": bos, "chunk": ci,
                                "tokens": len(raw), "pre_rope_k_shape": [1, 2, len(raw), 8]})
    return records


def test_mask_cache_decode(m):
    evidence = []
    for j in (2, 4):
        for visibility in ("all", "none"):
            cm = candidate(m, j, lower_layers=None if visibility == "all" else [])
            doc = cm.write_document(CHUNKS)
            sink, chunks = cm.prepare_read(doc, SELECTED)
            memory = 1 + sum(len(CHUNKS[i]) for i in SELECTED)
            calls, handles = [], []
            for layer in range(j):
                def makehook(l):
                    def hook(_module, _args, kwargs):
                        cache = kwargs["past_key_values"]
                        calls.append({"layer": l, "mask": kwargs["attention_mask"].detach().clone(),
                                      "positions": kwargs["position_ids"].detach().clone(),
                                      "before": cache.get_seq_length(l)})
                    return hook
                handles.append(cm.layers[layer].register_forward_pre_hook(makehook(layer), with_kwargs=True))
            try:
                qh, lower, qpos = cm.write_prefill(QUERY)
                logits, upper, ppos = cm.read_prefill(sink, chunks, qh)
                assert qpos == ppos == memory + len(QUERY)
                prefix = []
                for step in range(5):
                    token = int(logits[0, -1].argmax())
                    prefix.append(token)
                    logits = cm.decode_step(token, lower, upper, qpos, ppos)
                    qpos += 1
                    ppos += 1
                    # Fresh full query+generated-prefix computation is an oracle
                    # for incremental cache updates, independent of decode_step.
                    ref = candidate(m, j, lower_layers=None if visibility == "all" else [])
                    rs, rc = ref.prepare_read(doc, SELECTED)
                    rh, _, _ = ref.write_prefill(QUERY + prefix)
                    expected, _, _ = ref.read_prefill(rs, rc, rh)
                    ref.clear_read()
                    equal(f"j{j}.{visibility}.incremental_step{step}", logits, expected)
                assert [lower.get_seq_length(i) for i in range(j)] == [ppos] * j
                assert [upper.get_seq_length(i) for i in range(j, 4)] == [ppos] * (4 - j)
            finally:
                for h in handles:
                    h.remove()
                cm.clear_read()
            # Hooks on shared model also saw the replay reference calls. Inspect
            # each observed call independently from its actual positions/cache.
            for call in calls:
                pos = call["positions"].flatten().tolist()
                assert pos == list(range(call["before"], call["before"] + len(pos)))
                n = call["before"] + len(pos)
                expected = torch.tensor([[k == 0 or (1 <= k < memory and visibility == "all")
                                          or (memory <= k <= p) for k in range(n)] for p in pos])
                actual = call["mask"][0, 0]
                if actual.dtype != torch.bool:
                    actual = actual == 0
                assert torch.equal(actual, expected), f"{visibility} observed native mask differs"
            evidence.append({"j": j, "visibility": visibility, "memory_tokens": memory,
                             "initial_query_positions": list(range(memory, memory + len(QUERY))),
                             "final_cache_length": ppos, "observed_native_layer_calls": len(calls),
                             "incremental_vs_full_query_steps": 5})
    return evidence


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def test_query_isolation_reload(m):
    cm = candidate(m)
    doc = cm.write_document(CHUNKS)
    old_tensors = [(x, x.clone(), x._version) for x in tensors(doc)]
    captures = cm.capture_calls
    query_b, selected_b = [62, 65], [1, 2]
    with tempfile.TemporaryDirectory(prefix="non_qwen_llama_") as directory:
        path = Path(directory) / "document.pt"
        torch.save(doc, path)
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        # No query may recapture chunks, including CPU/file staging.
        with patch.object(cm, "capture_lower", side_effect=AssertionError("unexpected recapture")):
            a = cm.query(doc, SELECTED, QUERY, max_new_tokens=6, eos_ids=[])
            b = cm.query(doc, selected_b, query_b, max_new_tokens=6, eos_ids=[])
            a2 = cm.query(doc, SELECTED, QUERY, max_new_tokens=6, eos_ids=[])
            disk_a = cm.query(reloaded, SELECTED, QUERY, max_new_tokens=6, eos_ids=[])
        fresh = candidate(m)
        fresh_b = fresh.query(copy.deepcopy(doc), selected_b, query_b, max_new_tokens=6, eos_ids=[])
        for name, first, second in (("AABA", a, a2), ("disk_reload", a, disk_a), ("Bfresh", b, fresh_b)):
            assert first["ids"] == second["ids"]
            for i, (x, y) in enumerate(zip(first["step_logits"], second["step_logits"])):
                equal(f"isolation.{name}.step{i}", x, y)
        assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
        for actual, old, version in old_tensors:
            assert torch.equal(actual, old) and actual._version == version
    assert cm.capture_calls == captures == len(CHUNKS) + 1
    assert cm._bottom is None
    return {"captures_at_write": captures, "query_recaptures": 0, "A_ids": a["ids"], "B_ids": b["ids"],
            "A_pack_tokens": a["pack_tokens"], "B_pack_tokens": b["pack_tokens"],
            "persistent_tensors_equal": True, "persistent_versions_equal": True,
            "disk_file_unchanged": True, "A_B_A_and_fresh_B_and_disk_A": True}


def test_native_eos_and_rejections():
    m = model(eos=0)
    with torch.no_grad():
        m.lm_head.weight.zero_()
    cm = candidate(m)
    doc = cm.write_document(CHUNKS)
    out = cm.query(doc, [0], QUERY, max_new_tokens=8)
    assert out["ids"] == [0] and out["stopped_on_eos"] and out["decoder_forwards"] == 0
    pack = torch.tensor([[1] + CHUNKS[0] + QUERY])
    with torch.no_grad():
        native = m.generate(pack, attention_mask=torch.ones_like(pack), do_sample=False, max_new_tokens=8)
    assert native[0, pack.shape[1]:].tolist() == [0]
    wrong = copy.deepcopy(doc)
    wrong["signature"]["model_binding"] = "different-weights"
    rejected = []
    for name, fn in (
        ("binding", lambda: cm.query(wrong, [0], QUERY, max_new_tokens=2)),
        ("duplicate_selection", lambda: cm.query(doc, [0, 0], QUERY, max_new_tokens=2)),
        ("missing_chunk", lambda: cm.query(doc, [9], QUERY, max_new_tokens=2)),
        ("empty_query", lambda: cm.query(doc, [0], [], max_new_tokens=2)),
        ("batch_gt_one", lambda: cm.capture_lower([[1, 2], [3, 4]])),
        ("native_context_limit", lambda: cm.capture_lower([3] * 513)),
        ("invalid_lower_layer", lambda: candidate(m, 2, lower_layers=[2])),
    ):
        try:
            fn()
        except (ValueError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError(f"{name} was not rejected")
        assert cm._bottom is None
    cfg = copy.deepcopy(m.config.rope_parameters)
    m.config.rope_parameters = {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    try:
        candidate(m)
    except ValueError:
        rejected.append("unvalidated_scaled_rope")
    else:
        raise AssertionError("scaled RoPE admitted")
    finally:
        m.config.rope_parameters = cfg
    return {"native_bos_id": 1, "native_eos_id": 0, "first_token_eos_stops": True,
            "native_generate_equal": True, "rejected": rejected}


def qwen_legacy():
    # Existing tests in a separate process: importing s15 changes that process's
    # global SDPA policy, so it must not contaminate the isolated Llama suite.
    import unittest
    import test_adapter_cpu
    torch.set_num_threads(2)  # the old module sets 1 at import; restore required 2
    names = ["test_all_arms_match_direct_reader_and_cache_survives_reopen",
             "test_constructor_and_generation_variants_are_rejected"]
    suite = unittest.TestSuite(test_adapter_cpu.AdapterTests(n) for n in names)
    # Remote legacy test predates the local exclusion of CacheBlend, whose
    # factory deliberately requires the different explicit-pack QA interface.
    # Keep shared files untouched and retain the failed unfiltered attempt.
    core_arms = tuple(a for a in test_adapter_cpu.run_driver.ARMS if a != "cacheblend16")
    with patch.object(test_adapter_cpu.run_driver, "ARMS", core_arms):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"complete": result.wasSuccessful(), "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors), "tests": names,
            "arms": list(core_arms), "excluded": {"cacheblend16": "separate explicit-pack interface"},
            "shared_test_files_modified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-legacy", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    output = {"started_utc": datetime.now(timezone.utc).isoformat(), "purpose": "CPU implementation check only",
              "pretrained_benchmark": False, "model_downloads": 0, "gpu_model_operations": 0,
              "hostname": os.uname().nodename, "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "device": "cpu", "dtype": "float32",
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "omp_threads": os.environ["OMP_NUM_THREADS"], "mkl_threads": os.environ["MKL_NUM_THREADS"],
              "atol": ATOL, "rtol": RTOL, "checks": []}
    try:
        if args.qwen_legacy:
            output.update(qwen_legacy())
            if not output["complete"]:
                raise AssertionError("existing Qwen checks failed")
        else:
            assert "s15_ruler_lower" not in sys.modules and "reader_adapter" not in sys.modules
            for implementation in ("sdpa", "eager"):
                m = model(implementation)
                for name, fn in (("j0_native", test_j0_native), ("capture_and_relocation", test_capture_and_relocation),
                                 ("mask_cache_decode", test_mask_cache_decode), ("query_isolation_reload", test_query_isolation_reload)):
                    result = fn(m)
                    output["checks"].append({"attention": implementation, "name": name, "passed": True, "evidence": result})
                    print(f"PASS {implementation} {name}", flush=True)
            output["checks"].append({"name": "native_eos_and_rejections", "passed": True,
                                     "evidence": test_native_eos_and_rejections()})
            assert "s15_ruler_lower" not in sys.modules and "reader_adapter" not in sys.modules
            output["shared_qwen_modules_not_imported"] = True
        output["complete"] = True
    except Exception:
        output["complete"] = False
        output["error"] = traceback.format_exc()
        print(output["error"], flush=True)
    finally:
        output["elapsed_seconds"] = time.monotonic() - start
        output["measurements"] = MEASUREMENTS
        output["comparison_count"] = len(MEASUREMENTS)
        output["max_abs_error"] = max((m["max_abs_error"] for m in MEASUREMENTS), default=0)
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({k: output[k] for k in ("complete", "elapsed_seconds", "comparison_count", "max_abs_error")}), flush=True)
    return 0 if output["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
