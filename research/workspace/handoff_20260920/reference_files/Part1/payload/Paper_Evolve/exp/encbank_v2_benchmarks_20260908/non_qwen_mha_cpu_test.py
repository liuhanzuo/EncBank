"""24-layer / 32:32-head tiny MHA and common five-arm protocol checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
import torch
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from non_qwen_explicit_reader import ARMS, NonQwenExplicitReader, native_reference, digest
from non_qwen_llama_candidate import DenseLlamaCandidate, Encbank

ERRORS = []
def equal(label, x, y):
    error = float((x.float() - y.float()).abs().max())
    ERRORS.append({"name": label, "max_abs_error": error, "shape": list(x.shape)})
    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5, msg=label)

def fixture():
    context, query = list(range(8, 24)), [31, 34, 37]
    chunks = [context[i:i + 5] for i in range(0, len(context), 5)]
    selected = [2, 0]
    selected_ids = [x for i in selected for x in chunks[i]]
    pack = [1] + selected_ids + query
    f = {"task": "tiny", "id": "mha32", "index": 0, "context_ids": context, "query_ids": query,
         "context_token_count": len(context), "query_token_count": len(query), "chunk_size": 5,
         "context_chunk_lengths": list(map(len, chunks)), "ordered_selected_indices": selected,
         "selected_chunk_lengths": [len(chunks[i]) for i in selected], "bos_token_id": 1,
         "eos_token_ids": [2], "suppress_first_eos": True, "max_new_tokens": 8,
         "read_pack_tokens": len(pack), "conservative_read_plus_cap": len(pack) + 8, "budget_valid": True}
    for key, ids in (("context_ids", context), ("query_ids", query), ("selected_ids", selected_ids), ("pack_ids", pack)):
        f[key + "_sha256"] = digest(ids)
    f["fixture_sha256"] = digest(f)
    return f

def tiny(impl):
    torch.manual_seed(531)
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=24,
                      num_attention_heads=32, num_key_value_heads=32, head_dim=2,
                      max_position_embeddings=8192, bos_token_id=1, eos_token_id=2,
                      attention_dropout=0.0, tie_word_embeddings=True,
                      rope_parameters={"rope_type": "default", "rope_theta": 130000.0})
    cfg._attn_implementation = impl
    return LlamaForCausalLM(cfg).float().eval()

def capture_check(m):
    evidence = []
    for bos in (False, True):
        cm = DenseLlamaCandidate(m, 8, model_binding="tiny24-mha32-theta130000", chunk_write_sink=bos)
        raw = [9, 12, 15, 18, 21]
        doc = cm.write_document([raw])
        ids = ([1] if bos else []) + raw
        seen, hooks = {}, []
        def save(name):
            def hook(_mod, _args, output):
                seen[name] = output.detach().clone()
            return hook
        for layer in range(8):
            hooks += [m.model.layers[layer].self_attn.k_proj.register_forward_hook(save((layer, "k"))),
                      m.model.layers[layer].self_attn.v_proj.register_forward_hook(save((layer, "v")))]
        hooks.append(m.model.layers[7].register_forward_hook(save("h")))
        offset = 4096  # exercise the candidate model's larger unscaled position range
        try:
            with torch.no_grad():
                out = m(input_ids=torch.tensor([ids]), position_ids=torch.arange(offset - int(bos), offset - int(bos) + len(ids)).unsqueeze(0), use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        entry = doc["chunks"][0]
        drop = int(bos)
        equal("MHA.hj.shifted", entry["h"], seen["h"][:, drop:])
        for layer, (k, v) in entry["kv"].items():
            equal("MHA.preK", k, seen[layer, "k"].view(1, len(ids), 32, 2).transpose(1, 2)[:, :, drop:])
            equal("MHA.V", v, seen[layer, "v"].view(1, len(ids), 32, 2).transpose(1, 2)[:, :, drop:])
            pos = torch.arange(offset, offset + len(raw)).unsqueeze(0)
            equal("MHA.relocatedK.nativecache", cm.rotate_key(k, pos), out.past_key_values.layers[layer].keys[:, :, drop:])
        evidence.append({"chunk_bos": bos, "layers": 8, "kv_shape": [1, 32, 5, 2], "relocation_offset": offset})
    return evidence

def common_entry_check(m, f):
    outputs = {arm: NonQwenExplicitReader(m, arm, j=8, model_binding="tiny24-mha32-theta130000").generate_fixture(f, capture_logits=True) for arm in ARMS}
    ref = native_reference(m, f, capture_logits=True)
    assert outputs["j0"]["ids"] == ref["ids"]
    for i, (x, y) in enumerate(zip(outputs["j0"]["step_logits"], ref["step_logits"])):
        equal(f"MHA.j0.native.step{i}", x, y)
    assert {x["fixture_sha256"] for x in outputs.values()} == {f["fixture_sha256"]}
    assert all(x["ordered_selected_indices"] == f["ordered_selected_indices"] and x["suppress_first_eos"] for x in outputs.values())
    # Compare original and BOS controls with the original generic Encbank decode
    # helper, which is the actual legacy first-step-suppression convention.
    for arm in ("pub", "pub_sink", "j0"):
        entry = NonQwenExplicitReader(m, arm, j=8, model_binding="tiny24-mha32-theta130000")
        cm = entry.reader
        chunks = [f["context_ids"][i:i + 5] for i in range(0, len(f["context_ids"]), 5)]
        saved = cm.write_sink
        cm.write_sink = False
        sink = cm.write_chunk([1])
        cm.write_sink = saved
        hs = [cm.write_chunk(chunks[i]) for i in f["ordered_selected_indices"]]
        ids = cm._decode_from_pack(sink, hs, f["query_ids"], 2, 8, True)
        assert ids == outputs[arm]["ids"]
    return {arm: {k: v for k, v in row.items() if k != "step_logits"} for arm, row in outputs.items()}

def visibility_and_incremental(m, f):
    evidence = []
    for arm in ("fix_all", "fix_none"):
        cm = NonQwenExplicitReader(m, arm, j=8, model_binding="tiny24-mha32-theta130000").reader
        chunks = [f["context_ids"][i:i + 5] for i in range(0, len(f["context_ids"]), 5)]
        selected = [chunks[i] for i in f["ordered_selected_indices"]]
        doc = cm.write_document(selected)
        sink, hs = cm.prepare_read(doc, [0, 1])
        memory = 1 + sum(map(len, selected))
        qh, bottom, qpos = cm.write_prefill(f["query_ids"])
        logits, top, ppos = cm.read_prefill(sink, hs, qh)
        allow = cm.lower_visibility(0, 3, memory + 3)[0, 0]
        assert allow[:, 0].all() and bool(allow[:, 1:memory].all()) == (arm == "fix_all")
        assert torch.equal(allow[:, memory:], torch.ones(3, 3, dtype=torch.bool).tril())
        prefix = []
        for i in range(3):
            token = int(logits[0, -1].argmax())
            prefix.append(token)
            logits = cm.decode_step(token, bottom, top, qpos, ppos)
            qpos += 1
            ppos += 1
            fresh = DenseLlamaCandidate(m, 8, model_binding="tiny24-mha32-theta130000", lower_layers=[] if arm == "fix_none" else None)
            rs, rh = fresh.prepare_read(doc, [0, 1])
            qq, _, _ = fresh.write_prefill(f["query_ids"] + prefix)
            expected, _, _ = fresh.read_prefill(rs, rh, qq)
            fresh.clear_read()
            equal(f"MHA.{arm}.incremental{i}", logits, expected)
        assert [bottom.get_seq_length(i) for i in range(8)] == [ppos] * 8
        assert [top.get_seq_length(i) for i in range(8, 24)] == [ppos] * 16
        cm.clear_read()
        evidence.append({"arm": arm, "j": 8, "lower_layers": 8, "upper_layers": 16, "final_cache_tokens": ppos, "incremental_steps": 3})
    return evidence

def eos_difference(m, f):
    def force_eos(_module, _args, logits):
        replacement = torch.zeros_like(logits)
        replacement[..., 2] = 10
        return replacement
    handle = m.lm_head.register_forward_hook(force_eos)
    try:
        output = {arm: NonQwenExplicitReader(m, arm, j=8, model_binding="tiny24-mha32-theta130000").generate_fixture(f) for arm in ARMS}
        assert all(row["ids"] == [0, 2] and row["first_eos_suppressed"] and row["decoder_forwards"] == 1 for row in output.values())
        assert native_reference(m, f)["ids"] == [0, 2]
        native_cm = DenseLlamaCandidate(m, 8, model_binding="tiny24-mha32-theta130000")
        doc = native_cm.write_document([[8, 9, 10]])
        original_candidate = native_cm.query(doc, [0], f["query_ids"], max_new_tokens=8)
        assert original_candidate["ids"] == [2]
        return {"all_five_arms_and_policy_matched_native": [0, 2], "previous_native_EOS_candidate": [2],
                "difference_observed_and_resolved_in_new_entry": True}
    finally:
        handle.remove()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"status": "running", "dtype": "float32", "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
              "torch_threads": 2, "interop_threads": 16, "real_weights_loaded": False,
              "geometry": {"layers": 24, "j": 8, "attention_heads": 32, "kv_heads": 32, "head_dim": 2,
                           "hidden": 64, "rope_theta": 130000, "rope_type": "default", "max_positions": 8192},
              "tiny_vs_real_difference": "same layer/head counts/theta; reduced width/head_dim/vocabulary; FP32 random weights", "checks": []}
    try:
        f = fixture()
        for implementation in ("sdpa", "eager"):
            m = tiny(implementation)
            for name, fn in (("native_capture_and_relocation", lambda: capture_check(m)),
                             ("common_five_arm_entry", lambda: common_entry_check(m, f)),
                             ("visibility_cache_decode", lambda: visibility_and_incremental(m, f)),
                             ("EOS_protocol_difference", lambda: eos_difference(m, f))):
                evidence = fn()
                result["checks"].append({"attention": implementation, "name": name, "passed": True, "evidence": evidence})
                print(f"PASS {implementation} {name}", flush=True)
        assert "s15_ruler_lower" not in sys.modules and "reader_adapter" not in sys.modules
        assert not torch.cuda.is_initialized()
        result.update(status="complete", complete=True, qwen_shared_modules_not_imported=True, cuda_initialized=False)
    except Exception:
        result.update(status="failed", complete=False, error=traceback.format_exc())
        print(result["error"], flush=True)
    result.update(comparisons=ERRORS, comparison_count=len(ERRORS), max_abs_error=max((x["max_abs_error"] for x in ERRORS), default=0))
    args.output.write_text(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
