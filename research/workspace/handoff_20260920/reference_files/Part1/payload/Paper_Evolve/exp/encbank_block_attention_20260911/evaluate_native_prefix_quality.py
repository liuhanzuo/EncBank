"""Remote eval-only native Encbank cold versus whole-prefix reuse QA.

Uses only naturally identical prepared document packs. No retrieval, prompt,
answer, model window or generation budget is changed to manufacture a hit.
An external controller must hold the existing public single-3090 GPU lease.
This runner produces quality observations, never formal speed or GPU memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

from evaluate_backend_quality import digest, _canonical_hash, _write_json, _safe_failure

HERE = Path(__file__).resolve().parent


def source_paths():
    names = ("evaluate_native_prefix_quality.py", "native_prefix_quality.py",
             "prefix_quality_protocol.py", "native_prefix_reader.py", "native_infra_readers.py",
             "evaluate_backend_quality.py", "backend_quality_results.py", "train_sparse.py",
             "prepare_data.py", "sparse_reader.py", "remote_gpu_guard.py", "remote_sparse_queue.py")
    paths = {name: HERE / name for name in names}
    paths.update({
        "evaluate_sft.py": HERE.parent / "beacon_encbank_20260909" / "evaluate_sft.py",
        "train_8b_baseline.py": HERE.parent / "encbank_v2_benchmarks_20260908" / "train_8b_baseline.py",
        "Encbank/encbank/model.py": HERE.parents[1] / "Encbank" / "encbank" / "model.py"})
    return paths


def source_hashes():
    return {name: digest(path) for name, path in source_paths().items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "init-adapter", "train", "dev", "out", "lease"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def run(args, lease_receipt):
    # main() validates the external lease before any Torch import.
    import torch
    import transformers
    from train_sparse import validate_prepared
    from prepare_data import read_rows
    from train_8b_baseline import attach_lora, restore_flat
    from evaluate_sft import eos_token_ids
    from encbank.model import Encbank
    from native_infra_readers import NativeEncbankReader
    from native_prefix_reader import NativePrefixEncbankReader
    from native_prefix_quality import run_native_prefix_group
    from prefix_quality_protocol import build_prefix_quality_plan, summarize_prefix_quality
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.integrations.sdpa_attention as sdpa
    from remote_gpu_guard import validate_worker_lease
    from remote_sparse_queue import ROOT

    out = args.out.resolve()
    if ROOT not in out.parents:
        raise ValueError("Quality outputs must stay under the remote experiment root")
    out.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    records = []
    plan = stops = None
    active_group = None
    model = cm = native = prefix = sink = memories = None
    metadata = dict(run_id=run_id, status="starting", lease=lease_receipt,
                    formal_inference_timing=False, formal_inference_memory=False)
    _write_json(out / "status.json", metadata)
    try:
        train, dev = read_rows(args.train), read_rows(args.dev)
        validate_prepared(train, dev)
        plan = build_prefix_quality_plan(dev, mode=args.mode, seed=args.seed)
        _write_json(out / "plan.json", plan)
        source = source_hashes()
        recipe = dict(model=str(args.model.resolve()), init_adapter_sha256=digest(args.init_adapter),
                      train_sha256=digest(args.train), dev_sha256=digest(args.dev),
                      mode=args.mode, seed=42, max_new_tokens=128, j=12, rank=32, alpha=32.,
                      plan_sha256=plan["plan_sha256"],
                      branches=["native_cold", "native_prefix"],
                      decoding="independent-free-greedy-natural-eos", teacher_forced_ce=False,
                      reuse_scope="same-original-ordered-pack-within-group")
        recipe_sha = _canonical_hash(recipe)
        metadata.update(status="loading", recipe=recipe, recipe_sha256=recipe_sha, source_sha256=source)
        _write_json(out / "metadata.json", metadata)
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 16:
            torch.set_num_interop_threads(16)
        admission = validate_worker_lease(args.lease, require_idle=True)
        metadata["admissions"] = {"before_torch_import": lease_receipt,
                                  "after_import_before_cuda": admission}
        _write_json(out / "metadata.json", metadata)
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        if "3090" not in torch.cuda.get_device_name(device) or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This QA recipe requires the leased single RTX3090 with BF16")
        # Same native writer/head dispatch policy as the existing quality pilot.
        sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
        validate_worker_lease(args.lease, require_idle=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval()
        validate_worker_lease(args.lease, require_idle=False)
        if (model.config.model_type != "qwen3" or model.config.num_hidden_layers != 36
                or model.config.num_key_value_heads != 8 or model.config.attention_dropout != 0.):
            raise ValueError("Expected dense Qwen3-8B,36 layers,8 KV heads,zero dropout")
        longest = max(sum(map(len, row["document_chunks"])) + len(row["prompt_ids"]) + 129
                      for row in plan["ordered_rows"])
        if longest > model.config.max_position_embeddings:
            raise ValueError("Prepared input and full generation allowance exceed model window")
        modules = attach_lora(model, 12, 32, 32., torch.float32)
        saved = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        if any(saved.get(k) != v for k, v in {"j": 12, "rank": 32, "alpha": 32.}.items()):
            raise ValueError("Initial strong adapter recipe differs from the fixed plan")
        restore_flat(modules, saved["named"])
        del saved
        if any(m.A.dtype != torch.float32 or m.B.dtype != torch.float32 for m in modules.values()):
            raise RuntimeError("Unmerged LoRA master weights must remain FP32")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        cm = Encbank(model, resume_j=12)
        native, prefix = NativeEncbankReader(cm).eval(), NativePrefixEncbankReader(cm).eval()
        stops = eos_token_ids(tokenizer, model)
        sink_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if sink_id is None or not stops:
            raise ValueError("Sink or stopping token is unavailable")
        metadata.update(status="running", host=platform.node(), gpu=torch.cuda.get_device_name(device),
                        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"], torch=str(torch.__version__),
                        transformers=transformers.__version__, cuda_runtime=torch.version.cuda,
                        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                        backbone_dtype="torch.bfloat16", lora_master_dtype="torch.float32",
                        autocast_dtype="torch.bfloat16", stop_token_ids=sorted(stops),
                        metric_scope="Qasper-train document-disjoint pilot, not official test or128k",
                        dispatch_scope="Native SDPA on remote3090; not proof of5090 dispatch/performance")
        _write_json(out / "metadata.json", metadata)

        def check_guard():
            validate_worker_lease(args.lease, require_idle=False)

        with (out / "records.jsonl").open("x", encoding="utf-8", buffering=1) as stream, \
                torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sink = cm.write_chunk([int(sink_id)]).detach()

            def on_record(record):
                location = next(row for row in plan["ordered_rows"] if row["id"] == record["id"])
                fields = ("ordinal", "group_index", "query_index_in_group", "pack_fingerprint", "expected_prefix_hit")
                if any(key in record and record[key] != location[key] for key in fields):
                    raise RuntimeError("Generation record differs from the planned question/pack")
                enriched = _safe_failure({**record, **{key: location[key] for key in fields}, "run_id": run_id,
                    "recipe_sha256": recipe_sha, "source_sha256": source,
                    "formal_inference_timing": False, "formal_inference_memory": False})
                stream.write(json.dumps(enriched, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(enriched)
                _write_json(out / "status.json", dict(status="running", run_id=run_id,
                    records_completed=sum(r.get("status") == "complete" for r in records),
                    target_records=2 * len(plan["ordered_ids"]), active_group=active_group,
                    updated_utc=datetime.now(timezone.utc).isoformat(), formal_inference_timing=False))

            for group in plan["groups"]:
                active_group = group["pack_fingerprint"]
                rows = [plan["ordered_rows"][i] for i in group["row_ordinals"]]
                check_guard()
                # Discard prior group before writing another pack. Helpers also
                # invalidate at group entry; no old h_j or prefix is recycled.
                prefix.invalidate()
                memories = [cm.write_chunk(chunk).detach() for chunk in rows[0]["document_chunks"]]
                run_native_prefix_group(native, prefix, sink, memories, rows, tokenizer,
                    stop_ids=stops, max_new_tokens=128, on_record=on_record, check_guard=check_guard)
                prefix.invalidate()
                memories = None
                active_group = None
            summary = summarize_prefix_quality(plan, records, stop_token_ids=sorted(stops), max_new_tokens=128)
            _write_json(out / "summary.json", summary)
            if summary["status"] != "complete":
                raise RuntimeError("Native prefix paired completeness checks failed")
        _write_json(out / "status.json", dict(status="complete", run_id=run_id,
            records_completed=len(records), target_records=2 * len(plan["ordered_ids"]),
            updated_utc=datetime.now(timezone.utc).isoformat(),
            formal_inference_timing=False, formal_inference_memory=False))
        return summary
    except Exception as exc:
        failure = _safe_failure(dict(status="failed", run_id=run_id, active_group=active_group,
            records_written=len(records), error=dict(type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc()), updated_utc=datetime.now(timezone.utc).isoformat(),
            formal_inference_timing=False, formal_inference_memory=False))
        if plan is not None and stops:
            try:
                partial = summarize_prefix_quality(plan, records, stop_token_ids=sorted(stops), max_new_tokens=128)
                partial["outer_execution_status"] = "failed"
                _write_json(out / "partial_summary.json", partial)
            except Exception as partial_error:
                failure["partial_summary_error"] = str(partial_error)
        _write_json(out / "failure.json", failure)
        _write_json(out / "status.json", failure)
        raise
    finally:
        memories = sink = prefix = native = cm = model = None
        gc.collect()


def main():
    args = parse_args()
    from evaluate_backend_quality import _validate_platform
    _validate_platform(args)
    from remote_gpu_guard import validate_worker_lease
    lease_receipt = validate_worker_lease(args.lease, require_idle=True)
    run(args, lease_receipt)


if __name__ == "__main__":
    main()
