"""Trained reuse semantic checks. Launch ONLY inside the existing remote guard.

Input JSON: {"packs": [{"document_ids": [...], "source_ids": [...],
"chunks": [[token_ids...], ...], "prompts": [[token_ids...], [token_ids...]],
"probe_indices": [[original_question_rows...], [original_question_rows...]]}]}.
Use two real source documents, each with two questions, as two separate packs.
This is a bounded implementation check, never a speed or answer-quality report.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1] / "Encbank", HERE.parent / "encbank_v2_benchmarks_20260908"):
    sys.path.insert(0, str(path))
from infra_protocol import digest, now, read_json, save_json


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint-map", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", choices=("D0", "A", "B", "D1", "FULL"), default=["D0", "A", "B", "D1", "FULL"])
    ap.add_argument("--generation-tokens", type=int, default=8)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--rho", type=float, default=.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.)
    ap.add_argument("--seed", type=int, default=20260913)
    return ap


def main(args, admission_check=None):
    args.out.mkdir(parents=True, exist_ok=False)
    packs = read_json(args.data)["packs"]
    if len(packs) < 2 or len({i for p in packs for i in p.get("document_ids", [])}) < 2:
        raise ValueError("Use at least two real source documents, with two questions in each fixed pack")
    for pack in packs:
        if (len(pack["chunks"]) < 2 or len(pack["prompts"]) != 2
                or pack["prompts"][0] == pack["prompts"][1] or not pack.get("document_ids")):
            raise ValueError("Each real-data pack needs document IDs, multiple blocks and two different questions")
    checkpoint_map = read_json(args.checkpoint_map)
    checkpoint_map = checkpoint_map.get("checkpoints", checkpoint_map)
    if admission_check is None:
        raise ValueError("Launch via the existing remote guard and supply admission_check")
    admission_check(require_idle=True)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    from train_8b_baseline import attach_lora, restore_flat
    from encbank.model import Encbank
    from reuse_benchmark import run_reuse_quality_probe
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    admission_check(require_idle=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    admission_check(require_idle=False)
    model = model.to("cuda:0").eval()
    modules = attach_lora(model, args.j, args.rank, args.alpha, torch.float32)
    result = {"status": "running", "started_at": now(), "arms": {},
              "input_sha256": digest(args.data), "checkpoint_map_sha256": digest(args.checkpoint_map),
              "timing_eligible": False, "formal_memory_eligible": False}
    save_json(args.out / "result.json", result)
    for arm in args.arms:
        admission_check(require_idle=False)
        args.arm = arm
        checkpoint = checkpoint_map["D0" if arm == "FULL" else arm]
        args.adapter = checkpoint["path"] if isinstance(checkpoint, dict) else checkpoint
        saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
        for key in ("j", "rank", "alpha"):
            if saved.get(key) != getattr(args, key):
                raise ValueError(f"Adapter {key} mismatches quality recipe")
        training = saved.get("metadata", {}).get("recipe", {})
        if (training.get("arm") != ("D0" if arm == "FULL" else arm)
                or training.get("m") != args.m or training.get("rho") != args.rho):
            raise ValueError("Expected this arm's trained checkpoint, including matching fusion/retention")
        restore_flat(modules, saved["named"])
        del saved
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encbank = Encbank(model, resume_j=args.j, tokenizer=tokenizer)
        receipts = []
        for index, pack in enumerate(packs):
            admission_check(require_idle=False)
            print(f"{now()} quality arm={arm} pack={index}", flush=True)
            receipt = run_reuse_quality_probe(args, torch, encbank, tokenizer,
                chunks=pack["chunks"], prompts=pack["prompts"], source_ids=pack.get("source_ids"),
                probe_indices=pack["probe_indices"])
            receipt["document_ids"] = pack["document_ids"]
            save_json(args.out / f"quality_{arm}_pack{index}.json", receipt)
            receipts.append(receipt)
            gc.collect()
        combined = dict(receipts[0])
        combined.update(passed=all(r["passed"] for r in receipts),
            checks={key: all(r["checks"][key] for r in receipts) for key in receipts[0]["checks"]},
            pack_receipts=receipts, document_ids=sorted({i for p in packs for i in p["document_ids"]}),
            unique_source_documents=len({i for p in packs for i in p["document_ids"]}),
            input_sha256=digest(args.data), finished_at=now())
        combined["next_action"] = None if combined["passed"] else "requires_path_quality_evaluation"
        result["arms"][arm] = combined
        save_json(args.out / "result.json", result)
    result.update(status="complete", passed=all(r["passed"] for r in result["arms"].values()), finished_at=now())
    save_json(args.out / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
