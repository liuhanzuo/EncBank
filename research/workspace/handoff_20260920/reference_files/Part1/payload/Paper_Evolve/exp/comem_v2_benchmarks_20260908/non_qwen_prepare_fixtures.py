"""SmolLM-native Qasper200 and RULER150 fixture/budget preparation; CPU only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch
from transformers import AutoTokenizer
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "COMem"), str(ROOT / "exp")]
from comem import selectors
from eval import ruler as R
from non_qwen_download import ALLOWED_ROOT, REVISION, REPO

BOUNDARY = "\n<COMEM_NON_QWEN_CONTEXT_QUERY_BOUNDARY_20260909>\n"
ARMS = ["pub", "pub_sink", "fix_all", "j0", "fix_none"]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def file_info(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def prepare_row(tok, task, source_id, index, context, query, question, answers, cap, selector, prompt, **extra):
    context_ids = tok.encode(context, add_special_tokens=False)
    query_ids = tok.encode(query, add_special_tokens=False)
    assert query_ids and tok.bos_token_id == 1 and tok.eos_token_id == 2
    chunks = [context_ids[i:i + 512] for i in range(0, len(context_ids), 512)]
    bare = tok.encode(question, add_special_tokens=False)
    selected = list(selectors.select_context_chunk_indices(selector, [torch.tensor(x) for x in chunks],
                       bare, 12, iter_hop_topk=4, iter_rounds=0))
    assert len(set(selected)) == len(selected) and all(0 <= i < len(chunks) for i in selected)
    selected_ids = [x for i in selected for x in chunks[i]]
    pack_ids = [tok.bos_token_id] + selected_ids + query_ids
    budget = len(pack_ids) + cap
    row = {"task": task, "id": str(source_id), "index": index, "answers": answers,
           "source_prompt": prompt, "context_text": context, "query_text": query,
           "retrieval_question": question, "context_ids": context_ids, "query_ids": query_ids,
           "context_token_count": len(context_ids), "query_token_count": len(query_ids),
           "context_chunk_lengths": list(map(len, chunks)), "ordered_selected_indices": selected,
           "selected_chunk_lengths": [len(chunks[i]) for i in selected], "selector": selector,
           "topk": 12, "chunk_size": 512, "iter_hop_topk": 4, "max_new_tokens": cap,
           "bos_token_id": 1, "eos_token_ids": [2], "suppress_first_eos": True,
           "read_pack_tokens": len(pack_ids), "conservative_read_plus_cap": budget,
           "max_position_embeddings": 8192, "budget_valid": budget <= 8192,
           "truncation": "none", "padding": "none", "boundary": "explicit independently tokenized context/query",
           "context_ids_sha256": digest(context_ids), "query_ids_sha256": digest(query_ids),
           "selected_ids_sha256": digest(selected_ids), "pack_ids_sha256": digest(pack_ids),
           "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **extra}
    row["fixture_sha256"] = digest(row)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ALLOWED_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "non_qwen_FIXTURES_READY.json").exists()
    assert (args.model_dir / "non_qwen_TOKENIZER_READY.json").exists()
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    cfg = json.loads((args.model_dir / "config.json").read_text())
    assert cfg["max_position_embeddings"] == 8192 and cfg["num_attention_heads"] == cfg["num_key_value_heads"] == 32
    assert cfg["num_hidden_layers"] == 24 and cfg["rope_theta"] == 130000 and cfg["rope_scaling"] is None
    protocol = {"repo": REPO, "revision": REVISION, "model_dir": str(args.model_dir), "j": 8,
                "arms": ARMS, "native_chat_template": tok.chat_template, "qasper_template": "native SmolLM user with default system; assistant generation prompt; no thinking option",
                "ruler_template": "existing completion-style CoMem RULER; full terminal question/prefix; no chat template",
                "context_query_tokenization": "independent; add_special_tokens=False; no truncation/padding",
                "decode": "all arms greedy, first-step EOS suppressed, natural EOS thereafter, task cap",
                "native_bos_id": tok.bos_token_id, "native_eos_id": tok.eos_token_id,
                "normal_encode_adds_bos": tok.encode("a test", add_special_tokens=True) != tok.encode("a test", add_special_tokens=False),
                "explicit_single_pack_sink": True, "max_position_embeddings": 8192,
                "source_context_window_claim": "16k synthetic source target; native dense read limited to checked <=8192 pack+cap",
                "qwen_comparison": "different tokenizer/draws; only within-model arms claim identical inputs/packs"}
    probe = tok.apply_chat_template([{"role": "user", "content": "tokenizer probe"}], tokenize=False, add_generation_prompt=True)
    native_ids = tok.apply_chat_template([{"role": "user", "content": "tokenizer probe"}], tokenize=True, add_generation_prompt=True)
    # Some Transformers versions return a mapping rather than a bare list.
    if hasattr(native_ids, "keys"):
        native_ids = native_ids["input_ids"]
    assert list(native_ids) == tok.encode(probe, add_special_tokens=False)
    protocol["template_direct_encode_equal"] = True
    protocol["template_probe_rendered"] = probe
    protocol["template_probe_ids"] = list(native_ids)
    sources = [args.model_dir / n for n in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json")]
    source = HERE / "data/longbench/qasper.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(rows) == 200
    prompts_path = HERE / "protocol_sources/longbench/config/dataset2prompt.json"
    caps_path = HERE / "protocol_sources/longbench/config/dataset2maxlen.json"
    prompts = json.loads(prompts_path.read_text())
    caps = json.loads(caps_path.read_text())
    all_rows, summaries, files = [], {}, {}
    groups = {}
    qasper = []
    for index, row in enumerate(rows):
        marked = prompts["qasper"].format(**dict(row, context=row["context"] + BOUNDARY))
        assert marked.count(BOUNDARY) == 1
        rendered = tok.apply_chat_template([{"role": "user", "content": marked}], tokenize=False, add_generation_prompt=True)
        context, query = rendered.split(BOUNDARY)
        qasper.append(prepare_row(tok, "qasper", row.get("_id", row.get("id", index)), index, context, query,
                                 row["input"], row["answers"], caps["qasper"], "bm25", rendered.replace(BOUNDARY, ""),
                                 source_row_sha256=digest(row), template_kind="native_chat"))
    assert len({x["id"] for x in qasper}) == 200
    groups["qasper"] = qasper
    print(json.dumps({"task": "qasper", "n": 200, "max_budget": max(x["conservative_read_plus_cap"] for x in qasper)}), flush=True)
    essay = ROOT / "exp/data/pg19_essay.txt"
    assert essay.exists()
    R._ESSAY_PATH = str(essay)
    R._ESSAY_WORDS_CACHE = None
    for task in ("niah_single_2", "niah_multikey_1", "variable_tracking"):
        # Stable explicit seed; independent of Python's randomized tuple hash.
        base_seed = 42 + int(hashlib.sha256((REVISION + "/" + task + "/16k").encode()).hexdigest()[:12], 16)
        icl = R._make_vt_icl(random.Random(base_seed + 777), 4) if task == "variable_tracking" else None
        fixtures = []
        for index in range(50):
            prompt, answers, gold = R._build_sample(task, 16384, tok, random.Random(base_seed * 1000 + index), icl)
            marker = "\nQuestion:" if task == "variable_tracking" else "\nWhat"
            boundary = prompt.rfind(marker)
            assert boundary > 0
            context, query = prompt[:boundary], prompt[boundary:]
            fixtures.append(prepare_row(tok, task, f"{task}/16k/{index}", index, context, query,
                                        R._bare_question(prompt), answers, 60 if task == "variable_tracking" else 48,
                                        "iter_bm25" if task == "variable_tracking" else "bm25", prompt,
                                        source_target_tokens=16384, stable_base_seed=base_seed, gold=gold,
                                        template_kind="completion", source_token_count=len(tok.encode(prompt, add_special_tokens=False)),
                                        scientific_label="five-needle retrieval" if task == "variable_tracking" else task))
        groups[task] = fixtures
        print(json.dumps({"task": task, "n": 50, "max_budget": max(x["conservative_read_plus_cap"] for x in fixtures)}), flush=True)
    for task, fixtures in groups.items():
        path = args.out / f"non_qwen_{task}.jsonl"
        assert not path.exists(), "retain previous fixture attempts"
        with path.open("x") as stream:
            for row in fixtures:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        files[task] = file_info(path)
        summaries[task] = {"n": len(fixtures), "budget_valid": sum(x["budget_valid"] for x in fixtures),
                           "source_tokens_range": [min(x["context_token_count"] + x["query_token_count"] for x in fixtures), max(x["context_token_count"] + x["query_token_count"] for x in fixtures)],
                           "query_tokens_range": [min(x["query_token_count"] for x in fixtures), max(x["query_token_count"] for x in fixtures)],
                           "read_pack_tokens_range": [min(x["read_pack_tokens"] for x in fixtures), max(x["read_pack_tokens"] for x in fixtures)],
                           "maximum_read_plus_cap": max(x["conservative_read_plus_cap"] for x in fixtures),
                           "cap": fixtures[0]["max_new_tokens"]}
        all_rows.extend(fixtures)
    # Prespecified by fixture geometry only, never by model output/accuracy.
    indices = set()
    for key in ("query_token_count", "read_pack_tokens", "context_token_count"):
        indices.add(min(qasper, key=lambda x: (x[key], x["index"]))["index"])
        indices.add(max(qasper, key=lambda x: (x[key], -x["index"]))["index"])
    for index in range(200):
        if len(indices) >= 8:
            break
        indices.add(index)
    smoke = {"qasper": sorted(indices), **{task: [0, 49] for task in groups if task != "qasper"}}
    assert len(smoke["qasper"]) == 8
    result = {"status": "ready" if all(x["budget_valid"] for x in all_rows) else "budget_failed",
              "prepared_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "samples": len(all_rows), "tasks": summaries, "files": files, "smoke_indices": smoke,
              "smoke_examples": 14, "smoke_arms": ARMS, "smoke_generations": 70,
              "additional_native_j0_reference_sequences": 4,
              "native_reference_examples": [{"task": "qasper", "index": smoke["qasper"][0]}, {"task": "qasper", "index": smoke["qasper"][-1]}, {"task": "niah_single_2", "index": 0}, {"task": "variable_tracking", "index": 0}],
              "formal_core_4_arm_generations": 1400, "formal_with_fix_none_5_arm_generations": 1750,
              "sources": [file_info(p) for p in sources + [source, prompts_path, caps_path, essay, Path(R.__file__), ROOT / "COMem/comem/selectors.py"]],
              "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(), "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(), "model_loaded": False, "gpu_started": False}
    assert not result["cuda_initialized"]
    (args.out / "non_qwen_FIXTURES_AUDIT.json").write_text(json.dumps(result, indent=2))
    if result["status"] == "ready":
        (args.out / "non_qwen_FIXTURES_READY.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "samples": 350, "tasks": summaries}), flush=True)
    return 0 if result["status"] == "ready" else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
