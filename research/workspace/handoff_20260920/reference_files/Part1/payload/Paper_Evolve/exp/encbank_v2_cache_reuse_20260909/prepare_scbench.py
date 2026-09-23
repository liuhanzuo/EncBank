"""Prepare only two official SCBench tasks, on CPU, without loading a model.

The raw source is preserved. Optional Qwen tokenization prepares a named
retrieval adaptation, NOT an unchanged SCBench leaderboard evaluation.
"""
from __future__ import annotations

import argparse
import ast
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
CODE_REV = "a4eb395f949ea39e871f9bc586d683390692c6be"
DATA_REV = "283310bb8c5ba6909dd9a6b1be087d2937f76f6d"
TASKS = ("scbench_qa_eng", "scbench_choice_eng")
RAW_INFO = {
    "scbench_qa_eng": (59504656, "127ddb3cef2fcfc63c02e662466dead7ced84e20de6a3924df6926891394e01c"),
    "scbench_choice_eng": (47592729, "646aee28b3808ad19e66010be650dc13c454732e541d632862243549e41c2947"),
}
SOURCE_FILES = ("eval_utils.py", "compute_scores.py", "run_scbench.py", "args.py", "readme.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reused = path.exists()
    if not reused:
        part = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "Encbank-SCBench-CPU-preparation/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
        if expected and (part.stat().st_size, sha_file(part)) != expected:
            raise ValueError(f"Downloaded source failed official LFS check: {part}")
        part.replace(path)
    info = {"url": url, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha_file(path), "reused_existing_file": reused}
    if expected and (info["bytes"], info["sha256"]) != expected:
        raise ValueError(f"Existing source failed official LFS check: {path}")
    return info


def compile_pure(path, names, assignments=(), namespace=None):
    ns = {} if namespace is None else namespace
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in functions} != set(names):
        raise ValueError(f"Missing official functions: {names}")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assignments:
                    ns[target.id] = ast.literal_eval(node.value)
    if not set(assignments).issubset(ns):
        raise ValueError("Missing official template/scorer mapping")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    return ns


@lru_cache(maxsize=4)
def official(sources):
    sources = Path(sources)
    ns = compile_pure(sources / "eval_utils.py", {"create_scdq_prompt", "get_ground_truth"},
        {"multiturn_templates_scdq", "multiturn_follow_up_templates_in_chat_tempate",
         "DATA_NAME_TO_MAX_NEW_TOKENS"})
    return compile_pure(sources / "compute_scores.py",
        {"get_score_one_longdialogue_qa_eng", "get_score_one_longbook_choice_eng"},
        {"Multiturnbench_to_Infinitebench"}, ns)


def score_prediction(prediction, ground_truth, task, sources=None):
    """Exact official SCBench task mapping; QA is substring accuracy, NOT F1."""
    ns = official(str(sources or HERE / "fixtures/scbench_sources"))
    name = ns["Multiturnbench_to_Infinitebench"][task]
    return float(ns["get_score_one_" + name](prediction, ground_truth, "Qwen3-8B"))


def score_query(pred, row):
    """Common runner interface: (official 0..1 accuracy, actual scored text)."""
    sample = row["sample"]
    return score_prediction(pred, sample["ground_truth"], sample["task"]), pred


def load_workload(path=None):
    """Return (manifest, documents, queries); no tokenizer/model/GPU is loaded."""
    path = Path(path or HERE / "fixtures/scbench")
    if path.is_file():
        path = path.parent
    documents = json.loads((path / "documents.json").read_text(encoding="utf-8"))
    queries = [json.loads(line) for line in (path / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not metadata["retrieval_adapted"] or not metadata["complete_two_task_sources"]:
        raise ValueError("Unready or mislabeled SCBench fixture")
    assert len(documents) == metadata["documents"] and len(queries) == metadata["queries"]
    assert len({d["document_id"] for d in documents}) == len(documents)
    assert len({q["id"] for q in queries}) == len(queries)
    return metadata, documents, queries


def iter_raw(path):
    with Path(path).open("rb") as f:
        index = 0
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(f"Blank line in official raw source: {path}")
            yield index, offset, len(line), json.loads(line)
            index += 1


def old_ib_index(path):
    contexts, by_context_question, questions, count = set(), defaultdict(list), defaultdict(list), 0
    for _, _, _, row in iter_raw(path):
        context_hash = sha_bytes(row["context"].encode())
        contexts.add(context_hash)
        entry = {"id": row["id"], "answer": row["answer"], "options": row.get("options", [])}
        by_context_question[(context_hash, row["input"])].append(entry)
        questions[row["input"]].append(entry)
        count += 1
    return {"rows": count, "contexts": contexts, "pairs": by_context_question, "questions": questions}


class NoThinkingTokenizer:
    """Only model-specific adaptation: pin Qwen thinking off in official formatter."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, enable_thinking=False, **kwargs)


def token_setup(model_path, old_exp):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU token preparation requires CUDA_VISIBLE_DEVICES empty")
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 16:
        torch.set_num_interop_threads(16)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    root = Path(old_exp).resolve().parents[1]
    sys.path.insert(0, str(root / "Encbank"))
    from encbank import selectors
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return torch, tokenizer, selectors, cfg


def prepare(args):
    fixtures = args.out / "fixtures"
    raw_dir, sources = fixtures / "scbench_raw", fixtures / "scbench_sources"
    receipts = []
    for name in SOURCE_FILES:
        receipts.append(download(f"https://raw.githubusercontent.com/microsoft/MInference/{CODE_REV}/scbench/{name}", sources/name))
    receipts.append(download(f"https://raw.githubusercontent.com/microsoft/MInference/{CODE_REV}/LICENSE", sources/"LICENSE"))
    receipts.append(download(f"https://huggingface.co/datasets/microsoft/SCBench/resolve/{DATA_REV}/README.md", sources/"dataset_README.md"))
    for task in TASKS:
        receipts.append(download(f"https://huggingface.co/datasets/microsoft/SCBench/resolve/{DATA_REV}/data/{task}.jsonl",
                                 raw_dir/f"{task}.jsonl", RAW_INFO[task]))
        print(json.dumps({"download_ready": task}), flush=True)
    ns = official(str(sources))
    assert ns["Multiturnbench_to_Infinitebench"][TASKS[0]] == "longdialogue_qa_eng"
    assert ns["Multiturnbench_to_Infinitebench"][TASKS[1]] == "longbook_choice_eng"
    old = {}
    for task in TASKS:
        ib_task = task.replace("scbench_", "longbook_")
        old[task] = old_ib_index(args.ib_dir / f"{ib_task}.jsonl")
    token_tools = token_setup(args.tokenizer, args.old_exp) if args.tokenizer else None
    totals, all_queries = {}, []
    common_dir = fixtures / "scbench"
    common_dir.mkdir(parents=True, exist_ok=True)
    doc_stream = (common_dir / "documents.json.tmp").open("w", encoding="utf-8") if token_tools else None
    if doc_stream:
        doc_stream.write("[\n")
    documents_written = 0
    for task in TASKS:
        groups, seen, ctx_seen, histogram, chars = [], set(), set(), Counter(), []
        overlap = Counter()
        for index, offset, line_bytes, raw in iter_raw(raw_dir/f"{task}.jsonl"):
            assert set(("id", "context", "multi_turns")).issubset(raw)
            assert isinstance(raw["context"], str) and raw["context"]
            assert isinstance(raw["multi_turns"], list) and len(raw["multi_turns"]) >= 1
            assert raw["id"] not in seen
            seen.add(raw["id"])
            context_hash = sha_bytes(raw["context"].encode())
            ctx_seen.add(context_hash)
            histogram[len(raw["multi_turns"])] += 1
            chars.append(len(raw["context"]))
            overlap["exact_context_groups"] += context_hash in old[task]["contexts"]
            labels = ns["get_ground_truth"](raw, task)
            bare = ns["create_scdq_prompt"](raw, task, None, False)
            assert len(bare["prompts"]) == 1 + len(labels)
            assert raw["context"] in bare["prompts"][0]
            group_id = f"{task}/{raw['id']}"
            group = {"group_id": group_id, "task": task, "source_id": raw["id"], "source_index": index,
                "source": {"file": f"fixtures/scbench_raw/{task}.jsonl", "byte_offset": offset,
                           "line_bytes": line_bytes, "context_sha256": context_hash,
                           "source_revision": DATA_REV},
                "context_chars": len(raw["context"]), "query_count": len(labels),
                "bare_context_prompt_sha256": sha_bytes(bare["prompts"][0].encode()),
                "context_reconstruction": "official create_scdq_prompt(raw,task,tokenizer,use_chat_template=True), Qwen enable_thinking=False",
                "query_ids": [], "context_tokens": None, "token_file": None}
            if token_tools:
                torch, tok, selectors, cfg = token_tools
                formatted = ns["create_scdq_prompt"](raw, task, NoThinkingTokenizer(tok), True)
                context_ids = tok.encode(formatted["prompts"][0], add_special_tokens=False)
                assert context_ids and sys.byteorder == "little"
                token_rel = f"fixtures/scbench_tokens/{task}/{raw['id']}.i32"
                token_file = args.out / token_rel
                token_file.parent.mkdir(parents=True, exist_ok=True)
                buf = array("i", context_ids)
                assert buf.itemsize == 4
                token_file.write_bytes(buf.tobytes())
                chunks = list(torch.tensor(context_ids, dtype=torch.long).split(512))
                group.update(context_tokens=len(context_ids), token_file=token_rel,
                    token_dtype="little-endian signed int32", token_file_sha256=sha_file(token_file),
                    formatted_context_prompt_sha256=sha_bytes(formatted["prompts"][0].encode()),
                    source_exceeds_model_window=len(context_ids) > cfg["max_position_embeddings"])
                order = [f"{group_id}/q{i}" for i in range(len(labels))]
                doc = {"document_id": group_id, "task": task, "source_id": raw["id"],
                    "context_ids": context_ids, "raw_context": raw["context"], "formatted_context": formatted["prompts"][0],
                    "context_tokens": len(context_ids), "context_sha256": context_hash,
                    "source": group["source"], "query_count": len(labels), "retrieval_adapted": True,
                    "query_ids_ordered": order, "prefixes": {"1": order[:1], "full": order}}
                if documents_written:
                    doc_stream.write(",\n")
                doc_stream.write(json.dumps(doc, ensure_ascii=False))
                documents_written += 1
            for turn_index, (turn, label, query_text) in enumerate(zip(raw["multi_turns"], labels, bare["prompts"][1:])):
                assert isinstance(turn["input"], str) and turn["input"]
                assert isinstance(turn["answer"], str) and turn["answer"]
                assert turn["input"] in query_text
                if task == "scbench_choice_eng":
                    assert len(turn["options"]) == 4 and turn["options"].count(turn["answer"]) == 1
                    assert all(option in query_text for option in turn["options"])
                matches = old[task]["pairs"].get((context_hash, turn["input"]), [])
                overlap["exact_context_question_queries"] += bool(matches)
                overlap["question_only_queries"] += turn["input"] in old[task]["questions"]
                exact = [m for m in matches if turn["answer"] in (m["answer"] if isinstance(m["answer"], list) else [m["answer"]])
                         and m["options"] == turn.get("options", [])]
                overlap["exact_context_question_answer_options_queries"] += bool(exact)
                query_id = f"{group_id}/q{turn_index}"
                query = {"query_id": query_id, "group_id": group_id, "task": task,
                    "source_id": raw["id"], "source_index": index, "turn_index": turn_index,
                    "input": turn["input"], "answer": turn["answer"], "options": turn.get("options", []),
                    "ground_truth": label, "query_text_official_bare": query_text,
                    "max_new_tokens": ns["DATA_NAME_TO_MAX_NEW_TOKENS"][task],
                    "official_metric": "case_insensitive_substring_accuracy" if task == "scbench_qa_eng" else "official_multiple_choice_accuracy",
                    "official_scorer": "get_score_one_" + ns["Multiturnbench_to_Infinitebench"][task],
                    "old_ib_context_question_ids": [m["id"] for m in matches],
                    "old_ib_exact_qa_options_ids": [m["id"] for m in exact]}
                assert query["max_new_tokens"] == 40
                assert score_prediction("", label, task, str(sources)) == 0.0
                assert score_prediction(label[-1], label, task, str(sources)) == 1.0
                if token_tools:
                    query_prompt = formatted["prompts"][turn_index+1]
                    query_tokens = tok.encode(query_prompt, add_special_tokens=False)
                    bare_ids = tok.encode(turn["input"], add_special_tokens=False)
                    selected = selectors.select_context_chunk_indices("bm25", chunks, bare_ids, 12)
                    read_tokens = 1 + len(query_tokens) + sum(len(chunks[i]) for i in selected)
                    assert query_tokens and read_tokens + 40 <= cfg["max_position_embeddings"]
                    query.update(query_prompt=query_prompt, query_token_ids=query_tokens,
                        bare_question_token_ids=bare_ids, selected_indices=list(selected),
                        read_pack_tokens=read_tokens, chunk_size=512, topk=12,
                        sink_id=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id,
                        eos_id=tok.eos_token_id, context_tokens=len(context_ids),
                        tokenization="separate context/query, no added special tokens, no truncation or padding")
                    query.update(id=query_id, document_id=group_id, stream_index=turn_index,
                        query_ids=query_tokens, formatted_query=query_prompt,
                        raw_query_suffix=query_text, retrieval_question=turn["input"], question=turn["input"],
                        answers=label, category=None, selector="bm25", source_refs=[dict(group["source"], turn_index=turn_index)],
                        pack={"input_tokens": len(context_ids)+len(query_tokens), "context_tokens": len(context_ids),
                              "query_tokens": len(query_tokens), "context_chunks": len(chunks),
                              "selected_indices": list(selected), "read_pack_tokens": read_tokens,
                              "truncation": "none", "context_query_boundary": "official scdq; independent tokenization; no padding"},
                        sample={"task": task, "answers": label, "ground_truth": label,
                                "max_new_tokens": 40, "question": turn["input"],
                                "options": turn.get("options", []), "official_metric": query["official_metric"]})
                all_queries.append(query)
                group["query_ids"].append(query_id)
            groups.append(group)
            print(json.dumps({"task": task, "prepared_group": index+1, "queries": len(labels),
                              "context_tokens": group["context_tokens"]}), flush=True)
        with (fixtures/f"{task}.groups.jsonl").open("w", encoding="utf-8") as f:
            for row in groups:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        queries = [q for q in all_queries if q["task"] == task]
        with (fixtures/f"{task}.queries.jsonl").open("w", encoding="utf-8") as f:
            for row in queries:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        totals[task] = {"groups": len(groups), "unique_contexts": len(ctx_seen), "queries": len(queries),
            "queries_per_group_histogram": dict(sorted(histogram.items())), "context_chars_range": [min(chars), max(chars)],
            "old_infinitebench_rows": old[task]["rows"], "old_infinitebench_unique_contexts": len(old[task]["contexts"]),
            "overlap": dict(overlap), "official_metric": queries[0]["official_metric"],
            "official_cap": 40, "self_answer_and_blank_checks": len(queries)*2}
        if token_tools:
            totals[task].update(context_tokens_range=[min(g["context_tokens"] for g in groups), max(g["context_tokens"] for g in groups)],
                source_groups_exceeding_model_window=sum(g["source_exceeds_model_window"] for g in groups),
                read_pack_tokens_range=[min(q["read_pack_tokens"] for q in queries), max(q["read_pack_tokens"] for q in queries)])
    receipt = {"status": "prepared", "created_at": now(), "hostname": socket.gethostname(),
        "code_revision": CODE_REV, "dataset_revision": DATA_REV, "tasks": totals,
        "download_receipts": receipts, "tokenizer": str(args.tokenizer) if args.tokenizer else None,
        "tokenized_retrieval_adaptation_ready": bool(token_tools), "model_loaded": False, "gpu_experiment_started": False,
        "scope": "Complete two named official raw tasks; no other SCBench tasks and no GPU execution",
        "protocol": "SCBench multi-request retrieval adaptation; Qwen thinking off; same full source; BM25 top12x512; independent questions, no earlier answers; natural EOS with shared first-token suppression; cap40",
        "not_leaderboard_equivalent": "Original source is complete; full-context prefill is replaced by retrieval. Original run_scbench can middle-truncate source to model budget; this adaptation does not.",
        "timing_not_ready": "No latency/quality generation produced; root must select finite local5090 smoke/full scope and gate it.",
        "cpu_threads": {"torch": 2, "interop": 16, "OMP": os.environ.get("OMP_NUM_THREADS"), "MKL": os.environ.get("MKL_NUM_THREADS")} if token_tools else "stdlib only"}
    if token_tools:
        assert not torch.cuda.is_initialized()
        receipt["cuda_initialized"] = False
        doc_stream.write("\n]\n")
        doc_stream.close()
        (common_dir/"documents.json.tmp").replace(common_dir/"documents.json")
        with (common_dir/"queries.jsonl").open("w", encoding="utf-8") as f:
            for query in all_queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")
        metadata = {"schema_version": 1, "benchmark": "SCBench two-task retrieval adaptation",
            "retrieval_adapted": True, "complete_two_task_sources": True,
            "documents": documents_written, "queries": len(all_queries), "tasks": totals,
            "document_count": documents_written, "unique_queries": len(all_queries), "prefixes": [1, "full"],
            "documents_file": "documents.json", "queries_file": "queries.jsonl",
            "protocol": "scbench-two-task-retrieval-reuse-v1",
            "model": "Qwen3-8B", "tokenizer_local_target": "/srv/encbank/legacy_workspace/models/Qwen3-8B",
            "tokenizer_preparation_path": str(args.tokenizer), "code_revision": CODE_REV, "dataset_revision": DATA_REV,
            "tokenizer_files": [{"file": name, "bytes": (args.tokenizer/name).stat().st_size,
                                 "sha256": sha_file(args.tokenizer/name)} for name in
                                ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
                                if (args.tokenizer/name).is_file()],
            "mode": "multi-request; no previous questions/answers", "chunk_size": 512, "topk": 12,
            "selector": "bare-question token BM25, original chunk order", "max_new_tokens": 40,
            "native_qwen_adaptation": "thinking disabled; BOS else EOS sink; shared first-token EOS suppression then natural EOS",
            "gpu_experiment_started": False, "latency_results_available": False,
            "load_workload_return": "(manifest:dict, documents:list, queries:list)",
            "score_query_return": "(official_accuracy_0_to_1, scored_prediction_text)"}
        save_json(common_dir/"metadata.json", metadata)
        save_json(common_dir/"manifest.json", metadata)
    save_json(fixtures/"scbench_PREPARED.json", receipt)
    print(json.dumps({"status": "prepared", "tasks": totals}, ensure_ascii=False), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--ib-dir", type=Path, default=HERE.parent/"encbank_v2_benchmarks_20260908/data/infinitebench")
    p.add_argument("--old-exp", type=Path, default=HERE.parent/"encbank_v2_benchmarks_20260908")
    p.add_argument("--tokenizer", type=Path, help="Optional existing local tokenizer directory; no model weights loaded")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
