"""CPU-only InfiniteBench data, prompt, and official-scoring helpers.

Only En.QA / En.MC are supported. Data are the unmodified official JSONL files.
The relevant pure functions are compiled from the vendored official sources,
avoiding compute_scores.py's unrelated eager evaluate.load("rouge") call.
No model, tokenizer, GPU, truncation, or chat wrapper is invoked here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import string
from typing import Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "infinitebench"
VENDOR_DIR = ROOT / "vendor" / "infinitebench" / "src"
TASKS = ("longbook_qa_eng", "longbook_choice_eng")
EXPECTED_COUNTS = {"longbook_qa_eng": 351, "longbook_choice_eng": 229}
MAX_NEW_TOKENS = {task: 40 for task in TASKS}


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"Supported tasks: {TASKS}; got {task!r}")


def _compile_functions(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in nodes}
    if found != names:
        raise RuntimeError(f"Official source {path} is missing {names - found}")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@lru_cache(maxsize=1)
def _official() -> dict:
    namespace = {"re": re, "string": string, "Counter": Counter,
                 "Path": Path, "Optional": Optional}
    templates = {}
    tree = ast.parse((VENDOR_DIR / "prompt.py").read_text(encoding="utf-8"))
    wanted = {"gpt4_templates": "gpt4", "yarn_mistral_templates": "yarn-mistral"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    templates[wanted[target.id]] = ast.literal_eval(node.value)
    if set(templates) != set(wanted.values()):
        raise RuntimeError("Required official prompt templates were not found")
    namespace["MODEL_TO_PROMPT_TEMPLATE"] = templates
    _compile_functions(VENDOR_DIR / "eval_utils.py", {"create_prompt", "get_answer"}, namespace)
    _compile_functions(VENDOR_DIR / "compute_scores.py", {
        "normalize_answer", "f1_score", "qa_f1_score",
        "get_score_one_longbook_qa_eng", "get_score_one_longbook_choice_eng",
    }, namespace)
    return namespace


def iter_task(task: str, data_dir: str | Path | None = None):
    """Yield raw official records, in file order, without changing their context."""
    _check_task(task)
    path = Path(data_dir) / f"{task}.jsonl" if data_dir is not None else DATA_DIR / f"{task}.jsonl"
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "context", "input", "answer"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            if task == "longbook_choice_eng" and len(row.get("options", [])) != 4:
                raise ValueError(f"{path}:{line_number}: expected exactly four options")
            yield row


def load_task(task: str, data_dir: str | Path | None = None) -> list[dict]:
    """Load all records; use iter_task for streaming large book contexts."""
    return list(iter_task(task, data_dir))


def format_prompt(row: dict, task: str, prompt_style: str = "yarn-mistral") -> str:
    """Apply an explicitly named official template; never silently omit the question."""
    _check_task(task)
    official = _official()
    if prompt_style not in official["MODEL_TO_PROMPT_TEMPLATE"]:
        raise ValueError("prompt_style must be 'yarn-mistral' or 'gpt4'")
    return official["create_prompt"](row, task, prompt_style, DATA_DIR)


def get_answer(row: dict, task: str):
    """Return official accepted labels; choices include answer text and option letter."""
    _check_task(task)
    return _official()["get_answer"](row, task)


def score_prediction(pred: str, row: dict, task: str) -> float:
    """Official score in [0, 1]: max normalized token F1 (QA), parsed accuracy (MC)."""
    _check_task(task)
    if not isinstance(pred, str):
        raise TypeError("Prediction must be a string; preserve missing/failed generation separately")
    scorer = _official()[f"get_score_one_{task}"]
    return float(scorer(pred, get_answer(row, task), "yarn-mistral"))


def audit_task(task: str, data_dir: str | Path | None = None) -> dict:
    """CPU protocol check on every record, including self-answer and blank scoring."""
    count = 0
    ids = set()
    context_chars = []
    label_shapes = Counter()
    errors = []
    for row in iter_task(task, data_dir):
        count += 1
        if row["id"] in ids:
            errors.append(f"duplicate id {row['id']}")
        ids.add(row["id"])
        context_chars.append(len(row["context"]))
        answer = get_answer(row, task)
        label_shapes[type(answer).__name__] += 1
        own_answer = answer[-1] if task == "longbook_choice_eng" else answer[0]
        if score_prediction(own_answer, row, task) != 1.0:
            errors.append(f"id {row['id']}: own answer did not score 1")
        if score_prediction("", row, task) != 0.0:
            errors.append(f"id {row['id']}: blank output did not score 0")
        prompt = format_prompt(row, task)
        if row["input"] not in prompt or row["context"] not in prompt:
            errors.append(f"id {row['id']}: prompt dropped question or context")
    if count != EXPECTED_COUNTS[task]:
        errors.append(f"count {count} != official expected {EXPECTED_COUNTS[task]}")
    return {
        "task": task, "count": count, "unique_ids": len(ids),
        "context_chars_min": min(context_chars), "context_chars_max": max(context_chars),
        "context_chars_mean": sum(context_chars) / count,
        "answer_shapes": dict(label_shapes), "max_new_tokens_official": MAX_NEW_TOKENS[task],
        "default_prompt_style": "yarn-mistral", "context_truncation": "none in helper",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = [audit_task(task, args.data_dir) for task in args.tasks]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if any(row["errors"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
