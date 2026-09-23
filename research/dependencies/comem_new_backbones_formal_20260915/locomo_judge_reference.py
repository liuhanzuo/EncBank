#!/usr/bin/env python
"""Semantic LoCoMo judge used by the paper.

The script grades categories 1--4 through an OpenAI-compatible chat-completions
endpoint and applies the local abstention rule to category 5. It never embeds
credentials or a provider-specific URL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b", re.IGNORECASE)


PROMPT = """You are grading a model's answer against the gold answer for a question about a long, multi-session dialogue (the LoCoMo benchmark).

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Grade whether the model answer is CORRECT. It is CORRECT if it conveys the same key information as the gold answer (a semantic match), even if phrased differently, more verbosely, or with extra correct context. It is WRONG if it contradicts the gold answer, omits the key information, or is empty or refuses when an answer exists. For date/time answers, accept any unambiguous equivalent phrasing.

Respond with ONLY one word: CORRECT or WRONG."""


def _endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def _parse_vote(text: str) -> int:
    vote = (text or "").strip().upper()
    if vote.startswith("CORRECT"):
        return 1
    if vote.startswith("WRONG"):
        return 0
    if "CORRECT" in vote and "WRONG" not in vote:
        return 1
    return 0


def _request(url: str, api_key: str, model: str, prompt: str, seed: int,
             retries: int) -> tuple[int, str]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": seed,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            return _parse_vote(text), text
        except (OSError, KeyError, ValueError, urllib.error.URLError):
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return 0, "[API_FAILURE]"


def _read_predictions(path: Path) -> list[dict]:
    files = sorted(path.glob("preds*.jsonl")) if path.is_dir() else [path]
    rows, seen = [], set()
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade LoCoMo predictions")
    parser.add_argument("--predictions", required=True,
                        help="prediction JSONL or directory containing preds*.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit("Set --base_url or OPENAI_BASE_URL.")
    url = _endpoint(args.base_url)
    api_key = os.environ.get(args.api_key_env, "")
    rows = _read_predictions(Path(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    judged = []
    for row in rows:
        category = int(row.get("category", -1))
        if category == 5:
            pred = (row.get("pred", "") or "").strip()
            correct = int(not pred or bool(_REFUSAL_RE.search(pred)))
            raw = "LOCAL_ABSTENTION_RULE"
        else:
            gold = " OR ".join(str(x) for x in row.get("answers", []))
            prompt = PROMPT.format(question=row.get("question", ""), gold=gold,
                                   pred=row.get("pred", ""))
            correct, raw = _request(url, api_key, args.model, prompt, args.seed,
                                    args.retries)
        judged.append({
            "id": row["id"], "category": category, "judge_correct": correct,
            "judge_raw": raw, "model": args.model,
        })
        with output.open("w", encoding="utf-8") as handle:
            for item in judged:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    score = 100.0 * sum(x["judge_correct"] for x in judged) / max(1, len(judged))
    print(json.dumps({"n": len(judged), "judge_score": score}, indent=2))


if __name__ == "__main__":
    main()
