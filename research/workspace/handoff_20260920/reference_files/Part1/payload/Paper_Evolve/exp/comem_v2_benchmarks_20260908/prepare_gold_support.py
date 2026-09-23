"""Map official HotpotQA supporting sentences into LongBench's expanded articles.

Alignment uses question identity, passage title identity, then exact or
Unicode/whitespace-normalized supporting sentence matching. No answer strings,
fuzzy similarity thresholds or inferred supporting facts enter the mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_offsets(text):
    chars, spans = [], []
    for i, char in enumerate(text):
        for expanded in unicodedata.normalize("NFKC", char):
            if expanded.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    spans.append([i, i + 1])
                elif chars:
                    spans[-1][1] = i + 1
            else:
                chars.append(expanded)
                spans.append([i, i + 1])
    if chars and chars[-1] == " ":
        chars.pop()
        spans.pop()
    return "".join(chars), spans


def occurrences(text, needle):
    start = 0
    while needle:
        found = text.find(needle, start)
        if found < 0:
            break
        yield found, found + len(needle)
        start = found + 1


def passage_sections(context):
    headings = list(re.finditer(r"(?m)^Passage (\d+):\s*\n([^\n]+)\n", context))
    sections = defaultdict(list)
    for number, heading in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(context)
        sections[normalized(heading.group(2))].append((heading.end(), end))
    return sections


def align_fact(context, sections, title, sentence):
    sections = sections.get(normalized(title), [])
    if not sections:
        return "title_not_found", []
    needle = sentence.strip()
    exact = [[start + left, start + right] for start, end in sections
             for left, right in occurrences(context[start:end], needle)]
    if exact:
        return "exact", exact
    matches = []
    for start, end in sections:
        canonical, offsets = normalized_offsets(context[start:end])
        for left, right in occurrences(canonical, normalized(needle)):
            matches.append([start + offsets[left][0], start + offsets[right - 1][1]])
    return ("unicode_whitespace", matches) if matches else ("sentence_not_found", [])


def build_mapping(longbench_rows, official_rows):
    questions = defaultdict(list)
    for row in official_rows:
        questions[normalized(row["question"])].append(row)
    records = []
    for index, lb in enumerate(longbench_rows):
        matches = questions[normalized(lb["input"])]
        record = {"index": index, "id": str(lb.get("_id", lb.get("id", index))),
                  "question": lb["input"], "context_chars": len(lb["context"]), "supporting_facts": []}
        if len(matches) != 1:
            record.update(source_match="missing" if not matches else "ambiguous", all_support_aligned=False)
            records.append(record)
            continue
        source = matches[0]
        paragraphs = dict(zip(source["context"]["title"], source["context"]["sentences"]))
        sections = passage_sections(lb["context"])
        for title, sent_id in zip(source["supporting_facts"]["title"], source["supporting_facts"]["sent_id"]):
            if title not in paragraphs or not 0 <= sent_id < len(paragraphs[title]):
                raise ValueError(f"Invalid official support annotation {source['id']}/{title}/{sent_id}")
            sentence = paragraphs[title][sent_id]
            status, spans = align_fact(lb["context"], sections, title, sentence)
            record["supporting_facts"].append({"title": title, "sent_id": sent_id,
                "sentence": sentence, "alignment": status, "context_char_spans": spans,
                "support_document_char_spans": [list(span) for span in sections.get(normalized(title), [])]})
        record.update(source_match="unique_question", source_id=source["id"],
            all_support_aligned=bool(record["supporting_facts"]) and all(f["context_char_spans"] for f in record["supporting_facts"]))
        records.append(record)
    statuses = Counter(f["alignment"] for r in records for f in r["supporting_facts"])
    return {"schema_version": 1, "task": "hotpotqa", "source": SOURCE_URL,
        "source_split": "official HotpotQA distractor validation", "license": "CC-BY-SA-4.0",
        "alignment_policy": "unique normalized question, normalized exact passage title, exact or Unicode/whitespace sentence match; no answer lookup/fuzzy inference",
        "summary": {"n": len(records), "source_matched_n": sum(r["source_match"] == "unique_question" for r in records),
            "all_support_aligned_n": sum(r["all_support_aligned"] for r in records), "fact_alignment_counts": dict(statuses)},
        "records": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--longbench", type=Path, default=HERE / "data/longbench/hotpotqa.jsonl")
    p.add_argument("--official-parquet", type=Path, default=HERE / "data/gold_support/hotpotqa_validation.parquet")
    p.add_argument("--out", type=Path, default=HERE / "data/gold_support/gold_support_mapping.json")
    args = p.parse_args()
    import pyarrow.parquet as pq
    rows = [json.loads(line) for line in args.longbench.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_mapping(rows, pq.read_table(args.official_parquet).to_pylist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
