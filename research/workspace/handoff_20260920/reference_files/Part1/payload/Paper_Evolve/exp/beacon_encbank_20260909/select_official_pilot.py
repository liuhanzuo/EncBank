"""Select a bounded common-input pilot from the fully audited official pool.

CPU only. Raw data and the eligible pool are read-only. A compact index chooses
IDs; a second streaming pass extracts only those complete prepared records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def key(seed, identity):
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def quotas(counts, total, minimum=5):
    if total < len(counts) or total > sum(counts.values()):
        raise ValueError("Insufficient eligible conversations or pilot budget")
    result = {s: min(n, minimum, total // len(counts)) for s, n in counts.items()}
    left = total - sum(result.values())
    while left:
        capacity = {s: counts[s] - result[s] for s in counts if counts[s] > result[s]}
        capacity_sum = sum(capacity.values())
        fractional = {s: left * n / capacity_sum for s, n in capacity.items()}
        additions = {s: min(capacity[s], int(v)) for s, v in fractional.items()}
        if not any(additions.values()):
            winner = sorted(capacity, key=lambda s: (-fractional[s], s))[0]
            additions[winner] = 1
        for source, n in additions.items():
            result[source] += n
            left -= n
    return result


def choose(index, *, train_count=1000, dev_per_source=5, seed=42):
    by_split = {"train": defaultdict(list), "dev": defaultdict(list)}
    seen, group_splits, document_splits, fingerprints = set(), {}, {}, set()
    for row in index:
        identity = row.get("conversation_id", row.get("id"))
        split = row["split"]
        if split not in by_split or not identity or identity in seen:
            raise ValueError("Unknown split or duplicate/missing eligible identity")
        seen.add(identity)
        for field, mapping in (("overlap_group_id", group_splits), ("document_id", document_splits)):
            value = row.get(field)
            if not value or (value in mapping and mapping[value] != split):
                raise ValueError(f"Missing or overlapping {field}")
            mapping[value] = split
        fingerprint = row.get("tokenizer_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Missing tokenizer fingerprint")
        fingerprints.add(fingerprint)
        if not 7200 <= row["original_template_token_count"] <= 20000:
            raise ValueError("Pool contains a conversation outside common length bounds")
        if row["target_token_count"] < 1:
            raise ValueError("Empty supervision in eligible pool")
        by_split[split][row["source"]].append({**row, "id": identity})
    if len(fingerprints) != 1 or not by_split["dev"]:
        raise ValueError("Mixed tokenizer versions or no held-out conversations")
    allocation = quotas({s: len(v) for s, v in by_split["train"].items()}, train_count)
    training = []
    for source, n in sorted(allocation.items()):
        training.extend(sorted(by_split["train"][source], key=lambda r: key(seed, r["id"]))[:n])
    training.sort(key=lambda r: key(seed, r["id"]))
    development = []
    for source, rows in sorted(by_split["dev"].items()):
        # Distinct overlap groups get priority; do not split a connected group to
        # manufacture source coverage. One QA per document is not assumed.
        selected_groups = set()
        for row in sorted(rows, key=lambda r: key(seed, r["id"])):
            if row["overlap_group_id"] in selected_groups:
                continue
            development.append(row)
            selected_groups.add(row["overlap_group_id"])
            if len(selected_groups) == dev_per_source:
                break
    pool = [row for rows in by_split["train"].values() for row in rows]
    stress = {}
    for dimension in ("raw_uncompressed_read_pack_tokens", "document_token_count",
                      "reader_template_token_count", "target_token_count"):
        row = max(pool, key=lambda r: (r[dimension], r["id"]))
        stress[row["id"]] = row
    for source, rows in sorted(by_split["train"].items()):
        row = max(rows, key=lambda r: (r["raw_uncompressed_read_pack_tokens"], r["id"]))
        stress[row["id"]] = row
    probe = sorted(stress.values(), key=lambda r: (-r["raw_uncompressed_read_pack_tokens"], r["id"]))
    probe_dev = []
    for source in sorted({r["source"] for r in development}):
        probe_dev.append(next(r for r in development if r["source"] == source))
    return {"seed": seed, "tokenizer_fingerprint": next(iter(fingerprints)),
            "allocation": allocation, "train": training, "dev": development,
            "probe_train": probe, "probe_dev": probe_dev,
            "missing_development_sources": sorted(set(allocation) - set(by_split["dev"])),
            "source_mixture": "source-proportional eligible training counts with a minimum of five where available",
            "development_policy": "up to five distinct overlap groups per available source; no group splitting",
            "probe_policy": "global pack/document/reader/target maxima plus maximum-pack representative of every source; quality not used"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--dev-per-source", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-raw-pack-tokens", type=int, default=0,
                        help="Optional shared resource subset; excludes whole conversations, never truncates")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new pilot output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.max_raw_pack_tokens < 0:
        raise ValueError("A common resource bound cannot be negative")
    index = list(read_jsonl(args.index))
    retained = [row for row in index if not args.max_raw_pack_tokens or
                row['raw_uncompressed_read_pack_tokens'] <= args.max_raw_pack_tokens]
    selection = choose(retained, train_count=args.train_count,
                       dev_per_source=args.dev_per_source, seed=args.seed)
    selection['common_resource_subset'] = {'max_raw_pack_tokens': args.max_raw_pack_tokens or None,
        'eligible_before': len(index), 'eligible_after': len(retained),
        'excluded_whole_conversations': len(index)-len(retained),
        'policy': 'all arms share complete selected conversations; no token or answer truncation',
        'retained_source_counts': dict(Counter(row['source'] for row in retained))}
    groups = {name: {r["id"]: r for r in selection[name]}
              for name in ("train", "dev", "probe_train", "probe_dev")}
    fetched = {name: {} for name in groups}
    wanted = set().union(*(set(ids) for ids in groups.values()))
    for row in read_jsonl(args.pool):
        identity = row.get("conversation_id", row.get("id"))
        if identity not in wanted:
            continue
        for name, records in groups.items():
            if identity not in records:
                continue
            if identity in fetched[name]:
                raise ValueError("Duplicate requested record in prepared pool")
            expected = records[identity]
            for field in ("document_id", "overlap_group_id", "source", "split",
                          "tokenizer_fingerprint", "target_token_count", "original_template_token_count"):
                if row.get(field) != expected.get(field):
                    raise ValueError(f"Prepared record/index differs at {identity}: {field}")
            fetched[name][identity] = row
    for name, records in groups.items():
        if set(fetched[name]) != set(records):
            raise ValueError(f"Selected records missing: {name}")
    files = {}
    for name, records in groups.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for identity in records:
                stream.write(json.dumps(fetched[name][identity], ensure_ascii=False) + "\n")
        files[name] = {"path": path.name, "sha256": sha256_file(path), "conversations": len(records),
                       "sources": dict(Counter(r["source"] for r in records.values())),
                       "target_tokens": sum(r["target_token_count"] for r in records.values()),
                       "document_tokens": sum(r["document_token_count"] for r in records.values())}
    for stage, prefix in (("main", ""), ("probe", "probe_")):
        ordered_ids = {"train_ids": list(groups[prefix+"train"]),
                       "dev_ids": list(groups[prefix+"dev"])}
        (args.out / f"{stage}_selection.json").write_text(
            json.dumps(ordered_ids, indent=2)+"\n", encoding="utf-8")
    receipt = {"complete": True, "seed": args.seed, "files": files,
               "selection": {**{k: v for k, v in selection.items() if k not in groups},
                             **{f"{name}_ids": list(records) for name, records in groups.items()}},
               "scope": "bounded data-only pilot selection; no model training or GPU resource claim",
               "pool_sha256": sha256_file(args.pool), "index_sha256": sha256_file(args.index)}
    (args.out / "selection.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "files": files}, ensure_ascii=False))


if __name__ == "__main__":
    main()
