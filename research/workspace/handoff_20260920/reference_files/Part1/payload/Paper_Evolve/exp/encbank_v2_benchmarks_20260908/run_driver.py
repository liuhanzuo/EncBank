"""Run an existing benchmark with an explicitly selected Encbank reader.

GPU admission belongs to the external queue. This file does not acquire a GPU lock.
Each attempt has a new native output directory; successful generations are reusable.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import importlib.metadata
import json
import math
import os
import sys
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for folder in (ROOT / "Encbank", ROOT / "exp", HERE, HERE / "vendor" / "babilong"):
    sys.path.insert(0, str(folder))

from reader_adapter import ARMS, GenerationCache, make_reader_factory, validate_options

BENCHMARKS = ("longbench", "longeval", "locomo", "babilong", "infinitebench")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


class ParsedArguments(Exception):
    def __init__(self, args):
        self.args_value = args


def capture_driver_args(module, argv):
    original = argparse.ArgumentParser.parse_args

    def capture(parser, args=None, namespace=None):
        parsed = original(parser, argv, namespace)
        raise ParsedArguments(parsed)

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            module.main()
        except ParsedArguments as exc:
            args = exc.args_value
    from eval import _cli
    _cli.normalize_args(args, task_attrs=("tasks",))
    return args


def file_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def resource_info(args):
    files = []
    model = Path(args.model_path)
    if model.is_dir():
        files.extend(p for p in model.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"})
    for attr in ("data_dir", "locomo_data", "lora_adapter", "retriever_path"):
        value = getattr(args, attr, None)
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return [file_info(p) for p in sorted(set(files))]


def pid_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, 0, pid)
        if handle:
            kernel.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # Access denied: conservatively treat as live.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class OutputLock:
    """Protect this output directory only; this is not a GPU allocation lock."""
    def __init__(self, output):
        self.path = output / "RUNNING.lock"

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            prior = json.loads(self.path.read_text(encoding="utf-8"))
            if pid_alive(prior["pid"]):
                raise RuntimeError(f"Output directory is in use by PID {prior['pid']}")
            self.path.unlink()  # The recorded process has exited; interrupted runs resume.
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": utc_now()}, f)
        return self

    def __exit__(self, *_):
        self.path.unlink()


def _shard_count(n, args, limit):
    if limit > 0:
        n = min(n, limit)
    return len(range(n)[args.shard_index::args.num_shards])


def install_data_checks(stack, module, benchmark, args, expected):
    """Preserve native data construction and scoring; reject silently missing cells."""
    sharded = args.num_shards > 1
    tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    if benchmark == "longbench":
        original = module.load_longbench_dataset
        tasks = args.tasks or module.DEFAULT_DATASETS
        if any(t not in module.DATASET2PROMPT for t in tasks):
            raise ValueError("Unsupported LongBench task; no prompt fallback is permitted.")
        def load(*a, **kw):
            data = original(*a, **kw)
            shard_tag = f"shard{args.shard_index}of{args.num_shards}" if sharded else "0"
            for task in tasks:
                count = _shard_count(len(data.get(task, [])), args, args.max_samples)
                if count <= 0:
                    raise ValueError(f"LongBench cell {task} has no samples")
                expected[f"{task}_{shard_tag}.jsonl"] = count
            return data
        stack.enter_context(patch.object(module, "load_longbench_dataset", load))
    elif benchmark == "locomo":
        original = module.build_locomo_samples
        def load(*a, **kw):
            data = original(*a, **kw)
            filtered = data
            if args.categories:
                categories = {int(c.strip()) for c in args.categories.split(",")}
                filtered = [s for s in data if s["category"] in categories]
            expected[f"preds{tag}.jsonl"] = _shard_count(len(filtered), args, args.max_samples)
            return data
        stack.enter_context(patch.object(module, "build_locomo_samples", load))
    elif benchmark == "longeval":
        for length in args.lengths:
            if length not in module._LENGTH_TOKENS:
                raise ValueError(f"Unsupported LongEval length {length}")
            expected[f"longeval_{length}{tag}.json"] = _shard_count(args.num_samples, args, -1)
    elif benchmark == "babilong":
        import prepare_babilong
        from babilong.prompts import DEFAULT_PROMPTS
        if any(t not in DEFAULT_PROMPTS for t in args.tasks):
            raise ValueError("Unsupported BABILong task")
        # Only local prepared cells are loaded. Official prompt/scoring layout stays native.
        def load(_dataset_name, length):
            data = {}
            for task in args.tasks:
                rows = prepare_babilong.load_task(task, length)
                data[task] = rows
                expected[f"{task}_{length}_*{tag}.csv"] = _shard_count(len(rows), args, args.limit)
            return data
        stack.enter_context(patch.object(module, "load_babilong_dataset", load))
    elif benchmark == "infinitebench":
        import prepare_infinitebench
        original = prepare_infinitebench.load_task
        def load(task, data_dir=None):
            rows = original(task, data_dir)
            expected[f"{task}_{args.shard_index}.jsonl"] = _shard_count(len(rows), args, args.max_samples)
            return rows
        stack.enter_context(patch.object(prepare_infinitebench, "load_task", load))


def validate_outputs(output, benchmark, args, expected):
    wanted_cells = (len(args.tasks) * len(args.lengths) if benchmark == "babilong" else
                    len(args.tasks) if benchmark == "infinitebench" else None)
    if not expected or (wanted_cells is not None and len(expected) != wanted_cells):
        raise RuntimeError("Driver did not visit every requested dataset cell")
    results = []
    official_scores = {}
    for pattern, count in expected.items():
        matches = list(output.glob(pattern))
        if len(matches) != 1 or count <= 0:
            raise RuntimeError(f"Missing/empty result cell {pattern}; expected {count} records")
        path = matches[0]
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))["records"]
        if len(rows) != count or any(r.get("pred") == "[OOM]" or r.get("output") == "[OOM]"
                or r.get("status") not in (None, "ok") for r in rows):
            raise RuntimeError(f"Incomplete result cell {path.name}: {len(rows)}/{count}")
        if benchmark in {"longbench", "longeval", "babilong", "infinitebench"}:
            key = "sample_index" if benchmark == "longeval" else "index"
            indices = [int(row[key]) for row in rows]
            wanted = [args.shard_index + i * args.num_shards for i in range(count)]
            if indices != wanted:
                raise RuntimeError(f"Missing, duplicate, or out-of-order source indices in {path.name}")
        for row in rows:
            if "score" in row and (not math.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1):
                raise RuntimeError(f"Invalid per-sample score in {path.name}")
        results.append({"file": str(path), "records": len(rows)})
        if benchmark == "babilong":
            from prepare_babilong import score_prediction
            task = path.name.split("_", 1)[0]
            values = [score_prediction(r["output"], r, task) for r in rows]
            official_scores[path.stem] = {"n": len(values), "accuracy": sum(values) / len(values), "scale": "0..1"}
    if official_scores:
        write_json(output / "scores_official.json", official_scores)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--arm", choices=ARMS, default="fix_all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Resolve configuration without loading model or writing outputs")
    parser.add_argument("--driver-help", action="store_true")
    wrapper, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    reserved = {"--out", "--output_dir", "--results_folder", "--output_name"}
    if any(arg.split("=", 1)[0] in reserved for arg in forwarded):
        parser.error("Native output overrides are not accepted; use wrapper --out only.")
    module = importlib.import_module(wrapper.benchmark + "_driver" if wrapper.benchmark in {"infinitebench", "babilong"} else "eval." + wrapper.benchmark)
    if wrapper.driver_help:
        with patch.object(sys, "argv", [module.__file__, "--help"]):
            module.main()
        return 0
    if wrapper.out is None:
        parser.error("--out is required")
    output = wrapper.out.resolve()
    args = capture_driver_args(module, forwarded + ["--out", "__ATTEMPT_OUTPUT__"])
    validate_options(args, wrapper.arm)
    explicit = wrapper.benchmark in {"infinitebench", "babilong"}
    if explicit and (args.sink_tokens != "bos" or args.selector not in {"bm25", "recency", "iter_bm25", "iter_bm25_adaptive", "dense_bge"}):
        raise ValueError("Explicit complete-query benchmark packs require a BOS/EOS sink and a token-based selector")
    if wrapper.arm == "cacheblend16" and not explicit:
        raise ValueError("cacheblend16 is available only for explicit-pack benchmark drivers")
    if wrapper.benchmark in {"longbench", "infinitebench"} and args.selector == "oracle":
        raise ValueError("This driver does not construct oracle needle indices; --selector oracle would be invalid.")
    if wrapper.benchmark == "babilong" and args.dataset_name != "RMT-team/babilong":
        raise ValueError("The prepared BABILong loader serves RMT-team/babilong only.")
    if wrapper.benchmark == "babilong":
        import prepare_babilong
        if any(t not in prepare_babilong.TASKS for t in args.tasks) or any(l not in prepare_babilong.LENGTHS for l in args.lengths):
            raise ValueError("Local prepared BABILong grid: --tasks qa1 qa2 qa3 qa5 --lengths 0k 1k 2k 4k 8k 16k 32k; select a subset explicitly.")
        args.data_dir = str(prepare_babilong.DATA_DIR)
    if wrapper.benchmark == "infinitebench" and args.data_dir is None:
        import prepare_infinitebench
        args.data_dir = str(prepare_infinitebench.DATA_DIR)
    sources = [Path(module.__file__), HERE / "reader_adapter.py", Path(__file__),
               ROOT / "exp" / "s15_ruler_lower.py", ROOT / "Encbank" / "encbank" / "model.py",
               ROOT / "Encbank" / "encbank" / "selectors.py", ROOT / "Encbank" / "eval" / "_common.py"]
    if wrapper.benchmark in {"babilong", "infinitebench"}:
        sources.append(HERE / f"prepare_{wrapper.benchmark}.py")
        sources.append(HERE / "benchmark_pack.py")
        vendor = HERE / "vendor" / wrapper.benchmark
        sources.extend(p for p in vendor.rglob("*.py") if p.is_file())
    if wrapper.arm == "cacheblend16":
        sources.extend([HERE / "cacheblend_contextual.py", ROOT / "Encbank/encbank/cacheblend.py"])
    configuration = {"schema_version": 1, "benchmark": wrapper.benchmark, "arm": wrapper.arm,
        "driver_options": vars(args), "sources": [file_info(p) for p in sources],
        "resources": resource_info(args), "cwd": str(Path.cwd()),
        "libraries": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "gpu_admission": "external queue; no internal GPU gate",
        "scoring": "official helper" if wrapper.benchmark in {"infinitebench", "babilong"} else "unchanged legacy driver; do not relabel as official",
        "resume": "exact token input and generation parameters; successful samples only"}
    if explicit:
        configuration["pack_protocol"] = "whole source/no truncation; explicit complete query; no chat wrapper; identical token selection/order; BOS then EOS sink fallback; first-token EOS suppressed"
    if wrapper.dry_run:
        print(json.dumps(configuration, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with OutputLock(output):
        contract_path = output / "run_config.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != configuration:
                raise ValueError("This output directory belongs to different options/code/resources. Use a new --out.")
        else:
            if any(p.name != "RUNNING.lock" for p in output.iterdir()):
                raise ValueError("Refusing to adopt a nonempty output directory with no run_config.json")
            write_json(contract_path, configuration)
        completed = output / "COMPLETED.json"
        if completed.exists():
            print(f"Already complete; outputs preserved: {completed}")
            return 0
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04d}"
        attempt.mkdir()
        native_output = attempt / "native"
        native_output.mkdir()
        metadata = {"started_at": utc_now(), "status": "running", "output_dir": str(native_output),
                    "driver_argv": forwarded + ["--out", str(native_output)], "pid": os.getpid()}
        write_json(attempt / "metadata.json", metadata)
        write_json(output / "LATEST.json", {"attempt": str(attempt), "output_dir": str(native_output)})
        cache = GenerationCache(output / "generations.sqlite3")
        def constructed(reader_info):
            metadata["reader"] = reader_info
            write_json(attempt / "metadata.json", metadata)
        try:
            with ExitStack() as stack:
                expected = {}
                install_data_checks(stack, module, wrapper.benchmark, args, expected)
                stack.enter_context(patch.object(module, "Encbank", make_reader_factory(wrapper.arm, cache, constructed, explicit_pack=explicit)))
                stack.enter_context(patch.object(sys, "argv", [module.__file__] + metadata["driver_argv"]))
                module.main()
                files = validate_outputs(native_output, wrapper.benchmark, args, expected)
            metadata.update(status="completed", files=files)
        except BaseException as exc:
            metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            metadata.update(finished_at=utc_now(), reused_generations=cache.hits, new_generations=cache.misses)
            cache.close()
            write_json(attempt / "metadata.json", metadata)
        write_json(completed, metadata)
        print(f"Completed; native outputs: {native_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
