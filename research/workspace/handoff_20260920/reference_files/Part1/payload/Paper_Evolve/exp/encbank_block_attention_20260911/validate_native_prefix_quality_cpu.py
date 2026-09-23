"""Bounded remote CPU validation for native-prefix QA generation and scheduling.

No model download, CUDA initialization or inference timing. Receipts bind the
actual workspace sources imported by tests, including the existing LoRA loader.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


def run_validation(out):
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run on remote Linux with CUDA_VISIBLE_DEVICES explicitly empty")
    os.environ["SPARSE_TEST_DEVICE"] = "cpu"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 16:
        torch.set_num_interop_threads(16)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checks must not initialize CUDA")
    modules = [importlib.import_module(name) for name in
        ("test_native_prefix_quality", "test_prefix_quality_protocol", "test_remote_prefix_quality_queue")]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / "tests.log").write_text(stream.getvalue(), encoding="utf-8")
    observations = getattr(modules[0], "QUALITY_OBSERVATIONS", [])
    (out / "observations.json").write_text(json.dumps(observations, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    integration = getattr(modules[0], "GROUP_PROTOCOL_OBSERVATIONS", [])
    (out / "integration_observations.json").write_text(json.dumps(integration, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from evaluate_native_prefix_quality import source_hashes
    sources = source_hashes()
    explicit = ("validate_native_prefix_quality_cpu.py", "remote_prefix_quality_queue.py")
    for name in explicit:
        sources[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    workspace = HERE.parents[1]
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if workspace in path.parents and path.suffix == ".py":
            name = Path(os.path.relpath(path, HERE)).as_posix()
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = dict(checked_utc=datetime.now(timezone.utc).isoformat(), device="cpu",
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cuda_initialized=torch.cuda.is_initialized(),
        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
        torch=str(torch.__version__), transformers=transformers.__version__, tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        tiny_generation_records=len(observations),
        passed=result.wasSuccessful() and not result.skipped and not torch.cuda.is_initialized(),
        source_sha256=sources, formal_inference_timing=False, formal_inference_memory=False,
        scope="CPU QA orchestration/cache checks and queue policy; no8B/GPU quality or timing")
    (out / "checks.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    receipt = run_validation(parser.parse_args().out)
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_sha256"}))
    raise SystemExit(0 if receipt["passed"] else 1)
