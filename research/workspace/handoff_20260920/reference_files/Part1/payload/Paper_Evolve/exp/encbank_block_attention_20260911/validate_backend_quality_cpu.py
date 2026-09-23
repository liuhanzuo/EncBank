"""Tiny generation/profile tests; CLI may run only on remote Linux CPU."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import unittest

HERE=Path(__file__).resolve().parent


def run_validation(device="cpu",out_dir=None):
    if device not in ("cpu","cuda"):
        raise ValueError("Expected cpu or externally admitted cuda")
    if device=="cpu" and sys.platform!="linux":
        raise RuntimeError("CPU/Torch tests run on remote Linux")
    import torch
    import transformers
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=16:
        torch.set_num_interop_threads(16)
    out=Path(out_dir) if out_dir is not None else HERE/"results"/("backend_quality_"+device)
    out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    previous=os.environ.get("SPARSE_TEST_DEVICE")
    os.environ["SPARSE_TEST_DEVICE"]=device
    stream=io.StringIO()
    try:
        import test_backend_quality_generation as tests
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
    finally:
        if previous is None:
            os.environ.pop("SPARSE_TEST_DEVICE",None)
        else:
            os.environ["SPARSE_TEST_DEVICE"]=previous
    log=out/(stamp+".log")
    log.write_text(stream.getvalue(),encoding="utf-8")
    from evaluate_backend_quality import source_hashes
    sources=source_hashes()
    for name in ("test_backend_quality_generation.py","validate_backend_quality_cpu.py",
                 "test_same_math_diagnostic.py","same_math_diagnostic.py","backend_parity.py","test_sparse_reader.py"):
        sources[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    receipt={"passed":result.wasSuccessful(),"tests_run":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
        "kind":"independent-free-generation-EOS-and-profile-validation-not-task-quality",
        "device":device,"host":platform.node(),"torch":str(torch.__version__),"python":sys.version,
        "transformers":transformers.__version__,
        "cuda_runtime":torch.version.cuda,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_num_threads":torch.get_num_threads(),"torch_num_interop_threads":torch.get_num_interop_threads(),
        "log":str(log),"source_sha256":sources,"updated_utc":datetime.now(timezone.utc).isoformat()}
    path=out/(stamp+".json")
    receipt["receipt_path"]=str(path)
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(stream.getvalue(),flush=True)
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false")
    result=run_validation("cpu",args.out)
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()
