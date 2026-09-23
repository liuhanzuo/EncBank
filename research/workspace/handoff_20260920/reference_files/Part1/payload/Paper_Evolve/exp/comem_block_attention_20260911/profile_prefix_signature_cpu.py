"""CPU-only Python metadata profile; NOT model inference speed or GPU memory.

Builds a width-reduced 36-layer Qwen using the real configuration and j12/r32
LoRA module layout. No checkpoint weights, prepared examples or forward pass.
The observed cost is only a CPU signature function on this diagnostic topology.
"""
from __future__ import annotations
import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent


def measure(function, rounds=5, iterations=20):
    for _ in range(3):
        function()
    values = []
    for _ in range(rounds):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        values.append((time.perf_counter_ns() - begin) / iterations / 1e6)
    return dict(milliseconds_per_call_by_round=values, median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values), rounds=rounds, calls_per_round=iterations)


def run(model_config, out, *, compare_candidate=False):
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Remote CPU-only: hide CUDA explicitly")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    sys.path.insert(0, str(HERE.parent / "comem_v2_benchmarks_20260908"))
    sys.path.insert(0, str(HERE.parents[1] / "COMem"))
    from train_8b_baseline import attach_lora
    from comem.model import CoMem
    from native_prefix_reader import NativePrefixCoMemReader, _tensor_signature
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 16:
        torch.set_num_interop_threads(16)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    torch.manual_seed(42)
    raw = json.loads(Path(model_config).read_text())
    if raw.get("num_hidden_layers") != 36 or raw.get("num_key_value_heads") != 8:
        raise ValueError("Expected the existing 36-layer/8-KV-head Qwen configuration")
    reduced = dict(raw, vocab_size=101, hidden_size=256, intermediate_size=512,
                   head_dim=8, num_attention_heads=32, num_key_value_heads=8,
                   bos_token_id=1, eos_token_id=2)
    config = Qwen3Config.from_dict(reduced)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    adapters = attach_lora(model, 12, 32, 32., torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cm = CoMem(model, resume_j=12)
    reader = NativePrefixCoMemReader(cm)
    scalar_types = (str, int, float, bool, type(None))

    def parameters():
        return tuple((name, _tensor_signature(t)) for name, t in model.named_parameters())

    def buffers():
        return tuple((name, _tensor_signature(t)) for name, t in model.named_buffers())

    def modules():
        return tuple((name, id(module), type(module).__module__, type(module).__qualname__,
            tuple(sorted((key, value) for key, value in vars(module).items()
                         if not key.startswith("_") and type(value) in scalar_types)))
            for name, module in model.named_modules())

    functions = {"parameters_and_tensor_metadata": parameters, "buffers_and_tensor_metadata": buffers,
                 "modules_and_public_scalars": modules,
                 "config_to_dict_and_json": lambda: json.dumps(cm.config.to_dict(), sort_keys=True, default=str),
                 "original_model_signature": reader._model_signature}
    candidate = None
    if compare_candidate:
        from signature_optimized_prefix_reader import SignatureOptimizedNativePrefixReader
        candidate = SignatureOptimizedNativePrefixReader(cm)
        if reader._model_signature() != candidate._model_signature():
            raise RuntimeError("Candidate signature does not equal the original tuple")
        functions["candidate_model_signature"] = candidate._model_signature
    before = reader._model_signature()
    results = {name: measure(function) for name, function in functions.items()}
    # Counterbalanced pairs limit order/warmup bias for this CPU microdiagnosis.
    pairs = []
    if candidate is not None:
        for index in range(6):
            order = ("original_model_signature", "candidate_model_signature")
            if index % 2:
                order = tuple(reversed(order))
            row = {name: measure(functions[name], rounds=1, iterations=30) for name in order}
            pairs.append(dict(order=list(order), timings=row))
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        reader._model_signature()
    profiler.disable()
    profiler.dump_stats(str(out / "signature.prof"))
    top = []
    for (filename, line, function), (primitive, calls, self_s, cumulative_s, _) in sorted(
            pstats.Stats(profiler).stats.items(), key=lambda item: item[1][2], reverse=True)[:20]:
        top.append(dict(file=Path(filename).name, line=line, function=function,
                        primitive_calls=primitive, calls=calls, self_s=self_s, cumulative_s=cumulative_s))
    if reader._model_signature() != before or torch.cuda.is_initialized():
        raise RuntimeError("Profile mutated signature state or initialized CUDA")
    if candidate is not None and candidate._model_signature() != before:
        raise RuntimeError("Candidate signature changed during diagnostic")
    sources = {Path(__file__).name: Path(__file__), "native_prefix_reader.py": HERE / "native_prefix_reader.py"}
    if candidate is not None:
        sources["signature_optimized_prefix_reader.py"] = HERE / "signature_optimized_prefix_reader.py"
    value = dict(checked_utc=datetime.now(timezone.utc).isoformat(), device="cpu", cuda_initialized=False,
        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
        torch=str(torch.__version__), transformers=transformers.__version__,
        model_config_sha256=hashlib.sha256(Path(model_config).read_bytes()).hexdigest(),
        diagnostic_config=config.to_dict(),
        diagnostic_shape=dict(layers=36, hidden=256, intermediate=512, vocab=101, q_heads=32, kv_heads=8, head_dim=8),
        layout=dict(j=12, rank=32, alpha=32., lora_modules=len(adapters),
                    modules=len(list(model.named_modules())), parameters=len(list(model.named_parameters())),
                    buffers=len(list(model.named_buffers()))),
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        component_timings=results, counterbalanced_pairs=pairs, original_cprofile_top_self_time=top,
        signatures_unchanged=True, compared_signatures_equal=candidate is not None,
        model_forward_calls=0, checkpoint_weights_loaded=False,
        formal_inference_timing=False, formal_inference_memory=False,
        scope="Python metadata function on reduced-width CPU topology; not8B/model/GPU inference speed or memory")
    (out / "profile.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-candidate", action="store_true")
    args = parser.parse_args()
    result = run(args.model_config, args.out, compare_candidate=args.compare_candidate)
    print(json.dumps({key: result[key] for key in ("layout", "component_timings", "model_forward_calls", "scope")}))
