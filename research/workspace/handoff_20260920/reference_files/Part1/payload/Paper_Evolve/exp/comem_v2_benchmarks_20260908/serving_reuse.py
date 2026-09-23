"""Write a whole document once, then load/reposition its cache for fresh queries.

Independent Qwen3 serving-cost harness. It never caches generated answers. Stored
fix_all chunks contain h_j plus post-k_norm/pre-RoPE K and V on CPU. Query masks,
query bottom-band prefill, upper-band prefill, and decoding execute every time.
pub/pub_sink store h_j only; j0 stores original token IDs and embeds at query time.

Production timing is restricted to the local Windows RTX 5090. GPU admission
uses exp/gpu_gate.py; CPU mode is explicitly test-only and never timing eligible.
Example:
  python serving_reuse.py --model /models/Qwen3-8B --context-file exp/data/pg19_essay.txt
      --out /results/serving --lengths 32768 131072 --query-counts 1 10 100
      --generation-lengths 16 128 --tiers cpu disk --arms fix_all pub pub_sink j0

The default cost workload uses 100 distinct excerpt-based questions and fixed
output lengths; it is a cost microbenchmark, not a quality benchmark. Disk reads
use the ordinary OS page cache; no cold-storage bandwidth claim is made.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for folder in (ROOT / "COMem", ROOT / "exp"):
    sys.path.insert(0, str(folder))

import torch
from transformers.cache_utils import DynamicCache
from comem import CoMem
from comem import selectors as selectors
from s15_ruler_lower import CoMemLower  # Same attention-kernel policy for all arms.

ARMS = ("fix_all", "pub", "pub_sink", "j0")
ALL_ARMS = ARMS + ("pub_lora",)
REQUIRED_GPU = "NVIDIA GeForce RTX 5090"


def device_provenance(device, *, allow_cpu=False):
    """Use the actual tensor device, never the requested label, as timing evidence."""
    device = torch.device(device)
    import transformers
    record = {"device": str(device), "hostname": socket.gethostname(),
              "platform": platform.system(), "platform_release": platform.release(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "torch_cpu_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
              "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
              "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "transformers": transformers.__version__, "required_gpu": REQUIRED_GPU,
              "timing_eligible": False}
    if device.type == "cpu" and allow_cpu:
        record.update(device_name="CPU", purpose="CPU correctness test only")
        return record
    if device.type != "cuda" or platform.system() != "Windows":
        raise ValueError("Serving timings require the local Windows RTX 5090; CPU requires --cpu-test-only")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    if properties.name != REQUIRED_GPU or index != 0:
        raise ValueError(f"Serving timing device must be local cuda:0 {REQUIRED_GPU}; got {index}: {properties.name}")
    record.update(device=f"cuda:{index}", device_name=properties.name,
                  total_memory_bytes=properties.total_memory,
                  compute_capability=[properties.major, properties.minor],
                  device_uuid=str(getattr(properties, "uuid", "unavailable")),
                  timing_eligible=True, purpose="local RTX 5090 serving measurement")
    return record


def validate_adapter_checkpoint(adapter_dir, marker_path=None):
    """The sync process publishes this marker only after validating the 4000-step export."""
    adapter_dir = Path(adapter_dir).resolve()
    marker_path = Path(marker_path) if marker_path else adapter_dir.parent / "TRANSFER_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (marker.get("complete") is not True or marker.get("step") != 4000
            or marker.get("adapter_load_mode") != "peft_unmerged"
            or Path(marker.get("adapter_dir", "")).resolve() != adapter_dir):
        raise ValueError("pub_lora requires a synchronized, unmerged, completed 4000-step final adapter")
    status = json.loads((adapter_dir.parent / "status.json").read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("step") != 4000 or status.get("target_steps") != 4000:
        raise ValueError("Intermediate checkpoints cannot supply final pub_lora timing")
    metadata = json.loads((adapter_dir.parent / "metadata.json").read_text(encoding="utf-8"))
    recipe = metadata["recipe"]
    expected = {"steps": 4000, "j": 12, "rank": 32, "alpha": 32,
                "loss": "published", "adapter_dtype": "float32"}
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("Adapter recipe differs from the published-method 4000-step baseline")
    for name in ("adapter.pt", "adapter_model.safetensors", "adapter_config.json"):
        file = adapter_dir / name
        expected_size = marker.get("files", {}).get("final/" + name)
        if not file.is_file() or file.stat().st_size <= 0 or expected_size != file.stat().st_size:
            raise ValueError(f"Incomplete or changed synchronized adapter file: {file}")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("r") != 32 or config.get("lora_alpha") != 32:
        raise ValueError("PEFT adapter configuration differs from the completed training recipe")
    return {"adapter_dir": str(adapter_dir), "completion_marker": str(marker_path.resolve()),
            "step": 4000, "load_mode": "peft_unmerged", "recipe": recipe,
            "files": marker["files"]}


def load_unmerged_adapter(model, adapter_dir):
    """Standard PEFT LoRA on an unquantized base, without optional TorchAO dispatch."""
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("pub_lora serving requires the unquantized base checkpoint")
    from unittest.mock import patch
    from peft import PeftModel
    # The user site has TorchAO 0.13; PEFT 0.20 otherwise probes it even for plain
    # nn.Linear weights and raises its >=0.16 requirement. No quantized layer is
    # used here. Skip only that optional dispatcher for the duration of loading.
    with patch("peft.tuners.lora.model.dispatch_torchao", return_value=None):
        wrapped = PeftModel.from_pretrained(model, str(adapter_dir), autocast_adapter_dtype=True).eval()
    adapted = wrapped.base_model.model
    adapter_parameters = [p for name, p in adapted.named_parameters() if "lora_" in name]
    if not adapter_parameters or any(p.dtype != torch.float32 for p in adapter_parameters):
        raise ValueError("pub_lora must retain the FP32 PEFT adapter branches")
    return adapted


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [map_tensors(v, function) for v in value]
    return value


def save_tensor_file(value, path):
    with Path(path).open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def save_json(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class ReusableReader:
    def __init__(self, model, j, tokenizer, arm="fix_all", model_id="unspecified", adapter_info=None):
        if arm not in ALL_ARMS:
            raise ValueError(f"Unknown arm {arm}")
        if getattr(model.config, "model_type", None) != "qwen3":
            raise ValueError("This independent store is validated for dense Qwen3 only")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("Use a single model device for the measured serving harness")
        has_adapter = any("lora_" in name for name, _ in model.named_parameters())
        if has_adapter != (arm == "pub_lora"):
            raise ValueError("Only the pub_lora arm may use a model with an active LoRA adapter")
        self.hardware = device_provenance(next(model.parameters()).device, allow_cpu=True)
        self.adapter_info = adapter_info
        self.arm, self.model_id = arm, str(model_id)
        self.cm = CoMemLower(model, j, tokenizer) if arm == "fix_all" else CoMem(
            model, 0 if arm == "j0" else j, tokenizer=tokenizer)
        self.cm.write_sink = arm in {"fix_all", "pub_sink"}
        if arm == "fix_all" and j <= 0:
            raise ValueError("fix_all requires j > 0")
        self.path = None
        self.payloads = None

    def clock(self):
        if self.cm.device.type == "cuda":
            torch.cuda.synchronize(self.cm.device)
        return time.perf_counter()

    def signature(self):
        config = self.cm.config
        return {"arm": self.arm, "j": self.cm.resume_j, "model_id": self.model_id,
                "dtype": str(self.cm.dtype), "model_config": config.to_dict(),
                "adapter": self.adapter_info,
                "sink_id": self.cm._sink_prefix_id(), "sink_write": "standalone, no prepended prefix"}

    @torch.no_grad()
    def _capture_payload(self, ids, sink=False):
        cm = self.cm
        if self.arm == "j0":
            return None
        if self.arm != "fix_all":
            saved = cm.write_sink
            try:
                if sink:
                    cm.write_sink = False
                return {"h": cm.write_chunk(ids)}
            finally:
                cm.write_sink = saved
        ids = cm._as_ids(ids)
        if not sink:
            prefix = torch.tensor([[cm._sink_prefix_id()]], device=cm.device)
            ids = torch.cat([prefix, ids], 1)
        h, kv = cm._capture_lower(ids)
        drop = 0 if sink else 1
        return {"h": h[:, drop:, :], "kv": {
            layer: [k[:, :, drop:, :], v[:, :, drop:, :]] for layer, (k, v) in kv.items()}}

    @torch.no_grad()
    def write_store(self, context_ids, path, chunk_size=512):
        """Capture every document chunk once; serialize compact CPU payloads."""
        path = Path(path)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if path.exists() and any(path.iterdir()):
            raise ValueError("Write requires a new empty store directory")
        path.mkdir(parents=True, exist_ok=True)
        if self.cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.cm.device)
            baseline = torch.cuda.memory_allocated(self.cm.device)
        else:
            baseline = 0
        started = self.clock()
        ids = torch.as_tensor(context_ids, dtype=torch.long, device="cpu").reshape(-1).clone()
        if ids.numel() == 0:
            raise ValueError("Document must be nonempty")
        chunks = [c.clone() for c in ids.split(chunk_size)]
        compute_s = copy_s = serialize_s = 0.0
        payload_bytes = 0
        t = self.clock()
        save_tensor_file(chunks, path / "tokens.pt")
        serialize_s += self.clock() - t
        entries = [("sink.pt", [self.cm._sink_prefix_id()], True)]
        entries += [(f"chunk_{i:06d}.pt", chunk, False) for i, chunk in enumerate(chunks)]
        if self.arm != "j0":
            for filename, chunk, sink in entries:
                t = self.clock()
                payload = self._capture_payload(chunk, sink=sink)
                compute_s += self.clock() - t
                t = self.clock()
                # clone severs views into a larger tensor containing the dropped BOS.
                cpu_payload = map_tensors(payload, lambda x: x.detach().to("cpu").contiguous().clone())
                copy_s += self.clock() - t
                payload_bytes += tensor_bytes(cpu_payload)
                t = self.clock()
                save_tensor_file(cpu_payload, path / filename)
                serialize_s += self.clock() - t
                del payload, cpu_payload
        metadata = {"schema_version": 1, "signature": self.signature(),
                    "n_tokens": ids.numel(), "n_chunks": len(chunks), "chunk_size": chunk_size,
                    "payload_tensor_bytes": payload_bytes,
                    "raw_token_bytes": tensor_bytes(chunks),
                    "cache_key_space": "post-k_norm/pre-RoPE K; unchanged V; depth-j hidden",
                    "validity": "Reuse only with the exact recorded checkpoint/adapter and tokenizer"}
        save_json(metadata, path / "store.json")
        total_s = self.clock() - started
        peak = torch.cuda.max_memory_allocated(self.cm.device) if self.cm.device.type == "cuda" else 0
        return {"write_total_s": total_s, "write_compute_s": compute_s,
                "write_device_to_cpu_s": copy_s, "write_serialize_s": serialize_s,
                "serialized_bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                "payload_tensor_bytes": payload_bytes, "raw_token_bytes": tensor_bytes(chunks),
                "n_document_tokens": ids.numel(), "n_document_chunks": len(chunks),
                "capture_calls": len(entries) if self.arm == "fix_all" else 0,
                "write_peak_allocated_bytes": peak, "write_incremental_peak_bytes": max(0, peak-baseline)}

    def open_store(self, path, tier="cpu"):
        if tier not in {"cpu", "disk"}:
            raise ValueError("tier must be cpu or disk")
        self.close_store()
        started = self.clock()
        self.path, self.tier = Path(path), tier
        self.metadata = json.loads((self.path / "store.json").read_text(encoding="utf-8"))
        # JSON round-trip makes config tuples/lists comparable.
        if self.metadata["signature"] != json.loads(json.dumps(self.signature())):
            raise ValueError("Store checkpoint/configuration/arm does not match the reader")
        self.chunks = torch.load(self.path / "tokens.pt", map_location="cpu", weights_only=True)
        startup_bytes = (self.path / "tokens.pt").stat().st_size + (self.path / "store.json").stat().st_size
        self.payloads = None
        if tier == "cpu" and self.arm != "j0":
            filenames = ["sink.pt"] + [f"chunk_{i:06d}.pt" for i in range(len(self.chunks))]
            self.payloads = {}
            for filename in filenames:
                self.payloads[filename] = torch.load(self.path / filename, map_location="cpu", weights_only=True)
                startup_bytes += (self.path / filename).stat().st_size
        return {"startup_load_s": self.clock() - started, "startup_read_bytes": startup_bytes,
                "resident_cpu_tensor_bytes": tensor_bytes(self.chunks) + tensor_bytes(self.payloads),
                "tier": tier, "disk_cache_policy": "ordinary OS page cache; no eviction"}

    def close_store(self):
        self.payloads = None
        self.chunks = []
        if hasattr(self.cm, "_bottom"):
            self.cm._bottom = None

    def _load_selected(self, indices):
        if self.arm == "j0":
            return [torch.tensor([self.cm._sink_prefix_id()], dtype=torch.long)] + [self.chunks[i] for i in indices], 0
        filenames = ["sink.pt"] + [f"chunk_{i:06d}.pt" for i in indices]
        if self.payloads is not None:
            return [self.payloads[f] for f in filenames], 0
        return [torch.load(self.path / f, map_location="cpu", weights_only=True) for f in filenames], sum(
            (self.path / f).stat().st_size for f in filenames)

    def _prepare_read(self, payloads):
        cm = self.cm
        if self.arm == "j0":
            h = [cm.embed_tokens(ids.reshape(1, -1)) for ids in payloads]
            return h[0], h[1:]
        sink_h = payloads[0]["h"]
        selected_h = [p["h"] for p in payloads[1:]]
        if self.arm == "fix_all":
            cache = DynamicCache(config=cm.config)
            positions, offset = [], 0
            for payload in payloads:
                n = payload["h"].shape[1]
                positions.append(torch.arange(offset, offset+n, device=cm.device).unsqueeze(0))
                offset += n
            for layer in range(cm.resume_j):
                keys = [cm._rotate(p["kv"][layer][0], pos) for p, pos in zip(payloads, positions)]
                values = [p["kv"][layer][1] for p in payloads]
                cache.update(torch.cat(keys, 2), torch.cat(values, 2), layer)
            cm._bottom = {"cache": cache, "M": offset, "q_off": offset}
        return sink_h, selected_h

    @torch.no_grad()
    def query_ids(self, query_ids, *, selected_indices=None, selector="bm25", topk=12,
                  bare_question_ids=None, max_new_tokens=16, force_length=False,
                  capture_logits=False):
        if self.path is None or not self.chunks:
            raise ValueError("Open a complete document store before querying")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        query = torch.as_tensor(query_ids, dtype=torch.long, device="cpu").reshape(-1)
        if query.numel() == 0:
            raise ValueError("Query must be nonempty")
        cm = self.cm
        if cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cm.device)
            baseline = torch.cuda.memory_allocated(cm.device)
        else:
            baseline = 0
        start = self.clock()
        if selected_indices is None:
            if selector not in {"bm25", "recency"}:
                raise ValueError("Serving workload supports explicit indices, bm25 or recency")
            selected_indices = selectors.select_context_chunk_indices(
                selector, self.chunks, list(bare_question_ids) if bare_question_ids is not None else query.tolist(), topk)
        indices = [int(i) for i in selected_indices]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.chunks) for i in indices):
            raise ValueError("Invalid or duplicate selected indices")
        retrieval_end = self.clock()
        cpu_payloads, load_bytes = self._load_selected(indices)
        load_end = self.clock()
        transfer_bytes = tensor_bytes(cpu_payloads) + tensor_bytes(query)
        payloads = map_tensors(cpu_payloads, lambda x: x.to(cm.device))
        gpu_query = query.to(cm.device)
        transfer_end = self.clock()
        step_logits, generated = [], []
        try:
            sink_h, selected_h = self._prepare_read(payloads)
            prepare_end = self.clock()
            q_h, bottom_cache, q_pos = cm.write_prefill(gpu_query)
            logits, top_cache, pack_pos = cm.read_prefill(sink_h, selected_h, q_h)
            first_logits = logits[0, -1].float().clone()
            _, eos = cm._bos_eos(cm.tokenizer)
            if eos is not None:
                first_logits[eos] = -float("inf")
            if capture_logits:
                step_logits.append(first_logits.cpu().clone())
            token = int(first_logits.argmax().item())
            generated.append(token)
            first_end = self.clock()
            for _ in range(1, max_new_tokens):
                logits = cm.decode_step(token, bottom_cache, top_cache, q_pos, pack_pos)
                q_pos += 1
                pack_pos += 1
                next_logits = logits[0, -1].float().clone()
                if force_length and eos is not None:
                    next_logits[eos] = -float("inf")
                if capture_logits:
                    step_logits.append(next_logits.cpu().clone())
                token = int(next_logits.argmax().item())
                if not force_length and eos is not None and token == eos:
                    break
                generated.append(token)
            end = self.clock()
            peak = torch.cuda.max_memory_allocated(cm.device) if cm.device.type == "cuda" else 0
            stats = {"selected_indices": indices, "read_tokens": 1+sum(len(self.chunks[i]) for i in indices)+len(query),
                     "query_tokens": len(query), "retrieval_s": retrieval_end-start,
                     "load_s": load_end-retrieval_end, "load_bytes": load_bytes,
                     "transfer_s": transfer_end-load_end, "transfer_bytes": transfer_bytes,
                     "rotate_prepare_s": prepare_end-transfer_end, "read_prefill_s": first_end-prepare_end,
                     "ttft_s": first_end-start, "decode_s": end-first_end, "total_s": end-start,
                     "generated_tokens": len(generated), "decode_steps": len(step_logits)-1 if capture_logits else pack_pos-(1+sum(len(self.chunks[i]) for i in indices)+len(query)),
                     "fixed_generation_length": force_length, "peak_allocated_bytes": peak,
                     "incremental_peak_bytes": max(0, peak-baseline), "capture_calls": 0}
            if capture_logits:
                stats["step_logits"] = step_logits
            return generated, stats
        finally:
            if hasattr(cm, "_bottom"):
                cm._bottom = None


def make_questions(tokens, tokenizer, count, chunk_size):
    """Distinct, deterministic excerpt questions; only workload cost is evaluated."""
    questions = []
    n_chunks = max(1, (len(tokens)+chunk_size-1)//chunk_size)
    for i in range(count):
        chunk = (i*n_chunks)//count
        excerpt = tokenizer.decode(tokens[chunk*chunk_size:chunk*chunk_size+32], skip_special_tokens=True)
        text = f'What does the passage beginning "{excerpt}" describe? Answer using the document. Request {i+1}.'
        questions.append({"id": i, "text": text, "bare_question": excerpt})
    return questions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32768, 131072])
    parser.add_argument("--query-counts", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--generation-lengths", nargs="+", type=int, default=[16, 128])
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ARMS))
    parser.add_argument("--tiers", nargs="+", choices=["cpu", "disk"], default=["cpu", "disk"])
    parser.add_argument("--j", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--allow-eos", action="store_true")
    parser.add_argument("--cpu-test-only", action="store_true",
                        help="Allow CPU correctness checks; results are ineligible as timing evidence")
    parser.add_argument("--adapter-ckpt", type=Path)
    parser.add_argument("--adapter-completion-marker", type=Path)
    parser.add_argument("--gpu-need-gb", type=float, default=22.0)
    parser.add_argument("--gpu-idle-slack-gb", type=float, default=5.0)
    parser.add_argument("--gpu-cap-gb", type=float, default=28.0,
                        help="Torch allocation cap in decimal GB, matching exp/gpu_gate.py")
    parser.add_argument("--gpu-wait-seconds", type=int, default=24*3600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not math.isfinite(args.gpu_idle_slack_gb) or not 0 < args.gpu_idle_slack_gb <= 5:
        parser.error("--gpu-idle-slack-gb must be > 0 and <= 5 GiB; admission uses strict <")
    if any(n <= 0 for n in args.lengths+args.query_counts+args.generation_lengths):
        parser.error("All workload sizes must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new empty output directory")
    if args.cpu_test_only != (torch.device(args.device).type == "cpu"):
        parser.error("CPU requires --cpu-test-only; this flag cannot label a CUDA run")
    if args.adapter_ckpt and (args.arms != ["pub_lora"] or args.j != 12):
        parser.error("An adapter may only be loaded in an isolated --arms pub_lora --j 12 process")
    if "pub_lora" in args.arms and not args.adapter_ckpt:
        parser.error("pub_lora requires --adapter-ckpt and the completed 4000-step sync marker")
    if args.adapter_completion_marker and not args.adapter_ckpt:
        parser.error("--adapter-completion-marker requires --adapter-ckpt")
    adapter_info = validate_adapter_checkpoint(args.adapter_ckpt, args.adapter_completion_marker) if args.adapter_ckpt else None
    admission = None
    if not args.cpu_test_only:
        if platform.system() != "Windows" or torch.device(args.device) != torch.device("cuda:0"):
            parser.error("Production serving timing is restricted to local Windows cuda:0 RTX 5090")
        expected_environment = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                "TOKENIZERS_PARALLELISM": "false"}
        if any(os.environ.get(k) != v for k, v in expected_environment.items()):
            parser.error("Production serving requires OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 and TOKENIZERS_PARALLELISM=false")
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 16:
            torch.set_num_interop_threads(16)
        print(json.dumps({"serving_thread_policy": expected_environment,
                          "torch_cpu_threads": torch.get_num_threads(),
                          "torch_interop_threads": torch.get_num_interop_threads()}), flush=True)
        from gpu_gate import acquire_gpu
        admission = acquire_gpu(need_gb=args.gpu_need_gb, idle_slack_gb=args.gpu_idle_slack_gb,
                                max_wait=args.gpu_wait_seconds, tag="serving_reuse " + str(args.out))
    hardware = device_provenance(args.device, allow_cpu=args.cpu_test_only)
    hardware["gpu_admission"] = admission
    if hardware["timing_eligible"]:
        torch.cuda.set_per_process_memory_fraction(
            min(args.gpu_cap_gb * 1e9 / hardware["total_memory_bytes"], 1.0), device=args.device)
        hardware["torch_memory_cap_bytes"] = int(args.gpu_cap_gb * 1e9)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa", local_files_only=True).to(args.device).eval()
    if adapter_info:
        model = load_unmerged_adapter(model, args.adapter_ckpt)
    source_started = time.perf_counter()
    source_text = args.context_file.read_text(encoding="utf-8")
    source_read_s = time.perf_counter()-source_started
    tokenization_started = time.perf_counter()
    source = tokenizer.encode(source_text, add_special_tokens=False)
    source_tokenization_s = time.perf_counter()-tokenization_started
    shared_source_preprocess_s = source_read_s+source_tokenization_s
    if not source:
        raise ValueError("Context source is empty")
    # One short unmeasured model warmup; document write remains fully measured.
    with torch.no_grad():
        model(input_ids=torch.tensor([source[:32]], device=args.device), use_cache=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_config=model.config.to_dict(), source_tokens=len(source), hardware=hardware,
                  adapter=adapter_info,
                  workload="controlled excerpt questions; document repeats only when source is too short",
                  prompt="bare completion; no chat wrapper; unscored cost workload",
                  accounting="Input boundary is an already-tokenized fixed-length document, plus raw query texts. end_to_end_total_s includes CPU tensor/chunk construction, whole-document write, startup load and all query totals. Query totals include query tokenization/retrieval/load/H2D/rotation/prefill/decode to token IDs. Full-source read/tokenization and workload prefix/repetition preparation are reported separately and NOT charged to each document. This is not raw-text document end-to-end latency. Model load/warmup is service-process startup and excluded; client question construction is excluded.",
                  shared_source_read_s=source_read_s, shared_source_tokenization_s=source_tokenization_s,
                  shared_source_preprocess_s=shared_source_preprocess_s,
                  fixed_generation_length=not args.allow_eos)
    save_json(config, args.out / "config.json")
    summary = []
    for length in args.lengths:
        document_started = time.perf_counter()
        tokens = (source*((length+len(source)-1)//len(source)))[:length]
        document_token_prepare_s = time.perf_counter()-document_started
        questions = make_questions(tokens, tokenizer, max(args.query_counts), args.chunk_size)
        save_json({"n_tokens": length, "source_repeated": length > len(source), "questions": questions},
                  args.out / f"workload_{length}.json")
        for arm in args.arms:
            reader = ReusableReader(model, args.j, tokenizer, arm, args.model, adapter_info)
            store_path = args.out / f"store_{length}_{arm}"
            write = reader.write_store(tokens, store_path, args.chunk_size)
            save_json(write, args.out / f"write_{length}_{arm}.json")
            for tier in args.tiers:
                for generation in args.generation_lengths:
                    startup = reader.open_store(store_path, tier)
                    rows = []
                    result_path = args.out / f"queries_{length}_{arm}_{tier}_g{generation}.jsonl"
                    with result_path.open("w", encoding="utf-8") as stream:
                        for question in questions:
                            begin = time.perf_counter()
                            q_ids = tokenizer.encode(question["text"], add_special_tokens=False)
                            bare = tokenizer.encode(question["bare_question"], add_special_tokens=False)
                            tokenize_s = time.perf_counter()-begin
                            generated, stats = reader.query_ids(q_ids, bare_question_ids=bare, topk=args.topk,
                                max_new_tokens=generation, force_length=not args.allow_eos)
                            stats.update(id=question["id"], generated_ids=generated, tokenization_s=tokenize_s,
                                         hardware=hardware)
                            stats["ttft_s"] += tokenize_s
                            stats["total_s"] += tokenize_s
                            stream.write(json.dumps(stats)+"\n")
                            stream.flush()
                            rows.append(stats)
                    for q in sorted(set(args.query_counts)):
                        selected = rows[:q]
                        totals = {key: sum(r[key] for r in selected) for key in (
                            "retrieval_s", "load_s", "load_bytes", "transfer_s", "transfer_bytes",
                            "rotate_prepare_s", "read_prefill_s", "ttft_s", "decode_s", "total_s",
                            "generated_tokens", "decode_steps", "tokenization_s")}
                        summary.append({"arm": arm, "context_tokens": length, "tier": tier,
                            "hardware": hardware, "adapter": adapter_info,
                            "Q": q, "G": generation, "fixed_generation_length": not args.allow_eos,
                            "source_repeated": length > len(source), "write": write, "startup": startup,
                            "input_boundary": "pretokenized document and raw query texts; output token IDs",
                            "source_preprocessing_charged": False,
                            "shared_source_preprocess_s": shared_source_preprocess_s,
                            "document_token_prepare_s": document_token_prepare_s,
                            "query_totals": totals, "end_to_end_total_s": write["write_total_s"]+startup["startup_load_s"]+totals["total_s"],
                            "incremental_peak_bytes": max(r["incremental_peak_bytes"] for r in selected),
                            "peak_allocated_bytes": max(write["write_peak_allocated_bytes"],
                                                        max(r["peak_allocated_bytes"] for r in selected)),
                            "mean_ttft_s": totals["ttft_s"]/q,
                            "decode_tokens_per_s": totals["decode_steps"]/totals["decode_s"] if totals["decode_s"] else None})
                    save_json(summary, args.out / "summary.json")
                    reader.close_store()
                    gc.collect()
            print(f"Completed {arm}, {length} document tokens", flush=True)
    save_json({"status": "complete", "cells": len(summary), "hardware": hardware,
               "adapter": adapter_info}, args.out / "COMPLETED.json")
    if not args.cpu_test_only:
        from gpu_gate import _release_lock
        _release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
