"""CPU-only guards for device eligibility, final adapters and resumable queue shape."""
import argparse
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import serving_reuse as serving
import serving_local_bootstrap as bootstrap
from test_serving_reuse_cpu import model_and_tokenizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class LocalServingTests(unittest.TestCase):
    def test_actual_gpu_and_cpu_labels(self):
        self.assertFalse(serving.device_provenance("cpu", allow_cpu=True)["timing_eligible"])
        with self.assertRaises(ValueError):
            serving.device_provenance("cpu")
        props = SimpleNamespace(name=serving.REQUIRED_GPU, total_memory=32*1024**3, major=12, minor=0)
        with patch.object(serving.platform, "system", return_value="Windows"), \
                patch.object(torch.cuda, "get_device_properties", return_value=props):
            self.assertTrue(serving.device_provenance("cuda:0")["timing_eligible"])
            props.name = "NVIDIA GeForce RTX 3090"
            with self.assertRaisesRegex(ValueError, "3090"):
                serving.device_provenance("cuda:0")
        with patch.object(serving.platform, "system", return_value="Linux"):
            with self.assertRaises(ValueError):
                serving.device_provenance("cuda:0")

    def test_only_synchronized_final_4000_step_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "final"
            final.mkdir()
            (final / "adapter.pt").write_bytes(b"trusted sync already checked tensor metadata")
            (final / "adapter_model.safetensors").write_bytes(b"trusted sync already checked tensors")
            write(final / "adapter_config.json", {"r": 32, "lora_alpha": 32})
            write(root / "metadata.json", {"recipe": {"steps": 4000, "j": 12, "rank": 32,
                  "alpha": 32, "loss": "published", "adapter_dtype": "float32"}})
            status = {"step": 4000, "target_steps": 4000, "complete": True}
            write(root / "status.json", status)
            marker = {"step": 4000, "complete": True, "adapter_load_mode": "peft_unmerged",
                      "adapter_dir": str(final.resolve()),
                      "files": {"final/" + p.name: p.stat().st_size for p in final.iterdir()}}
            write(root / "TRANSFER_COMPLETE.json", marker)
            self.assertEqual(serving.validate_adapter_checkpoint(final)["step"], 4000)
            for bad in ({**status, "step": 2000}, {**status, "complete": False}):
                write(root / "status.json", bad)
                with self.assertRaises(ValueError):
                    serving.validate_adapter_checkpoint(final)
            write(root / "status.json", status)
            (final / "adapter_model.safetensors").write_bytes(b"truncated")
            with self.assertRaises(ValueError):
                serving.validate_adapter_checkpoint(final)

    def test_unmerged_fp32_adapter_is_isolated_to_pub_lora(self):
        from peft import get_peft_model, LoraConfig
        model, tok = model_and_tokenizer()
        with patch("peft.tuners.lora.model.dispatch_torchao", return_value=None):
            wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"],
                                                      layers_to_transform=[2], task_type="CAUSAL_LM"))
        with tempfile.TemporaryDirectory() as tmp:
            wrapped.save_pretrained(tmp)
            base, _ = model_and_tokenizer()
            model = serving.load_unmerged_adapter(base, tmp)
            self.assertTrue(all(p.dtype == torch.float32 for name, p in model.named_parameters() if "lora_" in name))
            for arm in serving.ARMS:
                with self.assertRaisesRegex(ValueError, "Only the pub_lora"):
                    serving.ReusableReader(model, 2, tok, arm)
            reader = serving.ReusableReader(model, 2, tok, "pub_lora")
            self.assertFalse(reader.cm.write_sink)
            reader.write_store(list(range(3, 19)), Path(tmp)/"cache", 8)
            reader.open_store(Path(tmp)/"cache", "cpu")
            generated, stats = reader.query_ids([21, 22], selected_indices=[0, 1], max_new_tokens=3, force_length=True)
            self.assertEqual(len(generated), 3)
            self.assertEqual(stats["capture_calls"], 0)

    def test_queue_protocol_and_resume_validate_hardware_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = bootstrap.jobs_for("base", ["smoke", "full"], root)
            self.assertEqual(sum(j["cells"] for j in jobs if j["phase"] == "smoke"), 16)
            self.assertEqual(sum(j["cells"] for j in jobs if j["phase"] == "full"), 96)
            self.assertEqual(jobs[0]["length"], 1024)
            self.assertEqual(jobs[0]["query_counts"], [1, 2])
            self.assertEqual(jobs[0]["generation_lengths"], [4])
            args = argparse.Namespace(model=root/"model", context_file=root/"context",
                                      adapter_ckpt=root/"final")
            job = jobs[0]
            attempt = job["folder"] / "attempts/0001"
            hardware = {"timing_eligible": True, "device_name": bootstrap.REQUIRED_GPU,
                        "platform": "Windows", "hostname": socket.gethostname(), **bootstrap.THREAD_HARDWARE}
            marker = {"status": "complete", "cells": job["cells"], "hardware": hardware}
            write(attempt / "COMPLETED.json", marker)
            config = bootstrap.expected_config(args, job)
            write(attempt / "config.json", config)
            rows = []
            for tier in job["tiers"]:
                for g in job["generation_lengths"]:
                    (attempt / f'queries_{job["length"]}_{job["arm"]}_{tier}_g{g}.jsonl').write_text('{}\n'*2)
                    for q in job["query_counts"]:
                        rows.append({"arm": job["arm"], "context_tokens": job["length"], "tier": tier,
                            "Q": q, "G": g, "hardware": hardware, "fixed_generation_length": True,
                            "query_totals": {"generated_tokens": q*g}})
            write(attempt / "summary.json", rows)
            self.assertEqual(bootstrap.completed_attempt(args, job), attempt)
            write(attempt / "COMPLETED.json", {**marker, "hardware": {**hardware, "torch_cpu_threads": 16}})
            self.assertIsNone(bootstrap.completed_attempt(args, job))
            write(attempt / "COMPLETED.json", marker)
            write(attempt / "config.json", {**config, "topk": 4})
            self.assertIsNone(bootstrap.completed_attempt(args, job))
            write(attempt / "config.json", config)
            write(attempt / "COMPLETED.json", {**marker, "hardware": {**hardware, "device_name": "RTX 3090"}})
            self.assertIsNone(bootstrap.completed_attempt(args, job))

    def test_child_threads_override_parent_environment(self):
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "8",
                                      "TOKENIZERS_PARALLELISM": "true"}):
            env = bootstrap.child_environment()
        for key, value in bootstrap.THREAD_ENV.items():
            self.assertEqual(env[key], value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
