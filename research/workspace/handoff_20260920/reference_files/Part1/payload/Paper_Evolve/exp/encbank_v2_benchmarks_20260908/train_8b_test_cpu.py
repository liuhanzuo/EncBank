"""CPU-only checks for the training entry point; never loads pretrained weights."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import train_8b_baseline as T
from encbank import Encbank
from train.distill import distill_logits_kl


def tiny():
    torch.manual_seed(23)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).float().eval()


def recipe():
    return SimpleNamespace(chunk=8, n_ctx=2, topk=16, lam=0.6,
                           logit_chunk=3, loss="published", model="tiny", rank=4,
                           alpha=4, j=2)


class TrainerCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_matches_published_trainer_value_and_gradient(self):
        torch.manual_seed(3)
        val, idx = torch.randn(3, 31).topk(9, dim=-1)
        a = torch.randn(3, 31, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        loss = T.support_loss(a, idx, val, 0.6, "published")
        ref = distill_logits_kl(b, idx, val, lam=0.6)
        torch.testing.assert_close(loss, ref)
        loss.backward(); ref.backward()
        torch.testing.assert_close(a.grad, b.grad)
        self.assertGreater(abs(float(T.support_loss(a.detach(), idx, val, .6, "legacy-s21") - loss.detach())), 1e-4)

    def test_teacher_and_published_student_match_original_encbank(self):
        model = tiny()
        cfg = recipe()
        reader = T.PublishedReader(model, cfg.j)
        window = torch.arange(24) + 3
        for teacher in (False, True):
            with torch.no_grad():
                got = model.lm_head(reader.hidden(window, 1, cfg.chunk, cfg.n_ctx, teacher))
                cm = Encbank(model, resume_j=0 if teacher else cfg.j)
                chunks = window.split(cfg.chunk)
                expected = cm.read_core(cm.write_chunk([1]),
                                        [cm.write_chunk(x) for x in chunks[:-1]],
                                        cm.write_chunk(chunks[-1]), logits_tail=cfg.chunk)
                torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
                if teacher:
                    full = model(torch.cat([torch.tensor([1]), window]).unsqueeze(0)).logits[:, -cfg.chunk:]
                    torch.testing.assert_close(got, full, atol=2e-6, rtol=2e-5)

    def test_checkpoint_loss_gradients_and_frozen_backbone(self):
        model = tiny()
        cfg = recipe()
        modules = T.attach_lora(model, cfg.j, cfg.rank, cfg.alpha, torch.float32)
        with torch.no_grad():
            for mod in modules.values():
                mod.B.normal_(std=.01)
        a = copy.deepcopy(model)
        b = copy.deepcopy(model)
        ma = {k: dict(a.named_modules())[self.module_path(k)] for k in modules}
        mb = {k: dict(b.named_modules())[self.module_path(k)] for k in modules}
        window = torch.arange(24) + 3
        loss_a = T.batch_loss(T.PublishedReader(a, cfg.j, True), ma, window, 1, cfg)
        loss_b = T.batch_loss(T.PublishedReader(b, cfg.j, False), mb, window, 1, cfg)
        loss_a.backward(); loss_b.backward()
        torch.testing.assert_close(loss_a, loss_b)
        for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name_a, name_b)
            if pa.requires_grad:
                self.assertIsNotNone(pa.grad)
                torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=1e-4)
            else:
                self.assertIsNone(pa.grad)
        self.assertGreater(sum(float(m.B.grad.abs().sum()) for m in ma.values()), 0)

    @staticmethod
    def module_path(k):
        _, index, proj = k.split(".")
        parent = "self_attn" if proj in T.TARGETS[:4] else "mlp"
        return f"model.layers.{index}.{parent}.{proj}"

    def test_restart_matches_uninterrupted_and_exports(self):
        cfg = recipe()
        def make():
            model = tiny()
            modules = T.attach_lora(model, cfg.j, cfg.rank, cfg.alpha, torch.float32)
            opt = torch.optim.AdamW([p for m in modules.values() for p in (m.A, m.B)],
                                    lr=1e-3, betas=(.9,.95), weight_decay=0, foreach=False)
            return model, modules, opt, T.TokenStream(np.arange(70) + 3, 24, 48)
        def advance(state, count):
            model, modules, opt, stream = state
            for _ in range(count):
                opt.zero_grad(set_to_none=True)
                T.batch_loss(T.PublishedReader(model, cfg.j), modules, stream.take(), 1, cfg).backward()
                torch.nn.utils.clip_grad_norm_([p for m in modules.values() for p in (m.A,m.B)], 1)
                opt.step()
        uninterrupted = make()
        advance(uninterrupted, 4)
        interrupted = make()
        advance(interrupted, 2)
        with tempfile.TemporaryDirectory(prefix="train_8b_cpu_") as tmp:
            ck = Path(tmp) / "state.pt"
            T.atomic_save(ck, {"named": T.flat_state(interrupted[1]), "optimizer": interrupted[2].state_dict(),
                               "cursor": interrupted[3].cursor, "rng": T.rng_state([])})
            loaded = torch.load(ck, weights_only=False)
            resumed = make()
            T.restore_flat(resumed[1], loaded["named"])
            resumed[2].load_state_dict(loaded["optimizer"])
            resumed[3].cursor = loaded["cursor"]
            T.restore_rng(loaded["rng"], [])
            advance(resumed, 2)
            self.assertEqual(uninterrupted[3].cursor, resumed[3].cursor)
            for key, value in T.flat_state(uninterrupted[1]).items():
                torch.testing.assert_close(value, T.flat_state(resumed[1])[key], atol=0, rtol=0)
            payload = {"named": T.flat_state(resumed[1]), "rank": cfg.rank, "alpha": cfg.alpha,
                       "j": cfg.j, "path": "base", "targets": list(T.TARGETS)}
            T.export_adapter(Path(tmp) / "final", payload, cfg, 4)
            from safetensors.torch import load_file
            exported = load_file(str(Path(tmp) / "final" / "adapter_model.safetensors"))
            self.assertEqual(set(exported), set(T.peft_state(payload["named"])))
            self.assertEqual(json.loads((Path(tmp) / "final" / "adapter_config.json").read_text())["r"], 4)
            from peft import PeftModel
            # Local torchao 0.13 conflicts with peft 0.20; ordinary fp32 LoRA does
            # not use torchao. Bypass only that optional dispatcher in this test.
            # The trainer itself never imports PEFT or changes its environment.
            with patch("peft.tuners.lora.torchao.is_torchao_available", return_value=False):
                wrapped = PeftModel.from_pretrained(tiny(), Path(tmp) / "final").eval()
            ids = torch.arange(12).unsqueeze(0) + 3
            with torch.no_grad():
                torch.testing.assert_close(wrapped(ids).logits, resumed[0](ids).logits,
                                           atol=2e-6, rtol=2e-5)

    def test_explicit_layer_distribution_and_cyclic_data(self):
        model = tiny()
        placement = T.distribute(model, ["cpu", "cpu"], [2])
        self.assertEqual(placement["layer_boundaries"], [0, 2, 4])
        with self.assertRaises(ValueError):
            T.distribute(model, ["cpu", "cpu"], [4])
        stream = T.TokenStream(np.arange(5), 4, 3)
        self.assertEqual(stream.take().tolist(), [3,4,0,1])
        self.assertEqual(stream.take().tolist(), [2,3,4,0])
        self.assertEqual(stream.cursor, 11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
