"""Tiny float32 CPU semantic tests; CUDA orchestration is mocked, never used.

The preparation/evaluation module has separate tests. Main-loop tests inject
prepared examples and an evaluator to exercise budgets and crash recovery with
real tiny-Qwen/Encbank forward, backward, optimizer, and checkpoint tensors.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

import train_sft as sft
from train_8b_baseline import flat_state

torch.set_num_threads(2)
if torch.get_num_interop_threads() != 16:
    torch.set_num_interop_threads(16)


def backbone():
    torch.manual_seed(73)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).cpu().eval()


def fixture(mode="beacon", checkpointing=True):
    model = backbone()
    modules = sft.attach_lora(model, 2, 4, 4, torch.float32)
    # A trained, nonzero adapter stand-in exercises both A and B gradients.
    with torch.no_grad():
        for module in modules.values():
            module.B.normal_(std=.01)
    net = sft.SFTModel(model, mode=mode, split=2,
        compression_ratio=1 if mode == "encbank" else 4, sink_token_id=1,
        writer_id="tiny-shared-initial-adapter", gradient_checkpointing=checkpointing)
    return net.eval(), modules


def examples():
    return [SimpleNamespace(id="example-a", document_id="train-a", document_chunks=((3, 4, 5, 6, 7, 8, 9, 10), (11, 12, 13, 14, 15, 16, 17)),
                            prompt_ids=(21, 22, 23), answer_ids=(31, 32, 2)),
            SimpleNamespace(id="example-b", document_id="train-b", document_chunks=((4, 5, 6, 7, 8, 9, 10, 11),),
                            prompt_ids=(24, 25), answer_ids=(2,))]


def evaluation_receipt(rows):
    return {"summary": {"examples": len(rows), "answer_ce": 1.},
            "records": [{"id": row.id, "answer_ce": 1.} for row in rows]}


def step(net, optimizer, cursor, index, total=3, accum=2):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = .003 * sft.lr_multiplier(index, total, 1)
    raw, target = 0, 0
    for _ in range(accum):
        ex = sft.example_at(examples(), cursor, 42)
        (net.answer_loss(ex.document_chunks, ex.prompt_ids, ex.answer_ids)/accum).backward()
        raw += sum(map(len, ex.document_chunks)) + len(ex.prompt_ids) + len(ex.answer_ids)
        target += len(ex.answer_ids)
        cursor += 1
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.)
    optimizer.step()
    net.writer_id = f"tiny-step{index+1}"
    return cursor, raw, target


class SFTSemantics(unittest.TestCase):
    def test_answer_ce_equals_independent_causal_next_token_predictions(self):
        for mode in ("beacon", "encbank", "pool"):
            for answer in ([31], [31, 32, 2]):
                with self.subTest(mode=mode, answer=answer):
                    net, _ = fixture(mode)
                    chunks, prompt = [[3, 4, 5, 6, 7, 8, 9]], [21, 22, 23]
                    loss = net.answer_loss(chunks, prompt, answer)
                    memories = [net.write_chunk(chunk) for chunk in chunks]
                    # Independent prefixes do not contain current/future targets.
                    terms = [F.cross_entropy(net.read_logits(prompt+answer[:i], memories)[0, -1:].float(),
                                             torch.tensor([target])) for i, target in enumerate(answer)]
                    torch.testing.assert_close(loss, torch.stack(terms).mean(), atol=2e-6, rtol=1e-5)
                    with self.assertRaises(ValueError):
                        net.answer_loss(chunks, [], answer)

    def test_detached_controls_checkpoint_upper_layers_and_match_gradients(self):
        for mode in ("encbank", "pool"):
            ordinary, _ = fixture(mode, False)
            checked, _ = fixture(mode, True)
            real_checkpoint = torch.utils.checkpoint.checkpoint
            with mock.patch.object(torch.utils.checkpoint, "checkpoint", wraps=real_checkpoint) as called:
                loss = checked.answer_loss([[3, 4, 5, 6, 7]], [21, 22], [31, 2])
                loss.backward()
                self.assertEqual(called.call_count, 2)  # both upper layers; no lower backward
            reference = ordinary.answer_loss([[3, 4, 5, 6, 7]], [21, 22], [31, 2])
            reference.backward()
            torch.testing.assert_close(loss, reference)
            for (name, p), (other, q) in zip(ordinary.named_parameters(), checked.named_parameters()):
                self.assertEqual(name, other)
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-5)
            self.assertIsNone(checked.beacon_embedding.grad)

    def test_gradient_and_optimizer_touch_only_intended_trainables(self):
        for mode in ("beacon", "encbank", "pool"):
            with self.subTest(mode=mode):
                net, _ = fixture(mode)
                frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
                before = net.beacon_embedding.detach().clone()
                optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.01)
                net.answer_loss([[3, 4, 5, 6, 7, 8]], [21, 22, 23], [31, 32, 2]).backward()
                reader = [p for n, p in net.named_parameters() if p.requires_grad and n != "beacon_embedding"]
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader))
                self.assertGreater(sum(float(p.grad.norm()) for p in reader), 0)
                if mode == "beacon":
                    self.assertGreater(float(net.beacon_embedding.grad.norm()), 0)
                self.assertTrue(all(p.grad is None for p in net.parameters() if not p.requires_grad))
                optimizer.step()
                self.assertEqual(torch.equal(before, net.beacon_embedding), mode != "beacon")
                current = dict(net.named_parameters())
                for name, value in frozen.items():
                    torch.testing.assert_close(current[name], value, atol=0, rtol=0)

    def test_original_adapter_restore_and_trainable_checkpoint_exclude_frozen_parts(self):
        net, modules = fixture("encbank")
        expected = flat_state(modules)
        model = backbone()
        restored_modules = sft.attach_lora(model, 2, 4, 4, torch.float32)
        sft.restore_flat(restored_modules, expected)
        for key, tensor in flat_state(restored_modules).items():
            torch.testing.assert_close(tensor, expected[key], atol=0, rtol=0)
        saved = sft.trainable_state(net)
        self.assertNotIn("beacon_embedding", saved)
        self.assertTrue(all(name.endswith((".A", ".B")) for name in saved))
        self.assertTrue(all("layers.2." in name or "layers.3." in name for name in saved))
        self.assertEqual(sum(t.numel() for t in saved.values()), sum(t.numel() for t in expected.values()))
        with self.assertRaises(ValueError):
            sft.restore_trainable(net, {**saved, "beacon_embedding": net.beacon_embedding.detach()})
        corrupted = copy.deepcopy(saved)
        corrupted[next(iter(corrupted))].fill_(float("nan"))
        with self.assertRaises(ValueError):
            sft.restore_trainable(net, corrupted)

    def test_optimizer_cursor_rng_resume_equals_uninterrupted_and_eval_is_readonly(self):
        for mode in ("beacon", "encbank", "pool"):
            with self.subTest(mode=mode):
                full, _ = fixture(mode)
                opt = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=.003,
                                       betas=(.9, .95), weight_decay=0, foreach=False)
                cursor = raw = targets = 0
                for index in range(3):
                    cursor, r, t = step(full, opt, cursor, index)
                    raw += r; targets += t
                paused, _ = fixture(mode)
                opt2 = torch.optim.AdamW([p for p in paused.parameters() if p.requires_grad], lr=.003,
                                        betas=(.9, .95), weight_decay=0, foreach=False)
                pcursor, praw, ptargets = step(paused, opt2, 0, 0)
                saved = {"trainable": sft.trainable_state(paused), "optimizer": copy.deepcopy(opt2.state_dict()),
                         "rng": torch.get_rng_state().clone()}
                before = sft.trainable_state(paused)
                with torch.no_grad():
                    ex = examples()[0]
                    memory = [paused.encode_chunk(c) for c in ex.document_chunks]
                    paused.read_logits(ex.prompt_ids, memory)
                    paused.answer_loss(ex.document_chunks, ex.prompt_ids, ex.answer_ids)
                self.assertTrue(torch.equal(saved["rng"], torch.get_rng_state()))
                for name, value in sft.trainable_state(paused).items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                resumed, _ = fixture(mode)
                sft.restore_trainable(resumed, saved["trainable"])
                opt3 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.003,
                                        betas=(.9, .95), weight_decay=0, foreach=False)
                opt3.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["rng"])
                for index in (1, 2):
                    pcursor, r, t = step(resumed, opt3, pcursor, index)
                    praw += r; ptargets += t
                self.assertEqual((cursor, raw, targets), (pcursor, praw, ptargets))
                self.assertEqual((cursor, raw, targets), (6, 96, 12))
                for name, value in sft.trainable_state(full).items():
                    torch.testing.assert_close(value, sft.trainable_state(resumed)[name], atol=0, rtol=0)

    def test_document_split_and_deterministic_equal_example_budget(self):
        train = [{"document_id": "a"}, {"document_id": "b"}]
        sft.validate_split(train, [{"document_id": "c"}])
        with self.assertRaises(ValueError):
            sft.validate_split(train, [{"document_id": "a"}])
        with self.assertRaises(ValueError):
            sft.validate_split([], [{"document_id": "a"}])
        values = examples()
        self.assertEqual({sft.example_at(values, i, 42).document_id for i in range(2)}, {"train-a", "train-b"})
        self.assertEqual([sft.example_at(values, i, 42).document_id for i in range(6)],
                         [sft.example_at(values, i, 42).document_id for i in range(6)])


class MainLoopCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tiny-sft-loop-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train, self.dev, self.adapter = [self.root/name for name in ("train.jsonl", "dev.jsonl", "init.pt")]
        self.train.write_text('{"document_id":"train-a"}\n{"document_id":"train-b"}\n')
        self.dev.write_text('{"document_id":"dev-only"}\n')
        _, modules = fixture("encbank")
        torch.save({"j": 2, "rank": 4, "alpha": 4, "step": 4000,
                    "named": flat_state(modules), "metadata": {"args": {"model": "synthetic"}}}, self.adapter)

    def run_main(self, out, evaluator, extra=()):
        argv = ["train_sft.py", "--model", "synthetic", "--init-adapter", str(self.adapter),
            "--train", str(self.train), "--dev", str(self.dev), "--out", str(out),
            "--device", "cuda:0", "--mode", "encbank", "--ratio", "1", "--j", "2", "--rank", "4",
            "--alpha", "4", "--steps", "2", "--grad-accum", "2", "--warmup", "1",
            "--save-every", "1", "--eval-every", "2", "--eval-limit", "1", "--final-eval-limit", "1", *extra]
        tiny = backbone()
        tiny.to = lambda *a, **kw: tiny  # only the orchestration device move is mocked
        prepared = examples()
        heldout = copy.copy(prepared[0]); heldout.document_id = "dev-only"
        fake_eval = SimpleNamespace(prepare_examples=lambda rows, tok, **kw: prepared if rows[0]["document_id"].startswith("train") else [heldout],
                                    evaluate_examples=evaluator)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "CPU_TEST_MOCK"}))
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(mock.patch.dict(sys.modules, {"evaluate_sft": fake_eval}))
            stack.enter_context(mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(bos_token_id=1, eos_token_id=2)))
            stack.enter_context(mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=tiny))
            stack.enter_context(mock.patch.object(torch, "autocast", side_effect=lambda *a, **kw: nullcontext()))
            for name in ("set_device", "synchronize", "set_rng_state", "manual_seed_all"):
                stack.enter_context(mock.patch.object(torch.cuda, name, return_value=None))
            stack.enter_context(mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True))
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_name", return_value="CPU mocked GPU; not hardware evidence"))
            stack.enter_context(mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state", side_effect=lambda *a, **kw: torch.get_rng_state()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            sft.main()
        self.assertFalse(torch.cuda.is_initialized())

    def test_final_evaluation_failure_resume_completes_without_more_training(self):
        out = self.root/"resume"
        evaluations = []
        def failing(net, tokenizer, rows, **kwargs):
            evaluations.append(net.writer_id)
            if net.writer_id.endswith("step2"):
                raise RuntimeError("Injected final evaluation interruption")
            return evaluation_receipt(rows)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            self.run_main(out, failing)
        state = torch.load(out/"last.pt", map_location="cpu", weights_only=False)
        self.assertEqual((state["step"], state["cursor"], state["raw_tokens"], state["target_tokens"]), (2, 4, 64, 8))
        self.assertIsNotNone(state["last_loss"])
        self.assertGreater(state["gradient_check"]["reader_norm"], 0)
        self.assertFalse(state["gradient_check"]["frozen_base_has_grad"])
        self.assertFalse(json.loads((out/"status.json").read_text())["complete"])
        before_log = (out/"train.jsonl").read_bytes()
        resumed = []
        def success(net, tokenizer, rows, **kwargs):
            resumed.append(net.writer_id)
            return evaluation_receipt(rows)
        self.run_main(out, success, ["--resume", str(out/"last.pt")])
        self.assertEqual(resumed, ["resume:step2"])
        self.assertEqual(before_log, (out/"train.jsonl").read_bytes())
        self.assertTrue((out/"eval_step2.json").exists())
        self.assertTrue(json.loads((out/"status.json").read_text())["complete"])
        restored = torch.load(out/"last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(restored["last_loss"], state["last_loss"])
        self.assertEqual(restored["gradient_check"], state["gradient_check"])
        complete_status = json.loads((out/"status.json").read_text())
        self.assertEqual(complete_status["loss"], state["last_loss"])
        self.assertEqual(complete_status["gradient_check"], state["gradient_check"])
        for name, value in state["trainable"].items():
            torch.testing.assert_close(value, restored["trainable"][name], atol=0, rtol=0)
        # Existing final evaluation must match the actual held-out IDs and CE.
        corrupted = json.loads((out/"eval_step2.json").read_text())
        corrupted["records"][0]["id"] = "unrelated-heldout-example"
        (out/"eval_step2.json").write_text(json.dumps(corrupted))
        with self.assertRaisesRegex(ValueError, "Existing evaluation"):
            self.run_main(out, success, ["--resume", str(out/"last.pt")])
        self.assertEqual(before_log, (out/"train.jsonl").read_bytes())

    def test_initial_evaluation_not_in_training_budget_and_resume_matches(self):
        seen = []
        def evaluate(net, tokenizer, rows, **kwargs):
            self.assertEqual({r.document_id for r in rows}, {"dev-only"})
            self.assertFalse(torch.is_grad_enabled())
            seen.append(net.writer_id)
            return evaluation_receipt(rows)
        full, resumed = self.root/"full", self.root/"resumed"
        self.run_main(full, evaluate)
        self.run_main(resumed, evaluate, ["--stop-after", "1"])
        pause = json.loads((resumed/"status.json").read_text())
        self.assertEqual((pause["step"], pause["training_examples_processed"], pause["raw_tokens"], pause["target_tokens"]), (1, 2, 32, 4))
        self.run_main(resumed, evaluate, ["--resume", str(resumed/"last.pt")])
        a, b = [torch.load(p/"last.pt", map_location="cpu", weights_only=False) for p in (full, resumed)]
        for name, value in a["trainable"].items():
            torch.testing.assert_close(value, b["trainable"][name], atol=0, rtol=0)
        self.assertEqual(seen.count("resumed:step0"), 1)
        self.assertEqual((b["step"], b["cursor"], b["raw_tokens"], b["target_tokens"]), (2, 4, 64, 8))
        self.assertNotIn("beacon_embedding", b["trainable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
