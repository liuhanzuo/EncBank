"""CPU fault injection for full-conversation checkpoints and final generation."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from official_sft import PreparedAssistantTurn, PreparedConversation
from train_official_sft import (PreparedDataset, decode_prepared, validate_splits,
    select_ids, sample_schedule, budget_at, make_checkpoint, restore_checkpoint,
    reconcile_train_log, evaluation_signature, evaluate_checkpoint,
    validate_evaluation, fingerprint_tensors, greedy_record, rouge_l,
    validate_tokenizer_fingerprint)


def row(cid="c1", doc="d1", split="train", source="activation_beacon/gpt_book"):
    turns = (
        PreparedAssistantTurn(cid+":t0", 0, 1, (10,), "one", ("one",), (0, 1), (11, 2), 2, .5),
        PreparedAssistantTurn(cid+":t1", 1, 3, (10, 11, 2, 10), "two", ("two",), (3, 4), (12, 2), 2, .5))
    protocol = {"format": "official-full-conversation-qwen3-comem-v1", "selector": "all_original_chunks_in_order",
        "truncated": False, "chunk_size": 512, "official_pretraining_LM_branch": "not included",
        "min_original_template_tokens": 0, "max_original_template_tokens": 1000}
    ex = PreparedConversation(cid, cid, doc, source, "raw.json", 0, split, "strict-test", {}, doc,
        ((7, 8),), (10, 11, 2, 10, 12), (11, 2, -100, 12, 2), (0, 1, 3, 4), turns,
        2, 9, 6, 8, 4, 2, protocol)
    return {**ex.to_dict(), "normalized_group_id": "norm:"+doc, "overlap_group_id": "overlap:"+doc,
            "tokenizer_fingerprint": "test-tokenizer", "future_extra_field": {"kept": True}}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


class OfficialTrainerTest(unittest.TestCase):
    def test_prepared_roundtrip_and_exact_shift_rejection(self):
        decoded = decode_prepared(row())
        self.assertEqual(decoded.extra_metadata["future_extra_field"], {"kept": True})
        self.assertEqual(decoded.target_token_count, 4)
        changed = row()
        changed["label_ids"] = [11, 7, -100, 12, 2]
        changed["turns"][0]["training_target_ids"] = [11, 7]
        with self.assertRaisesRegex(ValueError, "shifted exactly once"):
            decode_prepared(changed)

    def test_long_summary_retains_all_targets_and_native_end(self):
        value = row(source="activation_beacon/booksum")
        value.update(query_ids=[10]+[11]*300, label_ids=[11]*300+[2],
            target_indices=list(range(301)), document_token_count=2,
            original_template_token_count=310, reader_template_token_count=302,
            raw_uncompressed_read_pack_tokens=304, target_token_count=301, assistant_turn_count=1)
        value["turns"] = [{"id": "c1:t0", "assistant_turn_index": 0, "message_index": 1,
            "prompt_ids": [10], "answer": "complete summary "*300, "references": ["complete summary "*300],
            "target_indices": list(range(301)), "training_target_ids": [11]*300+[2],
            "target_token_count": 301, "conversation_token_weight": 1.}]
        ex = decode_prepared(value)
        self.assertEqual(ex.target_token_count, 301)
        self.assertEqual(ex.turns[0].training_target_ids[-1], 2)
        self.assertEqual(ex.turns[0].answer, value["turns"][0]["answer"])
        self.assertEqual(len(ex.target_indices), len(ex.labels))

    def test_dataset_group_split_and_matched_schedule_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root/"train.jsonl", [row(), row("c2", "d2")])
            write_rows(root/"dev.jsonl", [row("v1", "v1", "dev")])
            train, dev = PreparedDataset(root/"train.jsonl", "train"), PreparedDataset(root/"dev.jsonl", "dev")
            validate_splits(train, dev)
            ids, _, _ = select_ids(train, dev)
            schedule = sample_schedule(ids, 6, 42, "seeded_shuffle")
            self.assertEqual(schedule, sample_schedule(ids, 6, 42, "seeded_shuffle"))
            budget = budget_at(train, schedule, 6)
            self.assertEqual((budget["target_tokens"], budget["conversations"], budget["assistant_turns"]), (24, 6, 12))
            bad = row("v2", "v2", "dev"); bad["overlap_group_id"] = "overlap:d1"
            write_rows(root/"bad.jsonl", [bad])
            with self.assertRaisesRegex(ValueError, "overlap_group_id overlap"):
                validate_splits(train, PreparedDataset(root/"bad.jsonl", "dev"))
            dev.index["v1"].pop("overlap_group_id")
            with self.assertRaisesRegex(ValueError, "complete overlap_group_id"):
                validate_splits(train, dev)

    def test_checkpoint_optimizer_all_rng_resume_matches_uninterrupted(self):
        torch.manual_seed(13); random.seed(13); np.random.seed(13)
        net = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=.01)
        metadata = {"recipe": {"steps": 4, "data_sha": "same"}, "model_signature": {"config": "tiny"}}
        def update(model, optimizer):
            x = torch.randn(4, 3) + random.random() + float(np.random.rand())
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).square().mean(); loss.backward(); optimizer.step()
        update(net, opt); update(net, opt)
        saved = make_checkpoint(net, opt, {"step": 2, "cursor": 4}, metadata)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"resume.pt"
            torch.save(saved, path)
            update(net, opt); update(net, opt)
            expected = fingerprint_tensors(dict(net.named_parameters()))
            resumed = torch.nn.Linear(3, 2)
            resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.01)
            restored = torch.load(path, weights_only=False)
            progress = restore_checkpoint(restored, resumed, resumed_opt, metadata, grad_accum=2)
            self.assertEqual(progress, {"step": 2, "cursor": 4})
            update(resumed, resumed_opt); update(resumed, resumed_opt)
            self.assertEqual(expected, fingerprint_tensors(dict(resumed.named_parameters())))
            other = deepcopy(metadata); other["recipe"]["data_sha"] = "changed"
            with self.assertRaisesRegex(ValueError, "recipe/data"):
                restore_checkpoint(restored, resumed, resumed_opt, other, grad_accum=2)
            broken = deepcopy(restored); broken["progress"]["cursor"] = 3
            with self.assertRaisesRegex(ValueError, "inside an optimizer step"):
                restore_checkpoint(broken, resumed, resumed_opt, metadata, grad_accum=2)

    def test_prepared_tokenizer_mismatch_fails_before_model_loading(self):
        from prepare_official_pool import tokenizer_receipt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"tokenizer.json").write_text("{}", encoding="utf-8")
            (root/"tokenizer_config.json").write_text("{}", encoding="utf-8")
            fingerprint = tokenizer_receipt(root)["tokenizer_fingerprint"]
            a, b = row(), row("v", "v", "dev")
            a["tokenizer_fingerprint"] = b["tokenizer_fingerprint"] = fingerprint
            write_rows(root/"train.jsonl", [a]); write_rows(root/"dev.jsonl", [b])
            train, dev = PreparedDataset(root/"train.jsonl", "train"), PreparedDataset(root/"dev.jsonl", "dev")
            validate_tokenizer_fingerprint(train, dev, root)
            (root/"tokenizer_config.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer differs"):
                validate_tokenizer_fingerprint(train, dev, root)

    def test_crash_log_tail_rolls_back_to_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.jsonl"
            path.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":', encoding="utf-8")
            reconcile_train_log(path, 2)
            self.assertEqual([json.loads(s)["step"] for s in path.read_text().splitlines()], [1, 2])
            self.assertEqual(len(list(Path(tmp).glob("*.discarded-*"))), 1)
            with self.assertRaisesRegex(ValueError, "missing/corrupt committed"):
                reconcile_train_log(path, 3)

    def test_final_eval_crash_after_ce_cannot_mark_complete_and_retries_generation(self):
        signature = evaluation_signature(2, "weights", "recipe", ["c1"], ["c1"], 128, 1024)
        calls = {"ce": 0, "gen": 0}
        def ce(cid):
            calls["ce"] += 1
            return {"conversation_id": cid, "source": "books", "target_tokens": 4, "ce_sum": 8., "conversation_ce": 2.}
        def broken(cid):
            calls["gen"] += 1
            raise RuntimeError("injected process interruption during final generation")
        def generated(cid):
            calls["gen"] += 1
            return {"id": cid+":t0", "conversation_id": cid, "source": "books", "assistant_turn_index": 0,
                "references": ["answer"], "generated_ids": [11, 2], "generated_tokens": 2,
                "finish_reason": "eos", "generation_completed": True, "task_type": "qa", "exact_match": 1., "token_f1": 1.}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"eval_step2.json"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                evaluate_checkpoint(path, signature, ce, broken)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name(path.name+".partial.jsonl").exists())
            result = evaluate_checkpoint(path, signature, ce, generated)
            validate_evaluation(result, signature)
            self.assertEqual(calls, {"ce": 2, "gen": 2})
            evaluate_checkpoint(path, signature, ce, broken)
            self.assertEqual(calls, {"ce": 2, "gen": 2})
            fake = deepcopy(result); fake["generation_records"] = []
            with self.assertRaisesRegex(ValueError, "Generation record IDs"):
                validate_evaluation(fake, signature)
            changed = {**signature, "model_state_id": "different"}
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                validate_evaluation(result, changed)
            self.assertIsNone(result["summary"]["quality_aggregate"])

    def test_greedy_uses_only_first_prompt_and_source_cache(self):
        class Tokenizer:
            eos_token_id, unk_token_id = 2, -1
            def convert_tokens_to_ids(self, _): return 2
            def decode(self, ids, **_): return "one" if list(ids) == [11] else "Question: name it"
        class Wrapper:
            def __init__(self): self.written, self.prompt = [], None
            def encode_chunk(self, chunk, **_): self.written.append(tuple(chunk)); return chunk
            def logits(self, token):
                result = torch.zeros(1, 1, 20); result[0, 0, token] = 1; return result
            def start_query(self, prompt, memories):
                self.prompt = tuple(prompt); return SimpleNamespace(logits=self.logits(11))
            def decode_token(self, token, state): state.logits = self.logits(2)
        wrapper = Wrapper(); ex = decode_prepared(row())
        record = greedy_record(wrapper, Tokenizer(), ex, 4, 8)
        self.assertEqual(wrapper.written, [(7, 8)])
        self.assertEqual(wrapper.prompt, (10,))
        self.assertEqual(record["generated_ids"], [11, 2])
        self.assertEqual(record["exact_match"], 1.)
        self.assertNotIn("rouge_l", record)

    def test_rouge_l_is_separate_summary_metric(self):
        self.assertAlmostEqual(rouge_l("a complete summary", "a complete summary"), 1.)
        self.assertEqual(rouge_l("", "a summary"), 0.)
        self.assertEqual(rouge_l("...", "a summary"), 0.)
        self.assertEqual(rouge_l("alpha", "beta "*1200), 0.)


if __name__ == "__main__":
    unittest.main()
