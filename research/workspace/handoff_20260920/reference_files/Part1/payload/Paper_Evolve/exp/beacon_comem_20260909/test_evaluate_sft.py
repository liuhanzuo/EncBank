"""CPU checks for data leakage, answer alignment, reuse, and natural stopping."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn.functional as F

from evaluate_sft import (
    PreparedExample, evaluate_examples, exact_match, prepare_examples, token_f1,
)


class ToyTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    def __init__(self):
        self.words = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<|im_end|>": 3}
        self.reverse = {value: key for key, value in self.words.items()}
        self.template_calls = []

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = len(self.words) + 10
                self.reverse[self.words[word]] = word
            result.append(self.words[word])
        return result

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.reverse.get(int(value), f"token{value}") for value in ids
                        if not skip_special_tokens or int(value) not in (0, 1, 2, 3, 4))

    def convert_tokens_to_ids(self, token):
        return self.words.get(token, 0)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.template_calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
        return [1] + self.encode(messages[0]["content"]) + [4]


def row(example_id="one", document_id="doc", context="first evidence second evidence target answer", question="target", answer="gold"):
    return {"id": example_id, "document_id": document_id, "context": context,
            "question": question, "answer": answer, "source": "toy", "split": "validation"}


class FakeWrapper:
    """Records every writer/input call; scripted generation is answer-independent."""

    def __init__(self, sequence, expected_ce_targets=()):
        self.sequence = list(sequence)
        self.writer_inputs = []
        self.query_inputs = []
        self.decode_inputs = []
        self.ce_inputs = []
        self.expected_ce_targets = tuple(expected_ce_targets)
        self.training = True
        self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 5]))

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def encode_chunk(self, ids, cache_device="cpu"):
        self.writer_inputs.append(tuple(ids))
        assert cache_device == "cpu"
        return torch.tensor(ids, dtype=torch.float32).view(1, -1, 1)

    @staticmethod
    def logits(token):
        value = torch.full((1, 1, 512), -20.0)
        value[0, 0, token] = 20.0
        return value

    def start_query(self, query_ids, memories):
        self.query_inputs.append(tuple(query_ids))
        return SimpleNamespace(logits=self.logits(self.sequence[0]), step=0)

    def decode_token(self, token, state):
        self.decode_inputs.append(token)
        state.step += 1
        state.logits = self.logits(self.sequence[min(state.step, len(self.sequence) - 1)])
        return state.logits

    def read_logits(self, ids, memories):
        self.ce_inputs.append(tuple(ids))
        value = torch.full((1, len(ids), 512), -20.0)
        for offset, token in enumerate(self.expected_ce_targets):
            value[0, len(ids) - len(self.expected_ce_targets) + offset, token] = 20.0
        return value


class PreparationChecks(unittest.TestCase):
    def test_full_document_question_only_retrieval(self):
        tok = ToyTokenizer()
        rows = [row(context="first old middle old target detail tail end", question="target", answer="old")]
        prepared = prepare_examples(rows, tok, chunk_size=2, max_chunks=1)
        self.assertEqual(prepared[0].selected_chunk_indices, (2,))
        self.assertEqual(prepared[0].original_context_tokens, 8)
        self.assertEqual(tok.decode(prepared[0].document_chunks[0]), "target detail")
        changed_answer = prepare_examples([{**rows[0], "answer": "tail"}], tok, chunk_size=2, max_chunks=1)
        self.assertEqual(prepared[0].document_chunks, changed_answer[0].document_chunks)
        self.assertEqual(prepared[0].prompt_ids, changed_answer[0].prompt_ids)
        self.assertTrue(all(call[3] is False for call in tok.template_calls))
        self.assertEqual(prepared[0].answer_ids[-1], 3)

    def test_selected_chunks_keep_source_order(self):
        tok = ToyTokenizer()
        prepared = prepare_examples([row(context="apple one junk two banana three", question="banana apple")],
                                    tok, chunk_size=2, max_chunks=2)
        self.assertEqual(prepared[0].selected_chunk_indices, (0, 2))

    def test_primary_training_answer_and_scoring_aliases(self):
        tok = ToyTokenizer()
        prepared = prepare_examples([{**row(answer="primary"), "answers": ["alias", "primary", "other", "alias"]}], tok)
        self.assertEqual(prepared[0].references, ("primary", "alias", "other"))
        self.assertEqual(prepared[0].answer_ids, tuple(tok.encode("primary")) + (3,))
        alias = tok.encode("alias")[0]
        result = evaluate_examples(FakeWrapper([alias, 3]), tok, prepared)
        self.assertEqual(result["records"][0]["exact_match"], 1.0)

    def test_explicit_filtering_and_full_references(self):
        tok = ToyTokenizer()
        rows = [row("longq", question="one two three"), row("longa", answer="a b c d e")]
        result = prepare_examples(rows, tok, max_question_tokens=2, max_answer_tokens=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.skipped, [{"id": "longq", "reason": "question_too_long", "question_tokens": 3}])
        self.assertEqual(result[0].references, ("a b c d e",))
        self.assertTrue(result[0].answer_truncated)
        self.assertEqual(tok.decode(result[0].answer_ids), "a b c")
        self.assertNotIn(3, result[0].answer_ids)
        guarded = prepare_examples([row()], tok, max_context_tokens=2)
        self.assertEqual(len(guarded), 0)
        self.assertEqual(guarded.skipped[0]["reason"], "context_too_long")

    def test_fixed_sampling_and_identity_checks(self):
        tok = ToyTokenizer()
        rows = [row(str(i)) for i in range(20)]
        a = prepare_examples(rows, tok, max_examples=4, seed=17)
        b = prepare_examples(list(reversed(rows)), tok, max_examples=4, seed=17)
        self.assertEqual({x.id for x in a}, {x.id for x in b})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_examples([row(), row()], tok)
        with self.assertRaisesRegex(ValueError, "Conflicting contexts"):
            prepare_examples([row("a"), row("b", context="another context")], tok)


class GenerationChecks(unittest.TestCase):
    def test_generation_ce_alignment_no_leak_and_query_reset(self):
        tok = ToyTokenizer()
        examples = prepare_examples([row("one"), row("two", question="evidence")], tok, chunk_size=2)
        gold = tok.encode("gold")[0]
        wrapper = FakeWrapper([gold, 3, 77], examples[0].answer_ids)
        result = evaluate_examples(wrapper, tok, examples, max_new_tokens=8, answer_ce=True)
        self.assertEqual(result["summary"]["exact_match"], 1.0)
        self.assertEqual(result["summary"]["token_f1"], 1.0)
        self.assertLess(result["summary"]["answer_ce"], 1e-6)
        self.assertEqual(result["summary"]["cache"]["chunk_writes"], 3)
        self.assertEqual(result["summary"]["cache"]["chunk_hits"], 3)
        self.assertEqual(wrapper.query_inputs, [x.prompt_ids for x in examples])
        self.assertEqual(wrapper.ce_inputs, [x.prompt_ids + x.answer_ids[:-1] for x in examples])
        self.assertEqual(wrapper.writer_inputs, list(examples[0].document_chunks))
        self.assertTrue(all(gold not in ids for ids in wrapper.writer_inputs))
        self.assertEqual(wrapper.decode_inputs, [gold, gold])
        self.assertTrue(wrapper.training)
        self.assertTrue(all(x["generated_ids"] == [gold, 3] for x in result["records"]))

    def test_chunk_cache_uses_retrieval_indices_not_only_document(self):
        tok = ToyTokenizer()
        examples = prepare_examples([
            row("a", context="apple detail banana detail", question="apple"),
            row("b", context="apple detail banana detail", question="banana"),
            row("c", context="apple detail banana detail", question="apple"),
        ], tok, chunk_size=2, max_chunks=1)
        wrapper = FakeWrapper([3])
        result = evaluate_examples(wrapper, tok, examples)
        self.assertEqual(wrapper.writer_inputs, [examples[0].document_chunks[0], examples[1].document_chunks[0]])
        self.assertEqual(result["summary"]["cache"]["chunk_writes"], 2)
        self.assertEqual(result["summary"]["cache"]["chunk_hits"], 1)
        bad = replace(examples[2], document_chunks=((999, 998),))
        with self.assertRaisesRegex(ValueError, "Conflicting document chunk"):
            evaluate_examples(wrapper, tok, [examples[0], bad])

    def test_model_eos_cap_and_alias_scoring(self):
        tok = ToyTokenizer()
        examples = prepare_examples([row(answer=["not it", "gold"])], tok)
        gold = tok.encode("gold")[0]
        stopped = evaluate_examples(FakeWrapper([gold, 5, 88]), tok, examples, max_new_tokens=8)
        self.assertEqual(stopped["records"][0]["finish_reason"], "eos")
        self.assertEqual(stopped["records"][0]["generated_ids"], [gold, 5])
        capped = evaluate_examples(FakeWrapper([gold]), tok, examples, max_new_tokens=2)
        self.assertEqual(capped["records"][0]["finish_reason"], "max_new_tokens")
        self.assertEqual(capped["records"][0]["generated_tokens"], 2)
        alias = evaluate_examples(FakeWrapper([gold, 3]), tok, examples)
        self.assertEqual(alias["records"][0]["exact_match"], 1.0)

    def test_per_example_artifact_and_empty_summary(self):
        tok = ToyTokenizer()
        examples = prepare_examples([row()], tok)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "predictions.jsonl"
            result = evaluate_examples(FakeWrapper([3]), tok, examples, output_path=destination)
            self.assertEqual(len(destination.read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue(destination.with_suffix(".summary.json").is_file())
            self.assertEqual(result["summary"]["protocol"]["hardware_timing"], "not-collected")
        empty = evaluate_examples(FakeWrapper([3]), tok, [])
        self.assertIsNone(empty["summary"]["exact_match"])

    def test_normalization(self):
        self.assertEqual(exact_match(" The, ANSWER! ", "answer"), 1.0)
        self.assertAlmostEqual(token_f1("red red blue", "red blue"), 0.8)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("red", ""), 0.0)

    def test_tiny_real_beacon_generation_and_manual_answer_ce(self):
        from test_beacon_comem import fixture

        tok = ToyTokenizer()
        examples = prepare_examples([row("a"), row("b", question="other detail")], tok, chunk_size=3)
        net = fixture()
        result = evaluate_examples(net, tok, examples, max_new_tokens=3, answer_ce=True)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["summary"]["cache"]["chunk_writes"], 2)
        self.assertEqual(result["summary"]["cache"]["chunk_hits"], 2)
        net.eval()
        example = examples[0]
        with torch.no_grad():
            memories = [net.encode_chunk(chunk) for chunk in example.document_chunks]
            logits = net.read_logits(example.prompt_ids + example.answer_ids[:-1], memories)
            start = len(example.prompt_ids) - 1
            expected = F.cross_entropy(logits[0, start:start + len(example.answer_ids)].float(),
                                       torch.tensor(example.answer_ids))
        self.assertAlmostEqual(result["records"][0]["answer_ce"], float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
