"""Token-mask/preparation tests. Optional real Qwen3 tokenizer is CPU-only."""
import copy
import math
import os
import unittest

from official_sft import (DOCUMENT_MARKER, END_TEXT, IGNORE_INDEX, ConversationFiltered,
                          prepare_conversation, prepare_conversations, render, token_ids)


class ChatTokenizer:
    """Character tokenizer plus atomic ChatML controls, with Qwen3 role behavior."""
    eos_token_id = 2
    SPECIAL = {"<|im_start|>": 1, END_TEXT: 2}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            match = next((s for s in self.SPECIAL if text.startswith(s, cursor)), None)
            end = cursor+len(match) if match else cursor+1
            ids.append(self.SPECIAL[match] if match else ord(text[cursor])+10)
            offsets.append((cursor, end)); cursor = end
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        inv = {v: k for k, v in self.SPECIAL.items()}
        return "".join(inv[token] if token in inv else chr(token-10) for token in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        parts = []
        for i, m in enumerate(messages):
            body = m["content"]
            if m["role"] == "assistant" and i > last_user and i == len(messages)-1:
                body = "<think>\n\n</think>\n\n"+body.lstrip("\n")
            parts.append("<|im_start|>"+m["role"]+"\n"+body+END_TEXT+"\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


def conversation(context="Source document body. "*60, answers=("A", "Long answer")):
    prefix, suffix = "Read the following chapter.\n\n", "\n\nSummarize its main claim without losing details."
    original = [{"role": "user", "content": prefix+context+suffix},
                {"role": "assistant", "content": answers[0]},
                {"role": "user", "content": "Now explain the second point and its relation to your summary."},
                {"role": "assistant", "content": answers[1]}]
    adapted = copy.deepcopy(original)
    adapted[0]["content"] = prefix+DOCUMENT_MARKER+suffix
    return {"conversation_id": "booksum:test:0", "document_id": "exact-document-id", "source": "booksum",
        "source_file": "booksum/train.16K.json", "source_index": 0, "split": "train",
        "context": context, "original_messages": original, "messages": adapted,
        "original_first_user_prefix": prefix, "original_first_user_suffix": suffix,
        "assistant_turn_count": 2, "parser": "strict-test-span", "raw_record_reference": {"index": 0},
        "turns": [{"id": "turn0", "assistant_turn_index": 0, "answer": answers[0], "answers": [answers[0]], "messages": adapted[:1]},
                  {"id": "turn1", "assistant_turn_index": 1, "answer": answers[1], "answers": [answers[1]], "messages": adapted[:3]}]}


class OfficialPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChatTokenizer()

    def prepare(self, sample=None, **kwargs):
        return prepare_conversation(sample or conversation(), self.tokenizer, min_length=0,
                                    max_length=20000, **kwargs)

    def test_full_document_chunks_no_retrieval_no_target_in_writer(self):
        sample = conversation(context="z"*1025)
        p = self.prepare(sample)
        self.assertEqual([len(c) for c in p.document_chunks], [512, 512, 1])
        self.assertEqual(self.tokenizer.decode([t for c in p.document_chunks for t in c]), sample["context"])
        query = self.tokenizer.decode(p.query_ids)
        self.assertNotIn(sample["context"], query)
        self.assertIn(DOCUMENT_MARKER, query)
        self.assertIn("Summarize its main claim without losing details.", query)
        self.assertEqual(p.document_token_count, 1025)
        self.assertEqual(p.raw_uncompressed_read_pack_tokens, 1+1025+len(p.query_ids))
        self.assertFalse(p.protocol["truncated"])

    def test_all_assistants_exact_targets_shift_and_template_controls_masked(self):
        sample = conversation(answers=("Repeated answer", "Repeated answer"))
        p = self.prepare(sample)
        full = self.tokenizer.encode(render(self.tokenizer, sample["messages"]))
        self.assertEqual(p.query_ids, tuple(full[:-1]))
        for index, label in enumerate(p.label_ids):
            if label != IGNORE_INDEX:
                self.assertEqual(label, full[index+1])
        self.assertEqual(len(p.label_ids), len(p.query_ids))
        self.assertEqual(sum(label != IGNORE_INDEX for label in p.labels), p.target_token_count)
        for turn in p.turns:
            self.assertEqual(self.tokenizer.decode(turn.training_target_ids), "Repeated answer"+END_TEXT)
            self.assertEqual(turn.training_target_ids[-1], self.tokenizer.eos_token_id)
        self.assertEqual(tuple(i for turn in p.turns for i in turn.target_indices), p.target_indices)
        self.assertNotIn("<think>", self.tokenizer.decode([p.labels[i] for i in p.target_indices]))
        self.assertNotIn("assistant", self.tokenizer.decode([p.labels[i] for i in p.target_indices]))

    def test_legitimate_generation_history_and_future_turn_exclusion(self):
        sample = conversation(answers=("FIRST_ANSWER", "SECOND_ANSWER"))
        p = self.prepare(sample)
        first, second = [self.tokenizer.decode(t.prompt_ids) for t in p.turns]
        self.assertNotIn("FIRST_ANSWER", first)
        self.assertNotIn("Now explain", first)
        self.assertIn("FIRST_ANSWER", second)
        self.assertIn("Now explain", second)
        self.assertNotIn("SECOND_ANSWER", second)
        self.assertTrue(first.endswith("<think>\n\n</think>\n\n"))
        damaged = copy.deepcopy(sample)
        damaged["turns"][0]["messages"] = sample["messages"][:3]
        with self.assertRaisesRegex(ValueError, "future target"):
            self.prepare(damaged)

    def test_conversation_equal_weight_uses_tokens_not_turns_or_characters(self):
        p = self.prepare(conversation(answers=("A", "Long answer")))
        self.assertEqual([t.target_token_count for t in p.turns], [2, 12])
        self.assertEqual(p.target_token_count, 14)
        self.assertAlmostEqual(sum(t.conversation_token_weight for t in p.turns), 1.)
        losses = [float(i+1) for i in range(14)]
        cursor, weighted = 0, 0.
        for turn in p.turns:
            n = turn.target_token_count
            weighted += (sum(losses[cursor:cursor+n])/n)*turn.conversation_token_weight
            cursor += n
        self.assertAlmostEqual(weighted, sum(losses)/14)
        self.assertNotAlmostEqual(p.turns[0].conversation_token_weight, .5)

    def test_whole_native_original_template_length_filter_not_reader_sum(self):
        sample = conversation(context="source "*100)
        accepted = self.prepare(sample)
        count = len(self.tokenizer.encode(render(self.tokenizer, sample["original_messages"])))
        self.assertEqual(count, accepted.original_template_token_count)
        self.assertLess(accepted.reader_template_token_count, count)
        result = prepare_conversations([sample], self.tokenizer, min_length=0, max_length=count-1)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.skipped[0]["reason"], "full_original_template_length")
        self.assertEqual(result.skipped[0]["original_template_tokens"], count)
        self.assertTrue(result.preparation_summary["complete"])
        exact = prepare_conversation(sample, self.tokenizer, min_length=count, max_length=count)
        self.assertEqual(exact.document_chunks, accepted.document_chunks)
        self.assertEqual(exact.labels, accepted.labels)

    def test_booksum_long_summary_is_not_short_answer_clipped(self):
        summary = "This is the complete chapter summary. "*20
        p = self.prepare(conversation(answers=(summary, "Further explanation.")))
        self.assertGreater(p.turns[0].target_token_count, 128)
        self.assertEqual(p.turns[0].answer, summary)
        self.assertEqual(self.tokenizer.decode(p.turns[0].training_target_ids), summary+END_TEXT)
        self.assertEqual(p.source, "booksum")
        self.assertEqual(p.source_file, "booksum/train.16K.json")
        self.assertEqual(p.raw_record_reference, {"index": 0})

    def test_invalid_document_span_history_or_template_transform_is_rejected(self):
        changes = [lambda x: x.update(context="wrong document"),
                   lambda x: x["messages"][2].update(content="silently changed instruction"),
                   lambda x: x["turns"][1].update(assistant_turn_index=5),
                   lambda x: x.update(assistant_turn_count=1)]
        for change in changes:
            sample = conversation(); change(sample)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(sample)
        # Native Qwen removes leading final-answer newlines: fail, do not hide it.
        with self.assertRaisesRegex(ValueError, "transformed answer"):
            self.prepare(conversation(answers=("A", "\nFull answer")))

    def test_empty_assistant_filters_whole_conversation_and_batch_continues(self):
        invalid = conversation(answers=("", "Still nonempty second answer"))
        invalid["conversation_id"] += ":empty"
        invalid["target_eligible"] = False
        good = conversation()
        result = prepare_conversations([invalid, good], self.tokenizer, min_length=0, max_length=20000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].conversation_id, good["conversation_id"])
        self.assertEqual(result[0].assistant_turn_count, 2)
        self.assertEqual(result.preparation_summary["input_conversations"], 2)
        self.assertEqual(result.preparation_summary["accepted_conversations"], 1)
        self.assertEqual(result.preparation_summary["filtered_conversations"], 1)
        self.assertEqual(result.preparation_summary["filtered_by_reason"], {"empty_assistant_target": 1})
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0]["reason"], "empty_assistant_target")
        self.assertEqual(result.skipped[0]["assistant_turn_indices"], [0])

    def test_ambiguous_cross_boundary_token_is_not_given_user_loss(self):
        class CrossingTokenizer(ChatTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                if kwargs.get("return_offsets_mapping") and "A"+END_TEXT in text:
                    position = text.index("A"+END_TEXT)
                    offsets = value["offset_mapping"]
                    index = next(i for i, pair in enumerate(offsets) if pair[0] == position)
                    offsets[index] = (position-1, position+1)
                return value
        with self.assertRaisesRegex(ValueError, "crosses an assistant"):
            prepare_conversation(conversation(), CrossingTokenizer(), min_length=0, max_length=20000)


@unittest.skipUnless(os.environ.get("OFFICIAL_SFT_QWEN_TOKENIZER"), "Set a local Qwen3 tokenizer path for CPU native-template integration")
class RealQwenTokenizerTests(unittest.TestCase):
    def test_two_turn_qwen3_earlier_and_final_assistant_boundaries(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["OFFICIAL_SFT_QWEN_TOKENIZER"], local_files_only=True)
        sample = conversation(context="A chapter about astronomy and observations. "*30,
                              answers=("The first claim concerns stars.", "Its second point is observational evidence."))
        prepared = prepare_conversation(sample, tokenizer, min_length=0, max_length=20000)
        self.assertEqual(prepared.assistant_turn_count, 2)
        self.assertGreater(prepared.target_token_count, 2)
        self.assertAlmostEqual(sum(t.conversation_token_weight for t in prepared.turns), 1.)
        for turn, answer in zip(prepared.turns, (sample["messages"][1]["content"], sample["messages"][3]["content"])):
            decoded = tokenizer.decode(turn.training_target_ids, skip_special_tokens=False)
            self.assertEqual(decoded, answer+END_TEXT)
            self.assertNotIn("think", decoded)
            self.assertEqual(tokenizer.decode(turn.prompt_ids, skip_special_tokens=False),
                             render(tokenizer, sample["messages"][:turn.message_index], generation=True))
        full = token_ids(tokenizer.apply_chat_template(sample["messages"], tokenize=True,
                                             add_generation_prompt=False, enable_thinking=False))
        self.assertEqual(prepared.query_ids, tuple(full[:-1]))
        for index in prepared.target_indices:
            self.assertEqual(prepared.labels[index], full[index+1])
        # Qwen's historical answer omits the empty-thought block; final answer includes it.
        text = render(tokenizer, sample["messages"])
        self.assertIn("assistant\nThe first claim", text)
        self.assertIn("assistant\n<think>\n\n</think>\n\nIts second", text)
        print("REAL_QWEN3_TOKENIZER", {"original_template_tokens": prepared.original_template_token_count,
              "reader_template_tokens": prepared.reader_template_token_count,
              "document_tokens": prepared.document_token_count,
              "assistant_target_counts": [t.target_token_count for t in prepared.turns],
              "target_tokens": prepared.target_token_count})


if __name__ == "__main__":
    unittest.main(verbosity=2)
