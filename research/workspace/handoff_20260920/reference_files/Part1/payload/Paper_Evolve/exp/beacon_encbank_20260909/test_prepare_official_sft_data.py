"""CPU data checks: exact reconstruction, conversation causality, and grouping."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from prepare_official_sft_data import (
    MEMORY_PLACEHOLDER, SOURCE_FILES, UnsupportedRecord, exact_document_id,
    iter_canonical_conversations, iter_canonical_examples, normalized_document_group,
    parse_first_user, parse_record, prepare_corpus, reconstruct_original_messages,
)


def record(first, answers=("first answer", "later answer")):
    result = [{"role": "user", "content": first}, {"role": "assistant", "content": answers[0]}]
    for index, answer in enumerate(answers[1:], start=1):
        result.extend(({"role": "user", "content": f"What is item {index}?"},
                       {"role": "assistant", "content": answer}))
    return {"conversations": result}


class ParserChecks(unittest.TestCase):
    def test_exact_spans_and_known_question_instruction(self):
        context = "  Chapter One\r\n\r\nA fact with   spaces.\n\n"
        source = "Story:\n\n" + context + "\n\nAnswer my question using the knowledge from the context:\nWhat happened?"
        parsed = parse_first_user(source, "gpt_book")
        self.assertEqual(parsed["context"], context)
        self.assertEqual(parsed["original_first_user_prefix"] + parsed["context"] + parsed["original_first_user_suffix"], source)
        self.assertNotIn("A fact", parsed["question"])
        self.assertIn("Answer my question", parsed["question"])
        self.assertIn(MEMORY_PLACEHOLDER, parsed["question"])

    def test_explicit_rule_and_longalpaca_markers(self):
        ruled = "Context information is below:\n----------\nDOC\n----------\nDefine a fact."
        self.assertEqual(parse_first_user(ruled, "gpt_paper")["context"], "DOC")
        single = "Below is a paper. Memorize the paper and answer my question after the paper.\n The paper begins. \nRAW DOC\n Now the paper ends. \nQuestion: Why?"
        first = parse_first_user(single, "longalpaca")
        self.assertEqual(first["context"], " \nRAW DOC\n ")
        dual = "There are two papers. Memorize them and answer my question after the paper.\n The first paper begins. P1 Now the first paper ends. The second paper begins. P2 Now the second paper ends.Compare the styles."
        second = parse_first_user(dual, "longalpaca")
        self.assertIn("P1", second["context"])
        self.assertIn("P2", second["context"])
        self.assertNotIn("Compare", second["context"])
        book = "Below is some paragraphs in the book, A Fictional Title. Memorize the content and answer my question after the book.\n EXACT BOOK \n Now the material ends. Please summarize the book in one paragraph."
        third = parse_first_user(book, "longalpaca")
        self.assertEqual(third["context"], " EXACT BOOK \n ")
        self.assertIn("Please summarize", third["question"])

    def test_booksum_long_targets_and_empty_history(self):
        answer = "summary sentence " * 2000
        raw = record("Context:\n\nComplete book chapter.\n\nSummarize Chapter One.", (answer, ""))
        before = deepcopy(raw)
        document, conversation = parse_record(raw, "booksum", 17)
        self.assertEqual(raw, before)
        self.assertEqual(conversation["turns"][0]["answer"], answer)
        self.assertEqual(conversation["assistant_turn_count"], 2)
        self.assertFalse(conversation["turns"][1]["target_eligible"])
        self.assertEqual(reconstruct_original_messages(conversation, document["context"]), raw["conversations"])
        self.assertNotIn("Give only", conversation["turns"][0]["question"])

    def test_no_guessed_boundaries_or_later_document(self):
        for first in ("Unrecognized intro\n\nDOC\n\nWhat is it?", "Story:\n\nDOC without a separate task", "Story:\n\nDOC\n\nAn ordinary narrative paragraph."):
            with self.assertRaises(UnsupportedRecord):
                parse_first_user(first, "gpt_book")
        raw = record("Story:\n\nDOC\n\nWhat happened?")
        raw["conversations"][2]["content"] = "Story:\n\nLATER DOC\n\nWhat happens next?"
        with self.assertRaisesRegex(UnsupportedRecord, "later_user"):
            parse_record(raw, "gpt_book", 0)

    def test_exact_variants_share_only_normalized_split(self):
        a, b = "Text  with\nspaces", "Text with spaces"
        self.assertNotEqual(exact_document_id(a), exact_document_id(b))
        self.assertEqual(normalized_document_group(a), normalized_document_group(b))
        d1, c1 = parse_record(record("Story:\n\n" + a + "\n\nWhat?"), "gpt_book", 0)
        d2, c2 = parse_record(record("Paper:\n\n" + b + "\n\nWhat?"), "gpt_paper", 0)
        self.assertNotEqual(d1["document_id"], d2["document_id"])
        self.assertEqual(c1["normalized_group_id"], c2["normalized_group_id"])
        self.assertEqual(c1["split"], c2["split"])

    def test_corpus_storage_references_and_legal_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir, output_dir = root / "raw", root / "processed"
            path = raw_dir / SOURCE_FILES["gpt_book"]
            path.parent.mkdir(parents=True)
            rows = [record("Story:\n\nDOC\n\nWhat happened?"), record("Story:\n\nDOC\n\nWhy?"), record("Unknown\n\nDOC\n\nWhat?")]
            original = "".join(json.dumps(item) + "\n" for item in rows)
            path.write_text(original, encoding="utf-8")
            report = prepare_corpus(raw_dir, output_dir, allow_missing=True)
            self.assertEqual(report["counts"]["accepted_conversations"], 2)
            self.assertEqual(report["counts"]["rejected_conversations"], 1)
            self.assertEqual(report["unique_exact_documents"], 1)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            saved = [json.loads(line) for line in (output_dir / "conversations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("context" not in item for item in saved))
            conversations = list(iter_canonical_conversations(output_dir))
            self.assertEqual(conversations[0]["original_messages"], rows[0]["conversations"])
            examples = list(iter_canonical_examples(output_dir))
            self.assertEqual([len(item["messages"]) for item in examples], [1, 3, 1, 3])
            self.assertEqual(examples[0]["context"], "DOC")
            self.assertNotIn("first answer", str(examples[0]["messages"]))
            self.assertIn("first answer", str(examples[1]["messages"]))
            self.assertNotIn("later answer", str(examples[1]["messages"]))
            self.assertEqual(len(examples[1]["original_messages"]), 3)
            with self.assertRaises(FileExistsError):
                prepare_corpus(raw_dir, output_dir, allow_missing=True)

    def test_incomplete_download_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                prepare_corpus(root / "raw", root / "processed")


if __name__ == "__main__":
    unittest.main()
