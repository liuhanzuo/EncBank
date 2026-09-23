"""Small CPU-only checks for source split, targets, and document exclusions."""
from collections import Counter
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from prepare_sft_data import (BenchmarkExclusions, TRAIN_MEMBER, answer_text,
                              load_official_train, paper_context, question_rows,
                              split_documents)


def annotation(text="supported answer", *, unanswerable=False, evidence=None):
    return {"answer": {"unanswerable": unanswerable, "free_form_answer": text,
                       "extractive_spans": [], "yes_no": None,
                       "evidence": ["ordinary source paragraph"] if evidence is None else evidence}}


class SFTPreparationTests(unittest.TestCase):
    def test_only_official_train_member_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.tgz"
            with tarfile.open(archive, "w:gz") as stream:
                for name, content in ((TRAIN_MEMBER, b'{"train-paper": {}}'),
                                      ("qasper-dev-v0.3.json", b'not valid JSON')):
                    item = tarfile.TarInfo(name)
                    item.size = len(content)
                    stream.addfile(item, io.BytesIO(content))
            self.assertEqual(list(load_official_train(archive)), ["train-paper"])

    def test_answer_types_and_unanswerable(self):
        self.assertIsNone(answer_text(annotation(unanswerable=True)))
        self.assertIsNone(answer_text(annotation(text="")))
        self.assertEqual(answer_text({"answer": {"yes_no": False}}), "No")
        self.assertEqual(answer_text({"answer": {"yes_no": True}}), "Yes")
        self.assertEqual(answer_text({"answer": {"extractive_spans": ["a", "b"]}}), "a, b")

    def test_context_never_uses_answers_or_evidence(self):
        paper = {"title": "source title", "abstract": "source abstract",
                 "full_text": [{"section_name": "body", "paragraphs": ["source body"]}],
                 "qas": [{"question": "secret question", "answers": [annotation("secret answer", evidence=["secret evidence"])]}]}
        context = paper_context(paper)
        self.assertIn("source body", context)
        self.assertNotIn("secret", context)

    def test_multiple_references_and_figure_only_filter(self):
        paper = {"title": "doc", "full_text": [], "qas": [
            {"question": "question", "question_id": "q1", "answers": [annotation("first"), annotation("second"), annotation("first")]},
            {"question": "visual question", "question_id": "q2", "answers": [annotation(evidence=["FLOAT SELECTED: image.png"])]},
            {"question": "impossible", "question_id": "q3", "answers": [annotation(unanswerable=True)]}]}
        counts = Counter()
        rows = question_rows("original-paper-id", paper, counts)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document_id"], "original-paper-id")
        self.assertEqual(rows[0]["answer"], "first")
        self.assertEqual(rows[0]["answers"], ["first", "second"])
        self.assertEqual(counts["figure_only_annotations"], 1)

    def test_document_holdout_is_not_row_split(self):
        groups = {f"paper{i}": [{"id": f"paper{i}:q{q}", "document_id": f"paper{i}", "context": f"different context {i}"}
                                for q in range(4)] for i in range(7)}
        train, dev, receipt = split_documents(groups, 9, 5, 42)
        self.assertEqual((len(train), len(dev)), (9, 5))
        self.assertFalse({x["document_id"] for x in train} & set(receipt["reserved_dev_document_ids"]))
        self.assertEqual(receipt["dev_questions_dropped_at_cap"], 3)
        self.assertEqual((train, dev, receipt), split_documents(dict(reversed(list(groups.items()))), 9, 5, 42))

    def test_title_and_shifted_shingle_exclusion(self):
        body = " ".join(f"distinctword{i}" for i in range(256))
        index = BenchmarkExclusions([("heldout", "Unrelated prefix " + body)])
        match = index.match("", "additional words here " + body + " end")
        self.assertEqual(match["kind"], "16_word_shingle_overlap")
        self.assertIsNone(index.match("", "entirely different unrelated text"))
        title = "A genuinely distinctive complete scientific paper title"
        self.assertEqual(BenchmarkExclusions([("heldout", "prefix " + title + " suffix")]).match(title, "different extraction")["kind"],
                         "paper_title_in_context")

    def test_identical_documents_do_not_cross_holdout(self):
        groups = {"a": [{"id": "a:q", "document_id": "a", "context": "same body"}],
                  "b": [{"id": "b:q", "document_id": "b", "context": "same body"}]}
        with self.assertRaises(AssertionError):
            split_documents(groups, 1, 1, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
