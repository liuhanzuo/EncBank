"""Check support alignment/coverage without evaluating a language model."""
import unittest

from prepare_gold_support import passage_sections, align_fact
from diagnose_gold_support import coverage, budgeted_oracle_indices
from diagnose_locomo_evidence import turn_spans, diagnostic


class EvidenceTests(unittest.TestCase):
    def test_sentence_alignment_never_substitutes_an_answer_word(self):
        context = "Passage 1:\nA title\nThe gold\n  is red. The answer is blue.\n"
        sections = passage_sections(context)
        kind, spans = align_fact(context, sections, "A title", "The gold is red.")
        self.assertEqual(kind, "unicode_whitespace")
        self.assertEqual(context[spans[0][0]:spans[0][1]], "The gold\n  is red.")
        kind, spans = align_fact(context, sections, "A title", "Blue was discovered in 1990.")
        self.assertEqual((kind, spans), ("sentence_not_found", []))

    def test_token_coverage_unknown_support_and_budget(self):
        offsets = [(i, i + 1) for i in range(30)]
        mapping = {"supporting_facts": [{"title": "T", "sent_id": 0, "alignment": "exact",
            "context_char_spans": [[8, 13]], "support_document_char_spans": [[0, 20]]}]}
        partial = coverage(mapping, offsets, 0, [0], 10)
        self.assertFalse(partial["all_facts_fully_retrieved"])
        self.assertTrue(partial["all_support_documents_touched"])
        full = coverage(mapping, offsets, 0, [0, 1], 10)
        self.assertTrue(full["all_facts_fully_retrieved"])
        self.assertEqual(budgeted_oracle_indices(full["facts"], [2], 1)["status"], "support_exceeds_budget")
        oracle = budgeted_oracle_indices(full["facts"], [2, 3], 3)
        self.assertEqual(oracle["indices"], [0, 1, 2])
        mapping["supporting_facts"][0]["context_char_spans"] = []
        unknown = coverage(mapping, offsets, 0, [0, 1, 2], 10)
        self.assertIsNone(unknown["all_facts_fully_retrieved"])

    def test_duplicate_dialogue_text_is_disambiguated_chronologically(self):
        fragment = 'Ann said, "Yes."\n\n'
        conversation = {"session_1": [{"speaker": "Ann", "text": "Yes.", "dia_id": "D1:1"}],
                        "session_2": [{"speaker": "Ann", "text": "Yes.", "dia_id": "D2:1"}]}
        context = "prefix\n" + fragment + "date\n" + fragment
        spans = turn_spans(context, conversation)
        self.assertGreater(spans["D2:1"][0], spans["D1:1"][0])
        row = {"id": "conv0_qa0", "index": 0, "category": 1, "score": 0.5,
            "evidence": ["D1:1; D2:1"], "pack": {"selected_indices": [0]}}
        result = diagnostic(row, spans, [(i, i + 1) for i in range(len(context))], 10)
        self.assertTrue(result["all_annotations_resolved"])
        self.assertFalse(result["all_support_turns_fully_retrieved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
