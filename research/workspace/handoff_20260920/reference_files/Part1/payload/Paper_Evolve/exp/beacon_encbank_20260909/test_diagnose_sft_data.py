"""CPU/stdlib checks for strict evidence coverage and boundary accounting."""
import unittest

from diagnose_sft_data import contains_snippet, describe, inspect_evidence, normalized


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def chunks(*texts):
    return [tuple(map(ord, text)) for text in texts]


class EvidenceDiagnosticTests(unittest.TestCase):
    def test_normalization_is_not_bag_of_words(self):
        self.assertEqual(normalized("Ａlpha\n  BETA"), "alpha beta")
        self.assertTrue(contains_snippet("alpha beta gamma", "alpha beta"))
        self.assertFalse(contains_snippet("alpha beta gamma", "beta alpha"))
        self.assertFalse(contains_snippet("alpha beta", "alpha gamma"))
        self.assertFalse(contains_snippet("nothing", "no"))
        self.assertNotEqual(normalized("1-2"), normalized("12"))

    def test_adjacent_boundary_and_inside_word_are_reconstructed(self):
        result = inspect_evidence("international law", chunks("inter", "national law"), [0, 1], ["international law"], CharacterTokenizer())
        snippet = result["snippets"][0]
        self.assertTrue(snippet["retrieved_complete_snippet"])
        self.assertTrue(snippet["retrieved_across_adjacent_chunk_boundary"])
        self.assertEqual(snippet["selected_single_chunk_matches"], [])

    def test_nonadjacent_chunks_cannot_fabricate_match(self):
        content = chunks("alpha ", "gap ", "gamma ", "alpha gamma")
        result = inspect_evidence("alpha gap gamma alpha gamma", content, [0, 2], ["alpha gamma"], CharacterTokenizer())
        snippet = result["snippets"][0]
        self.assertTrue(snippet["source_text_match"])
        self.assertFalse(snippet["retrieved_complete_snippet"])
        self.assertTrue(snippet["nonadjacent_join_would_false_match"])
        self.assertEqual(snippet["status"], "source_locatable_but_not_fully_retrieved")

    def test_missing_source_annotation_is_not_retrieval_miss(self):
        result = inspect_evidence("alpha beta", chunks("alpha ", "beta"), [1], ["alpha", "missing gamma"], CharacterTokenizer())
        self.assertEqual(result["category"], "source_annotation_not_fully_locatable")
        self.assertEqual(result["source_and_decoded_locatable_snippets"], 1)
        self.assertEqual(result["snippets"][1]["status"], "source_text_not_locatable")

    def test_empty_annotations_are_unknown_not_success(self):
        result = inspect_evidence("alpha beta", chunks("alpha beta"), [0], ["", "  "], CharacterTokenizer())
        self.assertEqual(result["category"], "no_text_evidence_annotation")
        self.assertFalse(result["all_annotations_retrieved"])

    def test_tokenizer_roundtrip_issue_is_separate(self):
        class ChangedDecoder(CharacterTokenizer):
            def decode(self, ids, **kwargs):
                return super().decode(ids, **kwargs).replace("alpha", "altered")
        result = inspect_evidence("alpha beta", chunks("alpha beta"), [0], ["alpha"], ChangedDecoder())
        self.assertEqual(result["category"], "tokenizer_roundtrip_mismatch")
        self.assertEqual(result["source_and_decoded_locatable_snippets"], 0)

    def test_locatable_denominator_and_answer_truncation_are_explicit(self):
        evidence = inspect_evidence("alpha beta", chunks("alpha beta"), [0], ["alpha", "missing"], CharacterTokenizer())
        result = describe([{"document_id": "p", "evidence": evidence, "primary_answer_kind": "extractive",
                            "answer_truncated": True, "answer_target_tokens": 128, "original_context_tokens": 2}])
        self.assertEqual(result["annotated_snippets"], 2)
        self.assertEqual(result["locatable_snippets"], 1)
        self.assertEqual(result["strict_snippet_recall_on_locatable_annotations"], 1.0)
        self.assertEqual(result["questions_with_all_annotations_retrieved"], 0)
        self.assertEqual(result["answer_truncated_examples"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
