"""CPU-only input and scorer checks; never load model weights."""
import copy
import unittest

from prepare_workload import (build_workload, full_input_ids, normalized_question,
                              official, retokenize_query, sample_for_query,
                              score_query, validate_workload)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(c) for c in text]


class WorkloadTests(unittest.TestCase):
    def inputs(self):
        samples, source = [], []
        index = 0
        for ci in range(2):
            history = f"Document {ci}: apples, dates and places."
            rows = []
            for qi, question in enumerate(("Where apples?", "Which date?", "Who visited?", " where  APPLES? ")):
                answer = "orchard" if qi in (0, 3) else "answer"
                rows.append(dict(question=question, answer=answer, category=4))
                samples.append(dict(index=index, id=f"conv{ci}_qa{qi}", task="category_4", category=4,
                    category_name="single_hop", question=question, retrieval_question=question,
                    answers=[answer], evidence=[], option_mapping=None,
                    marked_prompt=history + official().BOUNDARY + "Answer: " + question,
                    max_new_tokens=32))
                index += 1
            source.append(dict(sample_id=f"test-{ci}", qa=rows))
        return samples, source

    def build(self, samples=None, source=None):
        if samples is None:
            samples, source = self.inputs()
        return build_workload(samples, source, CharacterTokenizer(), source_path="synthetic.json",
                              chunk_size=8, selector="bm25", topk=2, prefixes=(1, 2, 3))

    def test_all_documents_distinct_questions_and_nested_prefixes(self):
        m, docs, queries = self.build()
        self.assertEqual((m["raw_eligible_queries"], m["unique_queries"], m["duplicates_removed"]), (8, 6, 2))
        self.assertFalse(m["timing_eligible"])
        self.assertTrue(validate_workload(m, docs, queries))
        for d in docs:
            self.assertEqual(d["prefixes"]["1"], d["prefixes"]["3"][:1])
            self.assertEqual(d["prefixes"]["full"], d["prefixes"]["3"])
        self.assertEqual(sum(len(q["source_refs"]) for q in queries), 8)

    def test_original_sample_roundtrip_and_exact_tokens(self):
        samples, _ = self.inputs()
        originals = {s["id"]: s for s in samples}
        _, docs, queries = self.build()
        dm = {d["document_id"]: d for d in docs}
        for q in queries:
            d = dm[q["document_id"]]
            self.assertEqual(sample_for_query(d, q), originals[q["id"]])
            ids, *_ = retokenize_query(CharacterTokenizer(), d, q)
            self.assertEqual(ids[0].tolist(), full_input_ids(d, q))

    def test_reproducible_document_independent_shuffle(self):
        a = self.build()
        b = self.build()
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])
        self.assertEqual([q.split("_qa")[1] for q in a[1][0]["query_ids_ordered"]],
                         [q.split("_qa")[1] for q in a[1][1]["query_ids_ordered"]])

    def test_conflicting_duplicate_answer_is_rejected(self):
        samples, source = self.inputs()
        samples[3]["answers"] = ["different"]
        source[0]["qa"][3]["answer"] = "different"
        with self.assertRaisesRegex(ValueError, "changes answer"):
            self.build(samples, source)

    def test_changed_context_with_same_conversation_is_rejected(self):
        samples, source = self.inputs()
        samples[1]["marked_prompt"] = "Changed document" + official().BOUNDARY + "Question?"
        with self.assertRaisesRegex(ValueError, "context changed"):
            self.build(samples, source)

    def test_prefix_and_pack_tampering_are_rejected(self):
        m, docs, queries = self.build()
        bad = copy.deepcopy(docs)
        bad[0]["prefixes"]["1"] = bad[0]["query_ids_ordered"][1:2]
        with self.assertRaisesRegex(ValueError, "prefix"):
            validate_workload(m, bad, queries)
        q = copy.deepcopy(queries[0])
        q["query_ids"][0] += 1
        d = next(d for d in docs if d["document_id"] == q["document_id"])
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            retokenize_query(CharacterTokenizer(), d, q)

    def test_scorer_is_the_official_category_wrapper(self):
        for category, answer, prediction in ((1, "apples, pears", "apples, pears"),
                                              (2, "May 2023", "May 2023"),
                                              (3, "Paris; extra detail", "Paris"),
                                              (4, "an orchard", "orchard")):
            q = {"sample": {"category": category, "answers": [answer]}}
            self.assertEqual(score_query(prediction, q), official().score_prediction(prediction, q["sample"]))
            self.assertEqual(score_query(prediction, q)[0], 1.0)
        with self.assertRaises(ValueError):
            score_query("[OOM]", {"sample": {"category": 4, "answers": ["x"]}})

    def test_question_normalization(self):
        self.assertEqual(normalized_question("  Which\n DATE? "), "which date?")


if __name__ == "__main__":
    unittest.main()
