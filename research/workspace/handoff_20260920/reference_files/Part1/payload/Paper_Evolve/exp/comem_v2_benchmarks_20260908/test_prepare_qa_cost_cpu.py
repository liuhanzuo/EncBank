"""Small standard-library checks for the QA cost input contract; no CUDA/model."""
import copy
import unittest

from prepare_qa_cost import ARMS, CAPS, PACK_KEYS, make_plan, native_decode_accounting, reference_output, validate_pair


class QACostContractTests(unittest.TestCase):
    def fixture(self):
        sample = dict(index=0, id="sample", task="qasper", question="Complete question?",
                      answers=["answer"], max_new_tokens=128)
        pack = dict(input_tokens=1097, context_tokens=1000, query_tokens=97, context_chunks=2,
                    selected_indices=[0, 1], read_pack_tokens=1098, truncation="none",
                    context_query_boundary="explicit; independently tokenized; no padding")
        pair = {arm: dict(sample, pack=copy.deepcopy(pack)) for arm in ARMS}
        return sample, pack, pair

    def test_matching_pair_and_complete_question(self):
        sample, pack, pair = self.fixture()
        validate_pair(sample, pack, pair)
        pair["j0"]["question"] = "Complete"
        with self.assertRaisesRegex(ValueError, "question differs"):
            validate_pair(sample, pack, pair)

    def test_retrieval_or_tokenization_mismatch_is_rejected(self):
        for field, bad in [("selected_indices", [1, 0]), ("query_tokens", 96), ("truncation", "tail")]:
            sample, pack, pair = self.fixture()
            pair["fix_all"]["pack"][field] = bad
            with self.assertRaisesRegex(ValueError, "pack differs"):
                validate_pair(sample, pack, pair)

    def test_native_terminal_eos_costs_an_extra_decode_forward(self):
        actual = native_decode_accounting([10, 11, 12], cap=128, eos_id=99)
        self.assertEqual(actual["generated_tokens"], 3)
        self.assertEqual(actual["decode_forward_calls"], 3)
        self.assertEqual(actual["sampled_tokens_including_terminal_eos"], 4)
        self.assertEqual(actual["termination_reason"], "eos")

    def test_native_cap_and_invalid_ids(self):
        actual = native_decode_accounting([10, 11, 12], cap=3, eos_id=99)
        self.assertEqual(actual["decode_forward_calls"], 2)
        self.assertEqual(actual["termination_reason"], "max_new_tokens")
        for ids in ([], [99], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                native_decode_accounting(ids, cap=3, eos_id=99)

    def test_text_retokenization_is_never_claimed_as_actual_generation(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text)
        output = reference_output({"pred": "a b", "score": 1.0}, Tokenizer())
        self.assertEqual(output["decoded_text_retokenized_tokens"], 3)
        self.assertIsNone(output["generated_tokens"])
        self.assertIsNone(output["generated_ids"])
        self.assertIsNone(output["termination_reason"])

    def test_finite_plan_has_no_gpu_runner_or_fixed_generation(self):
        from pathlib import Path
        rows = {task: [dict(index=i, pack={"read_pack_tokens": 10 + i}) for i in range(200)]
                for task in CAPS}
        plan = make_plan(Path("prepared"), rows)
        smoke = [j for j in plan["jobs"] if j["phase"] == "smoke"]
        full = [j for j in plan["jobs"] if j["phase"] == "full"]
        self.assertEqual(sum(len(j["indices"]) for j in smoke), 8)
        self.assertEqual(sum(len(j["indices"]) * j["timed_repetitions"] for j in full), 2400)
        self.assertEqual(sum(j["unmeasured_warmups"] for j in full), 4)
        self.assertIsNone(plan["runner_entrypoint"])
        self.assertFalse(plan["gpu_started"])
        self.assertTrue(all(j["allow_later_eos"] for j in plan["jobs"]))


if __name__ == "__main__":
    unittest.main()
