"""CPU-only formal provenance and final-marker interruption checks."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch
from transformers import AutoTokenizer
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
import non_qwen_formal_driver as D

class FormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(D.ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
        rows = [json.loads(line) for line in (D.REMOTE / "outputs/diagnostics/non_qwen_fixtures_0531_attempt1/non_qwen_qasper.jsonl").read_text().splitlines()]
        cls.fixture = rows[0]
        cls.smoke_path = D.SMOKE / "records/pub__qasper__000.json"
        cls.pred = json.loads(cls.smoke_path.read_text())

    def test_original_smoke_identity_decode_score_pass(self):
        self.assertTrue(D.validate_prediction(self.pred, self.fixture, "pub", self.tok))

    def test_changed_scored_fields_rejected(self):
        changes = {"model_revision": "wrong", "effective_j": 0, "decoded_output": "changed text",
                   "official_score": 1.234, "suppress_first_eos": False, "decoder_forwards": -1,
                   "fixture_sha256": "wrong", "ordered_selected_indices": [], "source_id": "wrong"}
        for key, value in changes.items():
            changed = copy.deepcopy(self.pred)
            changed[key] = value
            with self.assertRaises(AssertionError, msg=key):
                D.validate_prediction(changed, self.fixture, "pub", self.tok)

    def test_generated_raw_provenance_exact_path_hash_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "attempts/0001/raw/pub__qasper__000.json"
            prediction = copy.deepcopy(self.pred)
            prediction["smoke_only"] = False
            D.write(raw, prediction)
            record = {"schema": "non-qwen-formal-prediction-v1", "model_revision": D.REVISION,
                      "seed": 42, "attn_impl": "sdpa", "prediction": prediction,
                      "provenance": {"kind": "formal_generation", "attempt": 1, "raw_file": str(raw), "raw_sha256": D.sha(raw)}}
            with patch.object(D, "FORMAL", root):
                D.validate_record(record, self.fixture, "pub", self.tok)
                bad_hash = copy.deepcopy(record)
                bad_hash["provenance"]["raw_sha256"] = "wrong"
                with self.assertRaises(AssertionError):
                    D.validate_record(bad_hash, self.fixture, "pub", self.tok)
                bad_attempt = copy.deepcopy(record)
                bad_attempt["provenance"]["attempt"] = 2
                with self.assertRaises(AssertionError):
                    D.validate_record(bad_attempt, self.fixture, "pub", self.tok)
                # Even a correctly updated hash cannot legitimize mismatched raw JSON.
                changed_raw = copy.deepcopy(prediction)
                changed_raw["unrecorded_field"] = True
                D.write(raw, changed_raw)
                record["provenance"]["raw_sha256"] = D.sha(raw)
                with self.assertRaises(AssertionError):
                    D.validate_record(record, self.fixture, "pub", self.tok)

    def complete_records(self):
        records = {}
        for arm in D.ARMS:
            for task, n in (("qasper", 200), ("niah_single_2", 50), ("niah_multikey_1", 50), ("variable_tracking", 50)):
                for i in range(n):
                    smoke = i < (8 if task == "qasper" else 2)
                    records[(arm, task, i)] = {"prediction": {"official_score": .25},
                                              "provenance": {"kind": "verified_smoke" if smoke else "formal_generation"}}
        return records

    def test_last_record_written_missing_marker_recovers_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.complete_records()
            base = {"model_revision": D.REVISION, "j": 8, "seed": 42, "new_this_attempt": 0}
            # Any accidental model invocation by this CPU completion helper fails.
            with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("model load forbidden")):
                result = D.finalize_verified_records(root, base, records)
                self.assertEqual(result["formal_rows"], 1750)
                self.assertEqual(result["verified_smoke_reused"], 70)
                self.assertEqual(result["formal_generations"], 1680)
                self.assertEqual(result["completed_cells"], 20)
                self.assertTrue(result["recovered_from_validated_records_without_gpu"])
                self.assertTrue((root / "non_qwen_FORMAL_COMPLETE.json").exists())
                before = (root / "non_qwen_FORMAL_COMPLETE.json").read_bytes()
                D.write(root / "status.json", {"status": "running"})
                D.finalize_verified_records(root, base, records)
                self.assertEqual(before, (root / "non_qwen_FORMAL_COMPLETE.json").read_bytes())
                self.assertEqual(json.loads((root / "status.json").read_text())["status"], "complete")
            self.assertFalse(torch.cuda.is_initialized())

    def test_incomplete_records_cannot_recover_complete_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            records = self.complete_records()
            records.pop(next(iter(records)))
            with self.assertRaises(AssertionError):
                D.finalize_verified_records(Path(temp), {"model_revision": D.REVISION, "j": 8}, records)
            self.assertFalse((Path(temp) / "non_qwen_FORMAL_COMPLETE.json").exists())

    def test_existing_marker_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.complete_records()
            D.finalize_verified_records(root, {"model_revision": D.REVISION, "j": 8}, records)
            path = root / "non_qwen_FORMAL_COMPLETE.json"
            broken = json.loads(path.read_text())
            broken["formal_rows"] = 1749
            D.write(path, broken)
            with self.assertRaises(AssertionError):
                D.finalize_verified_records(root, {"model_revision": D.REVISION, "j": 8}, records)

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FormalTests))
    assert not torch.cuda.is_initialized()
    print(json.dumps({"complete": result.wasSuccessful(), "tests_run": result.testsRun,
                      "device": "cpu", "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                      "cuda_initialized": False, "new_model_generation": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
