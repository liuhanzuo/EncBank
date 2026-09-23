"""Small stdlib-only protocol tests; no Torch import, model, CUDA or gate."""
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from backend_parity import (_cache_comparison, _hot_arguments, logits_metrics,
                            route_metrics, run_backend_parity, validate_state, vector_metrics)


class FakeTensor:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.dtype = 'torch.bfloat16'
        self.device = 'cuda:0'
        self._version = 0
        self.values = values
    def stride(self):
        return tuple(math.prod(self.shape[i+1:]) for i in range(len(self.shape)))
    def storage_offset(self): return 0
    def data_ptr(self): return id(self)
    def untyped_storage(self): return self
    def nbytes(self): return self.numel()*self.element_size()
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 2
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def reshape(self, *_args): return self
    def tolist(self): return self.values


class FakeState:
    pass


def pair(tokens):
    return SimpleNamespace(k=FakeTensor([1, 8, tokens, 4]), v=FakeTensor([1, 8, tokens, 4]))


class FakeReader:
    training = False
    j, m, L, rho, probe_mode = 1, 2, 3, .5, 'block'
    num_heads, num_kv_heads, head_dim = 32, 8, 4
    attention_chunk_size, score_chunk_size = 512, 512
    def __init__(self, model, delta=0., reference=None):
        self.model, self.delta, self.reference = model, delta, reference
        self.last_state = None
        self.consumed = []
    def prefill(self, _sink, documents, prompt, probe_indices=None, **_kwargs):
        if self.reference is not None:
            assert self.reference.last_state() is None, 'Reference request state was not released'
        state = FakeState()
        state.query_position = len(prompt)
        state.pack_position = 10+len(prompt)
        state.selected_indices = [0]
        state.route_stats = dict(selected_indices=[0], block_scores=[.6, .4], original_query_start=10,
                                 document_kv_tokens_by_layer={'1': 10, '2': 4})
        state.doc_kv = {1: pair(10), 2: pair(4)}
        state.query_kv = {i: pair(len(prompt)) for i in range(3)}
        self.last_state = weakref.ref(state)
        return FakeTensor([1, 1, 3], [0., 1.+self.delta, 2.+self.delta]), state
    def decode_step(self, token, state):
        self.consumed.append(token)
        state.query_position += 1
        state.pack_position += 1
        state.query_kv = {i: pair(state.query_position) for i in range(3)}
        return FakeTensor([1, 1, 3], [0., 1.+self.delta, 2.+self.delta])


class BackendParityProtocolTests(unittest.TestCase):
    def test_metrics_report_small_error_without_hiding_flip(self):
        metrics = logits_metrics([0., 1., 2.], [.01, 1.01, 2.01])
        self.assertTrue(metrics['passed'])
        self.assertAlmostEqual(metrics['max_abs'], .01)
        self.assertAlmostEqual(metrics['rms'], .01)
        self.assertAlmostEqual(metrics['reference']['top2_margin'], 1.)
        flip = logits_metrics([0., 1., 1.001], [0., 1.002, 1.001])
        self.assertTrue(flip['within_error_thresholds'])
        self.assertFalse(flip['greedy_equal'])
        self.assertFalse(flip['passed'])
    def test_threshold_nan_and_mismatch_fail(self):
        self.assertFalse(logits_metrics([0., 1.], [0., 1.16])['passed'])
        self.assertFalse(logits_metrics([0., 1.], [.03, 1.03])['passed'])
        self.assertFalse(logits_metrics([0., 1.], [0., float('nan')])['passed'])
        self.assertFalse(logits_metrics([0., 1.], [1.])['passed'])
        self.assertEqual(vector_metrics([], [])['rms'], 0.)
    def test_routes_scores_may_differ_but_every_other_field_is_checked(self):
        ref = dict(selected_indices=[0, 2], block_scores=[.5, .1, .4], document_kv_bytes=42)
        other = {**ref, 'block_scores': [.51, .09, .4]}
        self.assertTrue(route_metrics(ref, other)['passed'])
        self.assertFalse(route_metrics(ref, {**other, 'selected_indices': [0, 1]})['passed'])
        self.assertFalse(route_metrics(ref, {**other, 'document_kv_bytes': 43})['passed'])
        self.assertFalse(route_metrics(ref, {**other, 'extra': None})['passed'])
        self.assertFalse(route_metrics(ref, {**other, 'block_scores': [float('inf'), .1, .4]})['passed'])
        self.assertFalse(route_metrics(ref, {**other, 'extra': {'nonfinite': 'nan'}})['passed'])
    def test_cache_metadata_is_evidence_not_content_proof(self):
        meta = {'tensor': {'version_available': True, 'version': 0, 'shape': [1, 8, 4, 128]}}
        self.assertTrue(_cache_comparison(meta, copy.deepcopy(meta))['passed'])
        changed = copy.deepcopy(meta)
        changed['tensor']['version'] = 1
        self.assertFalse(_cache_comparison(meta, changed)['passed'])
        meta['tensor']['version_available'] = False
        self.assertFalse(_cache_comparison(meta, meta)['passed'])
        self.assertIn('not proven', _cache_comparison(meta, meta)['evidence_scope'])
    def test_hot_argument_contract(self):
        entries = [SimpleNamespace(source_key='a'), SimpleNamespace(source_key='b')]
        _, kwargs = _hot_arguments(entries, None, 2)
        self.assertEqual(kwargs['document_keys'], ['a', 'b'])
        with self.assertRaises(ValueError): _hot_arguments([None], None, 1)
        _, kwargs = _hot_arguments([None], ['missing'], 1)
        self.assertEqual(kwargs['document_keys'], ['missing'])
        with self.assertRaises(ValueError):
            _hot_arguments({'hot_entries': entries, 'document_keys': ['a', 'b']}, ['x', 'y'], 2)
    def test_end_to_end_sequential_state_release_and_receipt(self):
        model = object()
        reference = FakeReader(model)
        candidate = FakeReader(model, .001, reference)
        fake_torch = SimpleNamespace(no_grad=nullcontext)
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', {'torch': fake_torch}):
            receipt = run_backend_parity(reference, candidate, None,
                [FakeTensor([1, 4, 32]), FakeTensor([1, 6, 32])], [1, 2], [0, 1], out_dir=folder)
            self.assertIs(receipt['passed'], True, receipt['errors'])
            self.assertEqual(len(receipt['stages']), 4)
            self.assertEqual(reference.consumed, candidate.consumed)
            self.assertIsNone(candidate.last_state())
            saved = json.loads(Path(receipt['receipt_path']).read_text())
            self.assertIs(saved['passed'], True)
            self.assertFalse(saved['formal_timing_eligible'])
            stage = receipt['stages'][0]['reference_state']
            bad = copy.deepcopy(stage['state'])
            bad['query_kv']['0']['k']['storage_bytes'] *= 100
            check = validate_state(bad, stage['route_stats'], layers=3, resume_j=1,
                expected_heads=8, head_dim=4, prompt_length=2, decode_step=0)
            self.assertFalse(check['passed'])
            bad = copy.deepcopy(stage['state'])
            bad['doc_kv']['1']['k']['shape'][1] = 32
            check = validate_state(bad, stage['route_stats'], layers=3, resume_j=1,
                expected_heads=8, head_dim=4, prompt_length=2, decode_step=0)
            self.assertFalse(check['passed'])
    def test_runtime_error_saved_as_boolean_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = run_backend_parity(FakeReader(object()), FakeReader(object()), None,
                                         [], [1], [0], out_dir=folder)
            self.assertIs(receipt['passed'], False)
            self.assertEqual(receipt['errors'][0]['type'], 'ValueError')
            self.assertFalse(json.loads(Path(receipt['receipt_path']).read_text())['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
