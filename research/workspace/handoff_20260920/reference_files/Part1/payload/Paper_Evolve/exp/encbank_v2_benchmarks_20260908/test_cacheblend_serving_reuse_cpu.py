"""CPU-only durable CacheBlend reuse, fresh-path equivalence and accounting."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before importing this CPU-only test")

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from cacheblend_serving_reuse import ReusableContextualCacheBlend, kv_inventory
from cacheblend_contextual import ContextualCacheBlend
from cacheblend_serving_driver import main
from serving_reuse import ReusableReader
import serving_reuse

torch.set_num_threads(2)
torch.set_num_interop_threads(16)
assert not torch.cuda.is_available()


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def decode(self, ids, **kwargs):
        return ' '.join(str(int(i)) for i in ids)
    def encode(self, text, **kwargs):
        return [3 + ord(c) % 90 for c in text]


def make_model(attention='sdpa'):
    torch.manual_seed(73)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048)
    cfg._attn_implementation = attention
    return Qwen3ForCausalLM(cfg).eval(), Tokenizer()


def fresh_reference(reader, chunks, query, count, force):
    cb, cm = reader.cb, reader.cm
    pieces = [cm._as_ids([cm._sink_prefix_id()])] + [cm._as_ids(c) for c in chunks] + [cm._as_ids(query)]
    offsets, offset = [], 0
    for piece in pieces:
        offsets.append(offset)
        offset += piece.numel()
    merged = cb.concat_kv_reindex([cb.prefill_chunk_full(p)[0] for p in pieces], offsets)
    stats = {}
    logits, _, mixed = cb.read(torch.cat(pieces, dim=1), merged, 1, len(query), cb.recompute_ratio, stats)
    cache = cb.decode_cache(mixed)
    next_logits = logits[0, -1].float().clone()
    next_logits[2] = -torch.inf
    ids, steps = [int(next_logits.argmax())], [next_logits]
    for step in range(1, count):
        next_logits = cb.decode_step(ids[-1], cache, offset + step - 1)[0, -1].float().clone()
        if force:
            next_logits[2] = -torch.inf
        steps.append(next_logits)
        token = int(next_logits.argmax())
        if not force and token == 2:
            break
        ids.append(token)
    return ids, steps, stats


class CacheBlendReuseTests(unittest.TestCase):
    def test_durable_cpu_disk_reuse_matches_fresh_logits_and_ids(self):
        # Noncontiguous/reordered packs and short last chunks test rotation and
        # boundaries; eager and SDPA independently exercise mask conventions.
        cases = [([31, 32, 33], [0, 2]), ([41, 42], [3, 1]), ([51, 52, 53, 54], [2, 0])]
        context = list(range(3, 29))
        for attention in ('eager', 'sdpa'):
            model, tok = make_model(attention)
            with tempfile.TemporaryDirectory() as temp:
                writer = ReusableContextualCacheBlend(model, 2, tok, 'tiny')
                with patch.object(writer.cb, 'prefill_chunk_full', wraps=writer.cb.prefill_chunk_full) as capture:
                    write = writer.write_store(context, temp, 8)
                    self.assertEqual(capture.call_count, 5)
                self.assertEqual(write['capture_calls'], 5)
                self.assertEqual(write['payload_tensor_bytes'], 27 * writer.cb.kv_bytes_per_tok())
                self.assertEqual(write['serialized_bytes'], sum(p.stat().st_size for p in Path(temp).iterdir()))
                for tier in ('cpu', 'disk'):
                    reader = ReusableContextualCacheBlend(model, 2, tok, 'tiny')
                    reader.open_store(temp, tier)
                    for query, indices in cases:
                        chunks = [context[i*8:(i+1)*8] for i in indices]
                        expected, logits, expected_stats = fresh_reference(reader, chunks, query, 6, True)
                        with patch.object(reader.cb, 'prefill_chunk_full', side_effect=AssertionError('query recaptured a chunk')):
                            actual, stats = reader.query_ids(query, selected_indices=indices,
                                max_new_tokens=6, force_length=True, capture_logits=True, capture_inventory=True)
                        self.assertEqual(actual, expected)
                        self.assertEqual(stats['cacheblend']['selected_positions'], expected_stats['selected_positions'])
                        for got, want in zip(stats['step_logits'], logits):
                            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                        self.assertEqual(stats['decode_steps'], 5)
                        self.assertEqual(stats['capture_calls'], 0)
                        self.assertFalse(stats['timing_eligible'])
                        self.assertEqual(stats['cacheblend']['n_recompute_ctx'], int(.16 * sum(map(len, chunks))))
                        inv = stats['online_kv_inventory']
                        n = 1 + sum(map(len, chunks)) + len(query)
                        self.assertEqual(inv['blended_after_prefill']['logical_tensor_bytes'], n * reader.cb.kv_bytes_per_tok())
                        self.assertEqual(inv['decode_end']['logical_tensor_bytes'], (n + 5) * reader.cb.kv_bytes_per_tok())
                        self.assertEqual(inv['decode_end']['unique_storage_bytes'], inv['decode_end']['logical_tensor_bytes'])
                    reader.close_store()

    def test_natural_eos_matches_existing_accuracy_entry(self):
        model, tok = make_model()
        with tempfile.TemporaryDirectory() as temp:
            reader = ReusableContextualCacheBlend(model, 2, tok, 'tiny')
            reader.write_store(list(range(3, 27)), temp, 8)
            reader.open_store(temp, 'cpu')
            chunks = [reader.chunks[i] for i in [2, 0]]
            expected = reader.cb.generate_explicit(chunks, [51, 52, 53], 1, 2, 10)
            actual, _ = reader.query_ids([51, 52, 53], selected_indices=[2, 0], max_new_tokens=10)
            self.assertEqual(actual, expected)

    def test_full_recompute_identity_and_store_immutability(self):
        model, tok = make_model()
        with tempfile.TemporaryDirectory() as temp:
            reader = ReusableContextualCacheBlend(model, 2, tok, 'tiny', recompute_ratio=1.)
            reader.write_store(list(range(3, 27)), temp, 8)
            reader.open_store(temp, 'cpu')
            original = {name: [(k.clone(), v.clone()) for k, v in item['kv']] for name, item in reader.payloads.items()}
            for query in ([31, 32], [41, 42, 43]):
                actual, stats = reader.query_ids(query, selected_indices=[2, 0], max_new_tokens=5,
                    force_length=True, capture_logits=True)
                running = torch.tensor([[1] + list(range(19, 27)) + list(range(3, 11)) + list(query)])
                expected = []
                for step in range(5):
                    logits = model(running, use_cache=False).logits[0, -1].float().clone()
                    logits[2] = -torch.inf
                    torch.testing.assert_close(stats['step_logits'][step], logits, atol=2e-6, rtol=2e-5)
                    token = int(logits.argmax())
                    expected.append(token)
                    running = torch.cat([running, torch.tensor([[token]])], 1)
                self.assertEqual(actual, expected)
            for name, pairs in original.items():
                for (k, v), (after_k, after_v) in zip(pairs, reader.payloads[name]['kv']):
                    self.assertTrue(torch.equal(k, after_k) and torch.equal(v, after_v))

    def test_inventory_deduplicates_views_and_uses_actual_storage(self):
        tensor = torch.zeros(1, 2, 8, 4)
        inv = kv_inventory([(tensor[:, :, :4], tensor[:, :, 4:])])
        self.assertEqual(inv['logical_tensor_bytes'], tensor.numel() * 4)
        self.assertEqual(inv['unique_storages'], 1)
        self.assertEqual(inv['unique_storage_bytes'], tensor.untyped_storage().nbytes())

    def test_retrieval_contract_and_store_signature_errors(self):
        model, tok = make_model()
        with tempfile.TemporaryDirectory() as temp:
            reader = ReusableContextualCacheBlend(model, 2, tok, 'tiny')
            reader.write_store(list(range(3, 27)), temp, 8)
            reader.open_store(temp, 'disk')
            _, recency = reader.query_ids([28, 29], selector='recency', topk=1, max_new_tokens=2)
            _, bm25 = reader.query_ids([4, 5], selector='bm25', topk=1, max_new_tokens=2)
            self.assertEqual(recency['selected_indices'], [2])
            self.assertEqual(bm25['selected_indices'], [0])
            for bad in ([0, 0], [3], [-1]):
                with self.assertRaises(ValueError):
                    reader.query_ids([29], selected_indices=bad)
            with self.assertRaises(ValueError):
                reader.query_ids([], selected_indices=[0])
            with self.assertRaises(ValueError):
                reader.write_store([1, 2], temp)
            with self.assertRaises(ValueError):
                ReusableContextualCacheBlend(model, 2, tok, 'tiny', recompute_ratio=.5).open_store(temp)
            reader.close_store()
            with self.assertRaises(ValueError):
                reader.query_ids([29])

    def test_standalone_driver_preserves_accounting_and_restores_dependencies(self):
        model, tok = make_model()
        original_class, original_arms = serving_reuse.ReusableReader, serving_reuse.ARMS
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            doc, out = Path(temp) / 'doc.txt', Path(temp) / 'out'
            doc.write_text('Apples and pears in an orchard. ' * 30)
            args = ['--model', 'tiny', '--context-file', str(doc), '--out', str(out),
                '--device', 'cpu', '--cpu-test-only', '--dtype', 'float32', '--j', '2',
                '--chunk-size', '8', '--topk', '2', '--lengths', '32', '--query-counts', '1', '3',
                '--generation-lengths', '2', '4']
            captures = []
            original = ContextualCacheBlend.prefill_chunk_full
            def record_capture(self, ids, *a, **kw):
                captures.append(torch.as_tensor(ids).numel())
                return original(self, ids, *a, **kw)
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tok), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model), \
                 patch.object(ContextualCacheBlend, 'prefill_chunk_full', record_capture):
                self.assertEqual(main(args), 0)
            self.assertEqual(captures, [1, 8, 8, 8, 8])
            rows = json.loads((out / 'summary.json').read_text())
            self.assertEqual(len(rows), 8)
            for row in rows:
                q, g, tier = row['Q'], row['G'], row['tier']
                raw = [json.loads(line) for line in (out / f'queries_32_cacheblend16_{tier}_g{g}.jsonl').read_text().splitlines()]
                self.assertEqual(row['query_totals']['generated_tokens'], q*g)
                self.assertEqual(row['query_totals']['decode_steps'], q*(g-1))
                self.assertAlmostEqual(row['end_to_end_total_s'], row['write']['write_total_s'] + row['startup']['startup_load_s'] + sum(x['total_s'] for x in raw[:q]))
                self.assertFalse(row['source_preprocessing_charged'])
                self.assertTrue(all(x['cacheblend']['bootstrap_full_layers'] == 2 for x in raw))
            self.assertFalse(json.loads((out / 'COMPLETED.json').read_text())['hardware']['timing_eligible'])
        self.assertIs(serving_reuse.ReusableReader, original_class)
        self.assertEqual(serving_reuse.ARMS, original_arms)

    def test_smoke_verification_counts_extra_fresh_sequences_and_retains_store(self):
        model, tok = make_model()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            doc, out = Path(temp) / 'doc.txt', Path(temp) / 'out'
            doc.write_text('Apples and pears. ' * 10)
            args = ['--model', 'tiny', '--context-file', str(doc), '--out', str(out), '--verify-fresh',
                '--device', 'cpu', '--cpu-test-only', '--dtype', 'float32', '--j', '2',
                '--chunk-size', '8', '--topk', '2', '--lengths', '32', '--query-counts', '1', '2',
                '--generation-lengths', '2', '4']
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tok), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model):
                self.assertEqual(main(args), 0)
            proof = json.loads((out / 'FRESH_REFERENCE_CHECK.json').read_text())
            self.assertEqual(proof['verified_queries'], 8)
            self.assertEqual(proof['extra_reference_sequences'], 8)
            self.assertFalse(proof['timing_eligible'])
            self.assertTrue(all(x['reuse_capture_calls'] == 0 and x['extra_reference_prefill_calls'] == 4 for x in proof['checks']))

    def test_wrong_driver_arm_rejected_before_model_or_gate(self):
        with patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=AssertionError('loaded model')):
            for args in (['--arms', 'pub'], ['--arms=pub'], ['--arms', 'cacheblend16', 'pub']):
                with self.assertRaises(ValueError):
                    main(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
