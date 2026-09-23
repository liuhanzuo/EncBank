"""Meaningful CPU correctness tests; no weights downloaded and no CUDA use."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
import tempfile
from pathlib import Path
import unittest
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import Qwen3Config, Qwen3ForCausalLM
from prefix_cache import PrefixReader, ReusableReader


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2


class PrefixCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=2048,
                            bos_token_id=1, eos_token_id=2, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.tok = Tokenizer()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.document = [10,11,12,13,20,21,22,23,20,21,24,25]
        self.count = 0

    def tearDown(self):
        self.tmp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def make(self, budget=None, document=None):
        self.count += 1
        doc = self.document if document is None else document
        p = PrefixReader(self.model, 1, self.tok, cache_budget_bytes=budget)
        j = ReusableReader(self.model, 1, self.tok, arm="j0")
        for reader, name in ((p,"prefix"),(j,"j0")):
            store = self.path/f"{self.count}_{name}"
            reader.write_store(doc, store, chunk_size=4)
            reader.open_store(store)
        return p,j

    def compare(self, p, j, indices, query=(60,61), force=True, g=5):
        args = dict(selected_indices=indices, max_new_tokens=g, force_length=force, capture_logits=True)
        with torch.enable_grad():
            got, stats = p.query_ids(query, **args)
            ref, refstats = j.query_ids(query, **args)
        self.assertEqual(got, ref)
        self.assertEqual(stats['decode_steps'], refstats['decode_steps'])
        self.assertEqual(len(stats['step_logits']), len(refstats['step_logits']))
        for a,b in zip(stats['step_logits'],refstats['step_logits']):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            self.assertFalse(a.requires_grad)
        self.assertTrue(all(x.grad is None for x in self.model.parameters()))
        self.assertAlmostEqual(stats['ttft_s']+stats['decode_s'],stats['total_s'],places=8)
        self.assertEqual(stats['cache_resident_bytes'],p.cache_info()['resident_bytes'])
        return stats

    def test_cold_full_hit_and_different_query_match_j0(self):
        p,j=self.make()
        cold=self.compare(p,j,[0,1])
        self.assertEqual(cold['cache_hit_context_tokens'],0)
        self.assertEqual(cold['cache_miss_context_tokens'],9)
        warm=self.compare(p,j,[0,1])
        self.assertEqual(warm['cache_hit_context_tokens'],9)
        self.assertEqual(warm['cache_fill_added_bytes'],0)
        changed=self.compare(p,j,[0,1],query=(70,71,72),force=False)
        self.assertEqual(changed['cache_hit_context_tokens'],9)

    def test_partial_edge_prefix_and_shared_bytes(self):
        p,j=self.make()
        self.compare(p,j,[0,1])
        partial=self.compare(p,j,[0,2])
        self.assertEqual(partial['cache_hit_context_tokens'],7)
        self.assertEqual(partial['cache_miss_context_tokens'],2)
        self.assertEqual(p.cache_info()['resident_context_tokens'],11)
        self.assertEqual(p.cache_info()['resident_bytes'],11*p.kv_bytes_per_token)
        self.assertEqual(self.compare(p,j,[0,1])['cache_hit_context_tokens'],9)
        self.assertEqual(self.compare(p,j,[2,0])['cache_hit_context_tokens'],1)

    def test_query_and_generated_tokens_never_enter_persistent_cache(self):
        p,j=self.make()
        self.compare(p,j,[0,1],query=(88,89),g=8)
        before=[x.clone() for node in p._nodes() for pair in node.kv for x in pair]
        tokens_before=[node.tokens for node in p._nodes()]
        self.compare(p,j,[0,1],query=(90,91,92),g=10)
        after=[x for node in p._nodes() for pair in node.kv for x in pair]
        self.assertEqual(tokens_before,[node.tokens for node in p._nodes()])
        self.assertEqual(len(before),len(after))
        for a,b in zip(before,after):
            self.assertTrue(torch.equal(a,b));self.assertEqual(b.device.type,'cpu')
            self.assertFalse(b.requires_grad)
        self.assertEqual(sum(len(node.tokens) for node in p._nodes()),9)
        self.assertEqual(p.cm._ctx_hj,[])

    def test_budget_leaf_eviction_and_recompute_after_eviction(self):
        probe,_=self.make()
        budget=9*probe.kv_bytes_per_token
        p,j=self.make(budget)
        evictions=0
        for indices in ([0,1],[0,2],[2,0],[0,1]):
            s=self.compare(p,j,indices)
            evictions+=s['evicted_nodes']
            self.assertLessEqual(s['cache_resident_bytes'],budget)
        self.assertGreater(evictions,0)

    def test_zero_and_sub_block_budget(self):
        p,j=self.make(0)
        for _ in range(2):
            s=self.compare(p,j,[0,1]);self.assertEqual(s['cache_resident_bytes'],0)
            self.assertEqual(s['cache_hit_context_tokens'],0)
        q,j=self.make(6*p.kv_bytes_per_token)
        self.compare(q,j,[0,1])
        s=self.compare(q,j,[0,1])
        self.assertEqual(s['cache_hit_context_tokens'],6)
        self.assertEqual(s['cache_resident_bytes'],6*p.kv_bytes_per_token)

    def test_block_boundary_longest_prefix_and_close_reset(self):
        doc=[10+i%40 for i in range(520)]
        p,j=self.make(document=doc)
        # One request crosses the actual512-token radix edge boundary.
        indices=list(range(130))
        s=self.compare(p,j,indices,query=(70,),g=2)
        self.assertEqual(s['context_tokens'],521)
        self.assertEqual(p.cache_info()['nodes'],2)
        self.assertEqual(self.compare(p,j,list(range(129)),query=(71,),g=2)['cache_hit_context_tokens'],517)
        p.close_store()
        self.assertEqual(p.cache_info()['resident_bytes'],0)
        with self.assertRaises(ValueError):p.query_ids([60],selected_indices=[0])

    def test_order_and_invalid_inputs(self):
        p,j=self.make()
        self.compare(p,j,[0,1])
        self.compare(p,j,[1,0])
        with self.assertRaises(ValueError):p.query_ids([60],selected_indices=[0,0])
        with self.assertRaises(ValueError):p.query_ids([],selected_indices=[0])
        with self.assertRaises(ValueError):p.open_store(self.path,tier='disk')


if __name__=='__main__':
    unittest.main(verbosity=2)
