"""New scalar-hook/cache-byte correctness tests; run remotely with CUDA hidden."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("This test requires explicitly hidden CUDA")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / "COMem", ROOT / "exp", HERE):
    sys.path.insert(0, str(p))
snapshot = HERE / "serving_reuse_online_kv_cpu_snapshot.py"
if snapshot.exists():
    spec = importlib.util.spec_from_file_location("serving_reuse", snapshot)
    serving = importlib.util.module_from_spec(spec)
    sys.modules["serving_reuse"] = serving
    spec.loader.exec_module(serving)
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import Qwen3Config, Qwen3ForCausalLM
from serving_reuse import ReusableReader
from cacheblend_serving_reuse import ReusableContextualCacheBlend
from online_kv_probe import inventory, measure_query


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, **kwargs):
        return [3+ord(c)%90 for c in text]


def model_and_tokenizer():
    torch.manual_seed(71)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval(), Tokenizer()


class InventoryTests(unittest.TestCase):
    def test_active_cache_bytes_and_ids_for_all_distinct_reader_paths(self):
        model, tok = model_and_tokenizer()
        with tempfile.TemporaryDirectory() as temp:
            for arm in ("fix_all", "pub", "pub_sink", "j0", "cacheblend16"):
                reader = ReusableContextualCacheBlend(model, 2, tok, "tiny") if arm == "cacheblend16" else ReusableReader(model, 2, tok, arm, "tiny")
                folder = Path(temp) / arm
                reader.write_store(list(range(3, 29)), folder, 8)
                reader.open_store(folder, "disk")
                for selected, query in (([3, 1], [31, 32, 33]), ([0, 2], [41, 42])):
                    expected, _ = reader.query_ids(query, selected_indices=selected, max_new_tokens=5, force_length=True)
                    actual = measure_query(reader, query, selected_indices=selected, max_new_tokens=5)
                    self.assertEqual(expected, actual["generated_ids"])
                    for boundary, extra in (("prefill_complete", 0), ("decode_complete", 4)):
                        n_pack, n_q = actual["read_tokens"]+extra, len(query)+extra
                        token_layers = 2*n_q+2*n_pack if arm in {"pub", "pub_sink"} else 4*n_pack
                        # Tiny model: 2 KV tensors × 2 heads × head_dim 8 × FP32 4 bytes.
                        self.assertEqual(actual["inventory"][boundary]["logical_tensor_bytes"], token_layers*2*2*8*4)
                        self.assertEqual(actual["inventory"][boundary]["populated_layer_entries"], 4)
                    self.assertFalse(actual["timing_eligible"])
                    self.assertNotIn("total_s", actual)
                reader.close_store()
        self.assertFalse(torch.cuda.is_initialized())

    def test_underlying_storage_deduplicates_aliases_and_counts_full_views(self):
        backing = torch.zeros(1, 2, 20, 8)
        left, right = backing[:, :, :5], backing[:, :, 5:10]
        actual = inventory([("test", 0, left, right), ("test", 1, left, right)])
        self.assertEqual(actual["unique_backing_storages"], 1)
        self.assertEqual(actual["unique_backing_storage_bytes"], backing.untyped_storage().nbytes())
        self.assertEqual(actual["logical_tensor_bytes"], 4*left.numel()*left.element_size())

    def test_hooks_restore_and_reader_clears_after_exception(self):
        model, tok = model_and_tokenizer()
        reader = ReusableReader(model, 2, tok, "fix_all", "tiny")
        with tempfile.TemporaryDirectory() as folder:
            reader.write_store(list(range(3, 19)), folder, 8)
            reader.open_store(folder, "disk")
            keys = set(reader.cm.__dict__)
            with patch.object(reader.cm, "read_prefill", side_effect=RuntimeError("probe failure")):
                with self.assertRaisesRegex(RuntimeError, "probe failure"):
                    measure_query(reader, [31, 32], selected_indices=[1, 0], max_new_tokens=5)
            self.assertIsNone(reader.cm._bottom)
            self.assertEqual(set(reader.cm.__dict__), keys)


if __name__ == "__main__":
    unittest.main()
