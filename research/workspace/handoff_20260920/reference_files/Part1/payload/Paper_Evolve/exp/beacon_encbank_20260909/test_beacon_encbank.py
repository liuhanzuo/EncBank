"""CPU semantic checks on a tiny real Qwen3, not quality or timing benchmarks."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from beacon_encbank import BeaconEncbank

torch.set_num_threads(2)


class TestLoRA(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.base = linear
        self.A = nn.Parameter(torch.randn(4, linear.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(linear.out_features, 4))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t()) @ self.B.t()


def fixture(ratio=4, checkpointing=False, document_sink=False):
    torch.manual_seed(73)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=256, attention_dropout=0.0,
                        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).cpu().eval()
    model.requires_grad_(False)
    for layer in model.model.layers[2:]:
        layer.self_attn.q_proj = TestLoRA(layer.self_attn.q_proj)
        layer.self_attn.v_proj = TestLoRA(layer.self_attn.v_proj)
    return BeaconEncbank(model, split=2, compression_ratio=ratio,
                       init_token_id=2, sink_token_id=1,
                       writer_id="tiny-qwen-seed73-test-writer",
                       document_write_sink=document_sink,
                       gradient_checkpointing=checkpointing)


class BeaconSemantics(unittest.TestCase):
    def test_complete_tail_and_physical_storage(self):
        for ratio in (4, 8):
            for length in (1, 4, 8, 9, 17):
                with self.subTest(ratio=ratio, length=length):
                    net = fixture(ratio, document_sink=True)
                    ids = list(range(3, 3 + length))
                    memory = net.encode_chunk(ids)
                    count = (length + ratio - 1) // ratio
                    self.assertEqual(tuple(memory.hidden.shape), (1, count, 64))
                    self.assertEqual(memory.source_tokens, length)
                    self.assertEqual(memory.hidden.untyped_storage().nbytes(), memory.tensor_bytes)
                    self.assertFalse(memory.hidden.requires_grad)
                    self.assertIsNone(memory.hidden.grad_fn)
                    expanded, rows = net.interleave(ids)
                    self.assertEqual(int(rows[-1]), expanded.shape[1] - 1)
                    ordinary = torch.ones(expanded.shape[1], dtype=torch.bool)
                    ordinary[rows.cpu()] = False
                    ordinary[0] = False
                    torch.testing.assert_close(expanded[:, ordinary, :], net.reader.embed_tokens(torch.tensor([ids])))

    def test_causal_context_and_interleaved_dependence(self):
        net = fixture()
        a = net.encode_chunk([3, 4, 5, 6, 7, 8, 9, 10]).hidden
        b = net.encode_chunk([3, 4, 5, 6, 20, 21, 22, 23]).hidden
        torch.testing.assert_close(a[:, 0], b[:, 0], atol=1e-6, rtol=1e-5)
        self.assertGreater(float((a[:, 1] - b[:, 1]).abs().max()), 1e-6)
        c = net.encode_chunk([30, 31, 32, 33, 7, 8, 9, 10]).hidden
        self.assertGreater(float((a[:, 1] - c[:, 1]).abs().max()), 1e-6)

    def test_answer_loss_reaches_beacon_and_reader_only(self):
        net = fixture()
        frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
        trainables = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainables, lr=0.01)
        before = net.beacon_embedding.detach().clone()
        query = [12, 13, 14, 15]
        logits = net([[3, 4, 5, 6, 7, 8, 9, 10]], query)
        loss = nn.functional.cross_entropy(logits[0, :-1], torch.tensor(query[1:]))
        loss.backward()
        self.assertTrue(torch.isfinite(net.beacon_embedding.grad).all())
        self.assertGreater(float(net.beacon_embedding.grad.norm()), 0.0)
        gradients = [p.grad for n, p in net.named_parameters() if n.endswith(".B")]
        self.assertTrue(any(g is not None and float(g.norm()) > 0 for g in gradients))
        self.assertTrue(all(p.grad is None for p in net.parameters() if not p.requires_grad))
        optimizer.step()
        self.assertFalse(torch.equal(before, net.beacon_embedding))
        for name, value in frozen.items():
            torch.testing.assert_close(dict(net.named_parameters())[name], value, atol=0, rtol=0)

    def test_prefill_decode_and_multiple_queries_reuse(self):
        net = fixture()
        memory = net.encode_chunk([3, 4, 5, 6, 7, 8, 9, 10, 11])
        original = memory.hidden.clone()
        for query in ([12, 13, 14], [21, 22, 23, 24]):
            state = net.start_query(query, [memory])
            expected = net.read_logits(query, [memory])[:, -1:]
            torch.testing.assert_close(state.logits, expected, atol=2e-6, rtol=1e-5)
            self.assertEqual(state.bottom_cache.get_seq_length(0), len(query))
            self.assertEqual(state.top_cache.get_seq_length(2), 1 + 3 + len(query))
            extended = list(query)
            for token in (31, 32, 33):
                extended.append(token)
                decoded = net.decode_token(token, state)
                fresh = net.read_logits(extended, [memory])[:, -1:]
                torch.testing.assert_close(decoded, fresh, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(memory.hidden, original, atol=0, rtol=0)

    def test_checkpointing_preserves_output_and_gradient(self):
        ordinary = fixture(checkpointing=False)
        checked = fixture(checkpointing=True)
        for model in (ordinary, checked):
            logits = model([[3, 4, 5, 6, 7, 8, 9]], [12, 13, 14])
            logits.square().mean().backward()
        torch.testing.assert_close(ordinary.beacon_embedding.grad, checked.beacon_embedding.grad)
        for (name, p), (other_name, q) in zip(ordinary.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)
        torch.testing.assert_close(ordinary([[3, 4, 5, 6]], [12, 13]),
                                   checked([[3, 4, 5, 6]], [12, 13]))

    def test_round_trip_all_trainables_and_memory(self):
        net = fixture()
        with torch.no_grad():
            for p in net.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.02)
        memory = net.encode_chunk([3, 4, 5, 6, 7, 8, 9])
        reference = net.read_logits([11, 12, 13], [memory]).detach()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            cache = Path(directory) / "memory.pt"
            net.save_trainable(checkpoint)
            net.save_memory(memory, cache)
            state = torch.load(checkpoint, weights_only=True)
            self.assertIn("beacon_embedding", state["trainable"])
            self.assertTrue(any(k.endswith(".B") for k in state["trainable"]))
            self.assertEqual(sum(x.numel() for x in state["trainable"].values()),
                             sum(p.numel() for p in net.parameters() if p.requires_grad))
            saved_memory = torch.load(cache, weights_only=True)
            self.assertEqual(set(saved_memory), {"configuration", "hidden", "source_tokens"})
            self.assertEqual(saved_memory["hidden"].untyped_storage().nbytes(), memory.tensor_bytes)
            restored = fixture()
            restored.load_trainable(checkpoint)
            loaded_memory = restored.load_memory(cache)
            torch.testing.assert_close(restored.read_logits([11, 12, 13], [loaded_memory]), reference)
            with self.assertRaises(ValueError):
                fixture(ratio=8).load_trainable(checkpoint)

    def test_invalid_inputs_and_no_memory_control(self):
        net = fixture()
        with self.assertRaises(ValueError):
            net.write_chunk([])
        with self.assertRaises(ValueError):
            net.start_query([], [])
        self.assertEqual(tuple(net.read_logits([3, 4], []).shape), (1, 2, 97))
        with self.assertRaises(ValueError):
            net.read_logits([3, 4], [fixture(ratio=8).encode_chunk([5, 6])])

    def test_writer_identity_and_wrapper_dtype_change(self):
        net = fixture()
        old = net.encode_chunk([3, 4, 5, 6])
        net.writer_id = "different-locked-writer"
        with self.assertRaises(ValueError):
            net.read_logits([7, 8], [old])
        net.double()
        memory = net.encode_chunk([3, 4, 5, 6])
        self.assertEqual(memory.hidden.dtype, torch.float64)
        state = net.start_query([7, 8], [memory])
        self.assertEqual(net.reader.dtype, torch.float64)
        torch.testing.assert_close(state.logits, net.read_logits([7, 8], [memory])[:, -1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
