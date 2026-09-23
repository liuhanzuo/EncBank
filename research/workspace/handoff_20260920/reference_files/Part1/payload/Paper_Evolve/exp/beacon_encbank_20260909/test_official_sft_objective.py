"""CPU comparison against the unchanged Encbank reader and prefix semantics."""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from test_train_sft import fixture
from official_sft_objective import conversation_loss


class ConversationObjectiveChecks(unittest.TestCase):
    chunks = [[3, 4, 5, 6, 7], [8, 9, 10, 11]]
    # Stand-in for two turns: only selected response/EOS positions are labeled.
    query = [20, 21, 31, 2, 22, 23, 41, 42]
    labels = [-100, 31, 2, -100, -100, 41, 42, 2]

    def test_loss_and_gradients_match_full_vocabulary_reader(self):
        for mode in ("beacon", "encbank", "pool"):
            with self.subTest(mode=mode):
                efficient, _ = fixture(mode)
                reference, _ = fixture(mode)
                actual = conversation_loss(efficient, self.chunks, self.query,
                                           self.labels, head_chunk_size=2)
                memories = [reference.write_chunk(c) for c in self.chunks]
                query = reference.reader.write_chunk(self.query)
                sink = reference.reader.write_chunk([reference.sink_token_id])
                if mode != "beacon":
                    sink = sink.detach().requires_grad_(True)
                logits = reference.reader.read_core(sink, reference._hidden_list(memories),
                                                    query, logits_tail=len(self.query))
                expected = F.cross_entropy(logits[0].float(), torch.tensor(self.labels))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                actual.backward()
                expected.backward()
                for (name, p), (other, q) in zip(efficient.named_parameters(), reference.named_parameters()):
                    self.assertEqual(name, other)
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad)
                        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-5)
                    else:
                        self.assertIsNone(p.grad)

    def test_writer_called_once_per_document_chunk_without_chat_tokens(self):
        net, _ = fixture()
        with mock.patch.object(net, "write_chunk", wraps=net.write_chunk) as writing:
            loss = conversation_loss(net, self.chunks, self.query, self.labels)
            loss.backward()
        self.assertEqual([call.args[0] for call in writing.call_args_list], self.chunks)

    def test_causal_prefix_targets_and_chunk_size_do_not_change_loss(self):
        net, _ = fixture("beacon", checkpointing=False)
        with torch.no_grad():
            actual = conversation_loss(net, self.chunks, self.query, self.labels, head_chunk_size=1)
            larger = conversation_loss(net, self.chunks, self.query, self.labels, head_chunk_size=100)
            memories = [net.write_chunk(c) for c in self.chunks]
            independent = []
            for index, target in enumerate(self.labels):
                if target != -100:
                    logits = net.read_logits(self.query[:index + 1], memories)[0, -1:]
                    independent.append(F.cross_entropy(logits.float(), torch.tensor([target])))
            torch.testing.assert_close(actual, larger)
            torch.testing.assert_close(actual, torch.stack(independent).mean(), atol=2e-6, rtol=1e-5)

    def test_rejects_empty_or_misaligned_supervision(self):
        net, _ = fixture()
        for labels in ([-100] * len(self.query), [], [1]):
            with self.assertRaises(ValueError):
                conversation_loss(net, self.chunks, self.query, labels)
        with self.assertRaises(ValueError):
            conversation_loss(net, [], self.query, self.labels)
        with self.assertRaises(ValueError):
            conversation_loss(net, self.chunks, self.query, self.labels, head_chunk_size=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
