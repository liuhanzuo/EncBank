"""CPU equivalence tests against the real Qwen3 MLP with original FP32 LoRA.

Run in the experiment's Transformers environment. No model checkpoint is
loaded and no CUDA device is initialized.
"""
from copy import deepcopy
import contextlib
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'encbank_v2_benchmarks_20260908'))
from train_8b_baseline import LoRALinear
from chunked_mlp import ChunkedGatedMLP, install_chunked_mlp


def mlp(dtype=torch.float32):
    config = Qwen3Config(hidden_size=24, intermediate_size=56, hidden_act='silu')
    result = Qwen3MLP(config).to(dtype=dtype).eval().requires_grad_(False)
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        projection = LoRALinear(getattr(result, name), 4, 5, torch.float32)
        with torch.no_grad():
            projection.B.normal_(std=.12)
        setattr(result, name, projection)
    return result


class TinyQwen3(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = Qwen3Config(hidden_size=24, intermediate_size=56)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.mlp = mlp(dtype)
        self.head = nn.Linear(24, 31, bias=False).to(dtype=dtype).requires_grad_(False)

    def forward(self, x, outer_checkpoint=False):
        for layer in self.model.layers:
            if outer_checkpoint:
                x = checkpoint(lambda value, current=layer: value + current.mlp(value), x,
                               use_reentrant=False)
            else:
                x = x + layer.mlp(x)
        return self.head(x)


class ChunkedMLPChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_fp32_complete_ce_and_all_parameter_input_gradients(self):
        self.compare_training(torch.float32, outer_checkpoint=False)

    def test_fp32_nested_layer_checkpoint_eval_mode_gradients(self):
        self.compare_training(torch.float32, outer_checkpoint=True)

    def test_bf16_autocast_nested_checkpoint_complete_ce_gradients(self):
        self.compare_training(torch.bfloat16, outer_checkpoint=True)

    def compare_training(self, dtype, outer_checkpoint):
        torch.manual_seed(42)
        original = TinyQwen3(dtype).eval()
        changed = deepcopy(original).eval()
        install_chunked_mlp(changed, 5)
        # Two batches, multiple blocks, an incomplete final block and all turns
        # supervised through the same answer-only CE mask in both executions.
        raw = torch.randn(2, 13, 24, dtype=dtype)
        inputs = [raw.clone().requires_grad_(True) for _ in range(2)]
        labels = torch.randint(31, (2, 13)); labels[:, 2::4] = -100
        outputs, losses = [], []
        for model, x in zip((original, changed), inputs):
            ctx = torch.autocast('cpu', dtype=torch.bfloat16) if dtype == torch.bfloat16 else contextlib.nullcontext()
            with ctx:
                output = model(x, outer_checkpoint)
                loss = F.cross_entropy(output.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)
            loss.backward()
            outputs.append(output.detach()); losses.append(loss.detach())
        rtol, atol = ((3e-2, 8e-4) if dtype == torch.bfloat16 else (2e-5, 2e-7))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=rtol, atol=atol)
        torch.testing.assert_close(losses[0], losses[1], rtol=rtol, atol=atol)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=rtol, atol=atol)
        left, right = dict(original.named_parameters()), dict(changed.named_parameters())
        self.assertEqual(left.keys(), right.keys())
        trainable = 0
        for name, parameter in left.items():
            self.assertEqual(parameter.requires_grad, right[name].requires_grad)
            if parameter.requires_grad:
                trainable += 1
                self.assertIsNotNone(parameter.grad, name)
                self.assertIsNotNone(right[name].grad, name)
                self.assertGreater(parameter.grad.float().norm().item(), 0, name)
                torch.testing.assert_close(parameter.grad, right[name].grad, rtol=rtol, atol=atol, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)
                self.assertIsNone(right[name].grad, name)
        self.assertEqual(trainable, 12)
        self.assertFalse(torch.cuda.is_initialized())

    def test_state_keys_parameter_objects_and_optimizer_survive_installation(self):
        torch.manual_seed(7)
        model = TinyQwen3(torch.float32).eval()
        before = dict(model.named_parameters())
        state = deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        group = list(optimizer.param_groups[0]['params'])
        paths = install_chunked_mlp(model, 5)
        self.assertEqual(len(paths), 2)
        after = dict(model.named_parameters())
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(before[name] is after[name] for name in before))
        self.assertEqual(state.keys(), model.state_dict().keys())
        model.load_state_dict(state, strict=True)
        self.assertTrue(all(any(p is q for q in model.parameters()) for p in group))
        self.assertTrue(all(type(layer.mlp.gate_proj) is LoRALinear for layer in model.model.layers))

    def test_eval_grad_mode_chunks_full_mlp_and_bounds_retained_temporaries(self):
        torch.manual_seed(8)
        original = mlp(torch.bfloat16)
        chunked = ChunkedGatedMLP(original, 5).eval()
        shapes, saved = [], []
        hooks = [getattr(chunked, name).register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape)))
            for name in ('gate_proj', 'up_proj', 'down_proj')]
        x = torch.randn(2, 13, 24, dtype=torch.bfloat16).requires_grad_(True)
        def keep(tensor):
            saved.append((tuple(tensor.shape), tensor.dtype))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(keep, lambda value: value):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = chunked(x)
        output.float().square().mean().backward()
        for hook in hooks: hook.remove()
        self.assertTrue(shapes)
        self.assertTrue(all(len(shape) == 2 and shape[0] <= 5 for shape in shapes))
        # Checkpoint retains only the original input blocks and empty dummies,
        # not the expanded-width gate/up/product or FP32-cast intermediates.
        self.assertTrue(all(shape == (0,) or (len(shape) == 2 and shape[0] <= 5 and shape[1] == 24)
                            for shape, _ in saved), saved)
        self.assertTrue(all(shape == (0,) or dtype == torch.bfloat16 for shape, dtype in saved))

    def test_no_grad_inference_uses_original_expression_and_disabled_adapters(self):
        torch.manual_seed(9)
        original = mlp()
        changed = ChunkedGatedMLP(deepcopy(original), 5).eval()
        for module in (original, changed):
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                getattr(module, name).enabled = False
        # Noncontiguous input, inference and adapter-disable semantics survive.
        x = torch.randn(2, 24, 13).transpose(1, 2)
        shapes = []
        hook = changed.gate_proj.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
        with torch.no_grad():
            torch.testing.assert_close(original(x), changed(x), rtol=0, atol=0)
        hook.remove()
        self.assertEqual(shapes, [x.shape])


if __name__ == '__main__':
    unittest.main(verbosity=2)
