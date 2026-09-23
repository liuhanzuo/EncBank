"""Synthetic tiny CPU compatibility check; never loads benchmark/model weights."""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
import torch
from transformers import LlamaConfig
from modeling_memoryllm import MemoryLLM


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
                         pad_token_id=0, num_blocks=4, num_tokens=16,
                         drop_memory_per_layer=True, add_decoder_lora=True,
                         add_bos_embedding=True,
                         lora_config={'r': 2, 'lora_alpha': 4, 'lora_dropout': 0.0,
                                      'inference_mode': False, 'target_modules': ['q_proj', 'v_proj']})
    config._attn_implementation = 'sdpa'
    model = MemoryLLM(config).eval()
    initial = model.memory.detach().clone()
    first_state = model.initialized.clone()
    context = torch.arange(3, 35).unsqueeze(0)
    query = torch.tensor([[3, 4, 5, 6]])
    with torch.inference_mode():
        model.inject_memory(context, update_memory=True)
        model.inject_memory(context.flip(1), update_memory=True)
        stored = model.memory.detach().clone()
        initialized = model.initialized.clone()
        output1 = model.generate(input_ids=query, do_sample=False, max_new_tokens=3,
                                 min_new_tokens=1, pad_token_id=0)
        assert torch.equal(stored, model.memory), 'Generation mutated memory'
        assert torch.equal(initialized, model.initialized), 'Generation changed initialized buffer'
        model.memory.data.copy_(initial)
        model.initialized.copy_(first_state)
        torch.manual_seed(123)
        # Seed immediately before each ingestion gives a replayable stochastic eviction path.
        model.inject_memory(context, update_memory=True)
        model.inject_memory(context.flip(1), update_memory=True)
        expected = model.memory.detach().clone()
        model.memory.data.copy_(initial)
        model.initialized.copy_(first_state)
        torch.manual_seed(123)
        model.inject_memory(context, update_memory=True)
        model.inject_memory(context.flip(1), update_memory=True)
        assert torch.equal(expected, model.memory), 'Seeded reset/replay differs'
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert not torch.cuda.is_initialized()
    result = {'status': 'PASS_OFFICIAL_TINY_CPU_ONLY', 'model_shape': '2 layers / hidden32 / 64 memory tokens',
              'official_source_imported': True, 'lora_and_decoder_adapter': True,
              'sdpa_injection_and_generation': True, 'generation_preserves_memory_exactly': True,
              'seeded_reset_replay_bitwise_equal': True, 'cuda_initialized': torch.cuda.is_initialized(),
              'real_checkpoint_loaded': False, 'benchmark_examples_evaluated': 0,
              'generated_synthetic_tokens': output1.shape[1] - query.shape[1]}
    report_name = sys.argv[1] if len(sys.argv) > 1 else 'official_cpu_check.json'
    if '/' in report_name or '\\' in report_name:
        raise ValueError('Report must be a filename inside the preparation directory')
    (HERE / report_name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
