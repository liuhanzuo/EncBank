"""Full LoCoMo MemoryLLM inference. Only label-free projected inputs are accepted."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/srv/encbank')
QUESTION_MARKER = '\n\n# Question\n'
FORBIDDEN = {'answer', 'answers', 'gold_answer', 'reference', 'references', 'category',
             'evidence', 'evidence_texts', 'adversarial_answer', 'is_abstention'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_no_labels(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError(f'Inference label fields found: {FORBIDDEN.intersection(value)}')
        for child in value.values():
            validate_no_labels(child)
    elif isinstance(value, list):
        for child in value:
            validate_no_labels(child)


def load_inputs(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError('Frozen inference input hash mismatch')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_no_labels(data)
    conversations = data['conversations']
    if len(conversations) != 10:
        raise ValueError('Expected all 10 conversations')
    items = [q for c in conversations for q in c['items']]
    ids = [q['source_id'] for q in items]
    if len(ids) != 1986 or len(set(ids)) != 1986:
        raise ValueError('Expected all 1986 unique source IDs')
    expected_ids = [f"conv{c['conversation_index']}_qa{i}" for c in conversations
                    for i in range(len(c['items']))]
    if ids != expected_ids or [c['conversation_index'] for c in conversations] != list(range(10)):
        raise ValueError('Source question ordering mismatch')
    return data


def chunk_right(token_ids, size=512, minimum=17):
    """Official right-aligned 512-token chunks, with a short leading tail merged."""
    if len(token_ids) < minimum:
        raise ValueError('Context shorter than the official safe injection minimum')
    first = len(token_ids) % size
    chunks = ([token_ids[:first]] if first else [])
    chunks += [token_ids[i:i + size] for i in range(first, len(token_ids), size)]
    if len(chunks[0]) < minimum:
        chunks = [chunks[0] + chunks[1]] + chunks[2:]
    if [t for c in chunks for t in c] != token_ids or min(map(len, chunks)) < minimum:
        raise AssertionError('Chunking dropped, reordered, or underfilled tokens')
    return chunks


def resolve_owned(path):
    p = Path(path).resolve()
    if not str(p).startswith(str(ROOT) + '/'):
        raise ValueError(f'Path outside own remote root: {p}')
    return p


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


class TokenTimer:
    def __init__(self):
        self.prompt_seen = False
        self.first = None
        self.count = 0

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first is None:
            self.first = time.perf_counter()
        self.count += int(value.numel())

    def end(self):
        pass


def execute(args, data):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(resolve_owned(args.source_dir)))
    from modeling_memoryllm import MemoryLLM

    for path in [args.model_dir, args.output_dir, args.source_dir, args.inputs]:
        resolve_owned(path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    output = resolve_owned(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / 'predictions.jsonl'
    all_ids = {q['source_id'] for c in data['conversations'] for q in c['items']}
    finished_ids = set()
    if predictions_path.exists():
        if not args.resume:
            raise RuntimeError('Existing predictions require explicit --resume')
        for line in predictions_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['source_id'] not in all_ids or row['source_id'] in finished_ids:
                raise ValueError('Unexpected or duplicate existing source ID')
            if row['protocol_id'] != args.protocol_id or row['input_sha256'] != args.input_sha256:
                raise ValueError('Resume protocol/input mismatch')
            finished_ids.add(row['source_id'])
    if finished_ids == all_ids:
        raise RuntimeError('All source IDs already complete; refusing redundant model load')
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f'Insufficient free GPU memory: {free / 2**30:.2f} GiB')
    torch.set_grad_enabled(False)
    device = torch.device('cuda:0')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = MemoryLLM.from_pretrained(args.model_dir, torch_dtype=dtype,
                                    attn_implementation='sdpa', local_files_only=True,
                                    low_cpu_mem_usage=True).eval().to(device)
    if any(p.device != device for p in model.parameters()):
        raise RuntimeError('Model parameters are not fully resident on the assigned GPU')
    if any(p.is_floating_point() and p.dtype != dtype for p in model.parameters()):
        raise RuntimeError('Unexpected model parameter precision')
    # The official driver restores a checkpoint-memory copy. Keep it resident on GPU.
    initial_memory = model.memory.detach().clone()
    initial_initialized = model.initialized.detach().clone()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    props = torch.cuda.get_device_properties(0)
    execution = {'protocol_id': args.protocol_id, 'input_sha256': args.input_sha256,
                 'model_dir': args.model_dir, 'model_revision': args.model_revision,
                 'dtype': args.dtype, 'attention_backend': 'sdpa', 'seed': args.seed,
                 'pid': os.getpid(), 'gpu_name': props.name, 'gpu_uuid': str(getattr(props, 'uuid', 'unavailable')),
                 'gpu_total_memory_bytes': total, 'initial_free_memory_bytes': free,
                 'cold_start_load_seconds': load_seconds, 'torch': torch.__version__,
                 'memory_reset': 'GPU-resident initial checkpoint memory and initialized buffer per conversation',
                 'offload': False, 'max_new_tokens': 48, 'chat_template': False,
                 'initial_memory_bytes': initial_memory.numel() * initial_memory.element_size(),
                 'status': 'RUNNING', 'expected_predictions': 1986}
    write_json(output / 'execution.json', execution)
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    stop_ids = sorted(set(x for x in stop_ids if x is not None and x >= 0))
    with predictions_path.open('a', encoding='utf-8', buffering=1) as sink, torch.inference_mode():
        for conversation in data['conversations']:
            pending = [q for q in conversation['items'] if q['source_id'] not in finished_ids]
            if not pending:
                continue
            seed = args.seed + conversation['conversation_index']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.memory.data.copy_(initial_memory)
            model.initialized.copy_(initial_initialized)
            ids = tokenizer(conversation['context_text'], add_special_tokens=False,
                            truncation=False).input_ids
            chunks = chunk_right(ids)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            write_start = time.perf_counter()
            for chunk in chunks:
                tensor = torch.tensor([chunk], dtype=torch.long, device=device)
                # Official public API supports its own memory-aware mask construction.
                delta = model.inject_memory(tensor, update_memory=True)
                del delta, tensor
            torch.cuda.synchronize()
            write_seconds = time.perf_counter() - write_start
            conversation_record = {'conversation_index': conversation['conversation_index'],
                                   'seed': seed, 'context_tokens': len(ids),
                                   'chunk_tokens': list(map(len, chunks)), 'write_seconds': write_seconds,
                                   'write_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                   'write_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                   'memory_initialized': int(model.initialized.item()),
                                   'memory_bytes': model.memory.numel() * model.memory.element_size()}
            write_json(output / f"conversation_{conversation['conversation_index']:02d}.json", conversation_record)
            for item in pending:
                query = tokenizer(item['query_text'], add_special_tokens=False,
                                  truncation=False, return_tensors='pt').input_ids.to(device)
                before_ptr = model.memory.data_ptr()
                before_version = model.memory._version
                before_initialized = int(model.initialized.item())
                timer = TokenTimer()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                generated = model.generate(input_ids=query, max_new_tokens=48, num_beams=1,
                                           do_sample=False, temperature=1.0,
                                           min_new_tokens=1, eos_token_id=stop_ids,
                                           pad_token_id=tokenizer.pad_token_id or stop_ids[0],
                                           use_cache=True, streamer=timer)
                torch.cuda.synchronize()
                stop = time.perf_counter()
                token_ids = generated[0, query.shape[1]:].tolist()
                if model.memory.data_ptr() != before_ptr or model.memory._version != before_version or int(model.initialized.item()) != before_initialized:
                    raise RuntimeError('Question generation changed conversation memory state')
                if not token_ids or len(token_ids) > 48 or timer.first is None:
                    raise RuntimeError('Invalid generation length or timing')
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                free_after, _ = torch.cuda.mem_get_info()
                row = {'protocol_id': args.protocol_id, 'input_sha256': args.input_sha256,
                       'source_id': item['source_id'], 'id': item['source_id'],
                       'source_ordinal': item['source_ordinal'],
                       'source_conversation_index': conversation['conversation_index'],
                       'source_question_index': item['source_question_index'],
                       'arm': 'memoryllm_8b_chat', 'question_text': item['question_text'],
                       'raw_generated_text': text, 'prediction': text,
                       'generated_token_ids': token_ids, 'generated_tokens': len(token_ids),
                       'query_tokens': int(query.shape[1]), 'natural_generation_seconds': stop - start,
                       'natural_ttft_seconds': timer.first - start,
                       'natural_decode_tokens_per_second': ((len(token_ids) - 1) / (stop - timer.first)) if len(token_ids) > 1 and stop > timer.first else None,
                       'phase_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'phase_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'available_device_memory_after_bytes': free_after,
                       'full_device_peak_memory_bytes': None,
                       'memory_timing_boundary': 'Natural generation; allocator peaks include resident model and GPU initial-memory backup, not full-device peaks.',
                       'status': 'ok'}
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
                finished_ids.add(item['source_id'])
                del generated, query
                write_json(output / 'progress.json', {'status': 'RUNNING', 'completed': len(finished_ids),
                           'expected': 1986, 'last_source_id': item['source_id']})
    if finished_ids != all_ids:
        raise RuntimeError('Incomplete prediction set')
    execution['status'] = 'COMPLETE'
    execution['prediction_sha256'] = digest(predictions_path)
    execution['completed_predictions'] = len(finished_ids)
    write_json(output / 'execution.json', execution)
    write_json(output / 'progress.json', {'status': 'COMPLETE', 'completed': len(finished_ids), 'expected': 1986})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--protocol-id', default='locomo1986_memoryllm8bchat_plain48_sdpa_v1')
    parser.add_argument('--model-dir')
    parser.add_argument('--model-revision', default='a8dec23c6ef973ec2253d81a20a0a76228801cef')
    parser.add_argument('--source-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minimum-free-gib', type=float, default=30.0)
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    data = load_inputs(args.inputs, args.input_sha256)
    if not args.execute_gpu:
        print(json.dumps({'status': 'INPUT_VERIFIED_NO_MODEL_LOAD', 'conversations': 10,
                          'items': 1986, 'input_sha256': args.input_sha256}))
        return
    if not all([args.model_dir, args.source_dir, args.output_dir]):
        parser.error('GPU execution requires model-dir, source-dir and output-dir')
    execute(args, data)


if __name__ == '__main__':
    main()
