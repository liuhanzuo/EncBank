"""Same-GPU Transformers Dense/Encbank inference comparison, no production edits."""
import copy
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

H = Path(__file__).resolve().parent


def save(name, value):
    with (H / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def event(**value):
    value['epoch'] = time.time()
    print(json.dumps(value), flush=True)


def main():
    from common import MODELS, load_model, load_state, tokenizer, torch
    import transformers
    from hybrid_reader import HybridReader
    from runtime_identity import resolve_configs
    from transformers.cache_utils import DynamicCache
    from batch_cache import merge_caches, decode_layers

    assert os.environ.get('ENCBANK_ISOLATED_KERNEL_PILOT') == '1'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(4203)
    torch.backends.cuda.enable_cudnn_sdp(False)
    free, total = torch.cuda.mem_get_info()
    save('gpu_preflight.json', dict(epoch=time.time(), hostname=os.uname().nodename,
        gpu=str(torch.cuda.get_device_properties(0)), free_bytes=free, total_bytes=total,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'),
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv'], text=True)))
    assert total - free < 2 * 2**30, 'Allocated GPU already occupied'
    plan = json.loads((H / 'baseline_plan.json').read_text())
    protocol = json.loads((H / 'protocol.json').read_text())
    identity, cfg = resolve_configs(plan, MODELS[1])
    assert hashlib.sha256(Path(plan['adapter_path']).read_bytes()).hexdigest() == plan['adapter_sha256']
    started = time.perf_counter()
    model = load_model(cfg)
    reader = HybridReader(model, cfg['j'])
    reader.attach()
    ckpt = torch.load(plan['adapter_path'], map_location='cpu', weights_only=False)
    resolve_configs(plan, identity, ckpt)
    load_state(reader, ckpt, identity)
    del ckpt
    model.requires_grad_(False)
    tok = tokenizer(cfg)
    save('model_ready.json', dict(epoch=time.time(), load_seconds=time.perf_counter()-started,
        torch=torch.__version__, transformers=transformers.__version__, model=cfg,
        adapter_sha256=plan['adapter_sha256'], job_id=os.environ['SLURM_JOB_ID'], benchmark_attempts=0,
        config_layer_types=reader.config.layer_types, attention=reader.config._attn_implementation))
    # Remove the wrappers altogether for stock Dense; restoring wrappers changes
    # only this fresh isolated process, never the installed library or production.
    targets = []
    for name, wrapped in reader.modules.items():
        parent, _, child = name.rpartition('.')
        targets.append((reader.core.get_submodule(parent), child, wrapped))

    def select(method):
        for parent, child, wrapped in targets:
            setattr(parent, child, wrapped.base if method == 'dense_hf' else wrapped)

    def hashed(x):
        return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    words = tok.encode('The experiment keeps each document memory immutable. A query reads selected states and then generates an answer. ', add_special_tokens=False)
    stream = words * 2000
    chunks_ids = [stream[i:i+512] for i in range(48)]
    methods = protocol['methods']
    results = []
    with torch.inference_mode():
        # Validate that stock forward and the existing manual full-layer path
        # agree on a short real input before timing either method.
        select('dense_hf')
        ids = reader.tensor(stream[:64])
        stock = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        manual = reader.logits(reader.full_hidden(stream[:64]))
        parity = dict(max_abs=float((stock-manual).abs().max()),
                      greedy_equal=bool(torch.equal(stock.argmax(-1), manual.argmax(-1))))
        assert torch.allclose(stock, manual, atol=2e-3, rtol=2e-3), parity
        del stock, manual, ids
        save('stock_forward_parity.json', parity)

        for batch, chunks in [(1, 12), (8, 12), (1, 48), (8, 48)]:
            case = f'b{batch}_h{chunks}'
            event(stage='prepare', case=case)
            histories = sum(chunks_ids[:chunks], [])
            queries = [stream[100+row:228+row] for row in range(batch)]
            select('encbank_hf')
            torch.cuda.synchronize()
            begin = time.perf_counter()
            memories = [reader.write(ids) for ids in chunks_ids[:chunks]]
            torch.cuda.synchronize()
            write_seconds = time.perf_counter() - begin
            h_sha = [hashed(m) for m in memories]
            initial, prefill = {}, {}

            for method in methods:
                select(method)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                lower_rows, upper_rows = [], []
                for query in queries:
                    if method.startswith('dense'):
                        output = model(input_ids=reader.tensor(histories+query), use_cache=True, logits_to_keep=1)
                        upper_rows.append(output.past_key_values)
                        del output
                    else:
                        lo, up = DynamicCache(config=reader.config), DynamicCache(config=reader.config)
                        qh = reader.layers(reader.core.embed_tokens(reader.tensor(query)), 0, reader.j, cache=lo)
                        out = reader.layers(torch.cat(memories+[qh], dim=1), reader.j, reader.L, cache=up)
                        # Include the first-token head in both prefill timings.
                        first = reader.logits(out)
                        lower_rows.append(lo)
                        upper_rows.append(up)
                        del qh, out, first, lo, up
                length = len(histories) + 128
                up, pad = merge_caches(upper_rows, [length]*batch, reader.config,
                    0 if method.startswith('dense') else reader.j, reader.L, consume=True)
                cache = dict(up=up, pad=pad, pos=torch.full((batch,), length, device='cuda', dtype=torch.long))
                if method == 'encbank_hf':
                    lo, lpad = merge_caches(lower_rows, [128]*batch, reader.config, 0, reader.j, consume=True)
                    cache.update(lo=lo, lpad=lpad, qpos=torch.full((batch,), 128, device='cuda', dtype=torch.long))
                initial[method] = cache
                del lower_rows, upper_rows, cache, up, pad
                if method == 'encbank_hf':
                    del lo, lpad
                torch.cuda.synchronize()
                prefill[method] = time.perf_counter() - begin
                event(stage='prefill_done', case=case, method=method, seconds=prefill[method])

            tokens = [torch.tensor([[words[(i+row)%len(words)]] for row in range(batch)], device='cuda') for i in range(protocol['teacher_forced_decode_steps'])]

            def step(method, cache, token):
                if method.startswith('dense'):
                    output = model(input_ids=token, past_key_values=cache['up'],
                        position_ids=cache['pos'][None, :, None].expand(4, -1, -1),
                        use_cache=True, logits_to_keep=1)
                    logits = output.logits
                else:
                    hidden = reader.core.embed_tokens(token)
                    hidden = decode_layers(reader, hidden, 0, reader.j, cache['lo'], cache['qpos'], cache['lpad'])
                    hidden = decode_layers(reader, hidden, reader.j, reader.L, cache['up'], cache['pos'], cache['pad'])
                    logits = reader.logits(hidden)
                    cache['qpos'] += 1
                cache['pos'] += 1
                return logits

            def run(method, count, greedy=False):
                select(method)
                cache = copy.deepcopy(initial[method])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated_before = torch.cuda.memory_allocated()
                begin = time.perf_counter()
                outputs = []
                token = tokens[0]
                for i in range(count):
                    logits = step(method, cache, token if greedy else tokens[i])
                    token = logits.argmax(-1)
                    outputs.append(token)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                token_ids = torch.cat(outputs, dim=1)
                finite = bool(torch.isfinite(logits).all())
                assert finite, method
                measurement = dict(seconds=seconds, steps=count, batch=batch,
                    per_sequence_tps=count/seconds, aggregate_tps=batch*count/seconds,
                    output_argmax_sha256=hashed(token_ids), finite_last_logits=finite,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    allocated_before_bytes=allocated_before,
                    peak_increment_bytes=torch.cuda.max_memory_allocated()-allocated_before)
                del logits, outputs, token_ids, token, cache
                return measurement

            for method in methods:
                run(method, 8)
            timings = {method: [] for method in methods}
            for order in protocol['orders']:
                for method in order:
                    measurement = run(method, protocol['teacher_forced_decode_steps'])
                    timings[method].append(measurement)
                    event(stage='timing', case=case, method=method, **measurement)
            greedy = {method: run(method, protocol['greedy_decode_steps'], True) for method in methods}
            medians = {method: statistics.median(x['seconds'] for x in timings[method]) for method in methods}
            row = dict(case=case, batch=batch, history_chunks=chunks, history_tokens=len(histories),
                query_tokens=128, fixed_steps=protocol['teacher_forced_decode_steps'],
                write_seconds_for_unique_history=write_seconds, prefill_seconds=prefill,
                prefill_policy='sequential per-row prefill then merge for every method; shared Encbank H write excluded and separately recorded',
                input_sha256=hashlib.sha256(json.dumps(dict(history=histories, queries=queries)).encode()).hexdigest(),
                timing_samples=timings, median_seconds=medians,
                per_sequence_tps={m:protocol['teacher_forced_decode_steps']/v for m,v in medians.items()},
                aggregate_tps={m:batch*protocol['teacher_forced_decode_steps']/v for m,v in medians.items()},
                encbank_vs_dense_speed=medians['dense_hf']/medians['encbank_hf'],
                dense_lora_vs_dense_speed=medians['dense_hf']/medians['dense_hf_lora'],
                greedy_decode=greedy, H_unchanged=[hashed(m) for m in memories]==h_sha,
                timing_memory_note='absolute memory includes the three immutable prepared comparison caches; not standalone serving footprint')
            assert row['H_unchanged']
            save(f'case_{case}.json', row)
            results.append(row)
            event(stage='case_done', case=case, per_sequence_tps=row['per_sequence_tps'])
            del initial, memories, tokens
            gc.collect()
            torch.cuda.empty_cache()
    save('model_probe_result.json', dict(status='MEASURED', cases=results,
        job_id=os.environ['SLURM_JOB_ID'], ended_epoch=time.time(), benchmark_attempts=0,
        production_approved=False, protocol=protocol))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save('model_probe_failure.json', dict(epoch=time.time(), traceback=traceback.format_exc()))
        raise
