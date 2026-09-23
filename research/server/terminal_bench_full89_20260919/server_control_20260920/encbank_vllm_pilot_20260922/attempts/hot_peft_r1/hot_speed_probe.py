"""Isolated, fixed-trace throughput measurement of the frozen Hot24 runtime."""
import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    case = next(x for x in protocol['runs'] if x['id'] == args.run_id)
    out = ROOT / 'results' / args.run_id
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import peft
    import transformers
    import common
    import hybrid_hot as hh
    from model_setup import load, tokens
    from encbank_peft import load_encbank_peft, file_sha
    from hybrid_reader import HybridReader

    common.ROOT = out
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert os.environ.get('ENCBANK_ISOLATED_KERNEL_PILOT') == '1'
    assert plan['fixed_decode_linear_rows'] == 32 and plan['row_decode_sdpa']
    torch.manual_seed(20260922)
    variant = case['variant']
    began = time.perf_counter()
    model, reader, tok, _ = load(plan, adapter=variant == 'hot_legacy')
    merge_seconds = 0
    if variant in ['hot_merged', 'cold_merged']:
        merge_start = hh.sync()
        model, owner, targets = load_encbank_peft(model, protocol['peft_adapter'], mode='merged_bf16')
        module_count = len(targets)
        assert module_count == 333
        del owner, targets
        reader = HybridReader(model, 21)
        assert not any('lora_' in name for name, _ in model.named_parameters())
        assert not any(type(m).__name__ in ['LoRALinear', 'LegacyArithmetic'] for m in model.modules())
        merge_seconds = hh.sync() - merge_start
    else:
        module_count = 0 if variant == 'dense' else len(reader.modules)
    model.requires_grad_(False)
    write(out / 'model_ready.json', dict(epoch=time.time(), job_id=os.environ['SLURM_JOB_ID'],
        host=os.uname().nodename, variant=variant, torch=torch.__version__, transformers=transformers.__version__,
        peft=peft.__version__, gpu=str(torch.cuda.get_device_properties(0)),
        load_seconds=time.perf_counter()-began, merge_seconds=merge_seconds, adapter_modules=module_count,
        merge_method='PEFT merge_and_unload(safe_merge=True)' if merge_seconds else None,
        runtime_weights_offloaded=False, merged_checkpoint_saved=False))

    request_path = Path(protocol['recorded_request'])
    assert file_sha(request_path) == protocol['recorded_request_sha256']
    messages = json.loads(request_path.read_text())['messages']
    corpus = tokens(tok, messages)
    trace = tok.encode('Inspect the program and its tests. Preserve the public interface. '
        'Run a small deterministic example, compare the output, and check the boundary conditions.\n',
        add_special_tokens=False)
    trace = (trace * ((1200 // len(trace)) + 1))[:1200]
    extension = tok.encode('\nTool output: the previous command completed. Continue checking the remaining cases.\n',
        add_special_tokens=False)
    extension = (extension * 10)[:64]
    assert len(corpus) > case['history_tokens'] + 600
    arm = 'dense' if variant == 'dense' else ('cold' if variant == 'cold_merged' else 'hot')
    capacity = 24 if arm == 'hot' else 0
    originals = {}
    original_sample = hh.sample

    def fixed_trace_with_sampling(logit, generator, temperature=1., top_p=.95, top_k=20):
        # Preserve the real sampling work; replay tokens solely to equalize paths.
        original_sample(logit, generator, temperature, top_p, top_k)
        cursor = originals[id(generator)]
        token = trace[cursor]
        originals[id(generator)] = cursor + 1
        return token

    hh.sample = fixed_trace_with_sampling

    def tensor_sha(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def bank_proof(session):
        return {repr(k): (value[1], tensor_sha(value[0])) for k, value in session.bank.items()}

    def one_call(sessions, inputs, steps, call_index):
        torch.cuda.reset_peak_memory_stats()
        started_epoch = time.time()
        begin = hh.sync()
        requests = [hh.Request(s, ids, seed=20260922+i+call_index*100, topk=12)
                    for i, (s, ids) in enumerate(zip(sessions, inputs))]
        after_prefill = hh.sync()
        for x in requests:
            originals[id(x.generator)] = 0
        # CPU evidence collection is explicitly excluded from throughput timers.
        proof_start = hh.sync()
        prefill_logits_cpu = torch.cat([x.logits for x in requests]).detach().cpu()
        h_before = bank_proof(sessions[0]) if arm != 'dense' else {}
        proof_seconds = hh.sync()-proof_start
        cohorts = []
        refresh_wall = 0.
        decode_begin = hh.sync()
        while len(requests[0].generated) < steps:
            assert all(len(x.generated) == len(requests[0].generated) and x.status is None for x in requests)
            cohorts.append(hh.quantum(requests, stop=set(), context=262144,
                steps=min(32, steps-len(requests[0].generated)), temperature=1.))
            if requests[0].refresh_needed() and len(requests[0].generated) < steps:
                before = hh.sync()
                for x in requests:
                    x.lower = x.upper = x.logits = None
                    x.prefill()
                refresh_wall += hh.sync()-before
        after_decode = hh.sync()
        first_generated = min(x.first for x in requests)
        cache_save_begin = hh.sync()
        if arm == 'dense':
            # Same independent saved-prefix clone as the actual worker.
            for s, x in zip(sessions, requests):
                s.dense_cache = hh.row_cache(x.upper, reader.config, 0, reader.L, 0, 0)
                s.dense_ids = list(x.ids)
        cache_save_seconds = hh.sync()-cache_save_begin
        end_epoch = time.time()
        end_logits_cpu = torch.cat([x.logits for x in requests]).detach().cpu()
        torch.save(dict(prefill=prefill_logits_cpu, end=end_logits_cpu), out/f'call_{call_index}_logits.pt')
        h_after = bank_proof(sessions[0]) if arm != 'dense' else {}
        surviving = set(h_before) & set(h_after)
        assert all(h_before[k] == h_after[k] for k in surviving), 'Existing H mutated'
        expected = digest(trace[:steps])
        assert all(digest(x.generated) == expected for x in requests)
        seconds = after_decode-decode_begin
        prefill_seconds = after_prefill-begin
        data = dict(call_index=call_index, started_epoch=started_epoch, ended_epoch=end_epoch,
            input_lengths=[len(x.initial) for x in requests], input_sha256=[digest(x.initial) for x in requests],
            output_tokens_per_row=steps, output_sha256=expected, batch=len(requests),
            prefill_wall_seconds=prefill_seconds, decode_and_refresh_wall_seconds=seconds,
            decode_cohort_wall_seconds=sum(c['seconds'] for c in cohorts), refresh_wall_seconds=refresh_wall,
            cache_save_seconds=cache_save_seconds, measured_model_wall_seconds=prefill_seconds+seconds+cache_save_seconds,
            first_token_after_all_admitted_seconds=first_generated-(after_prefill+proof_seconds),
            row_tokens_per_second=steps/seconds, aggregate_tokens_per_second=steps*len(requests)/seconds,
            end_to_end_aggregate_tokens_per_second=steps*len(requests)/(prefill_seconds+seconds+cache_save_seconds),
            cohorts=cohorts, row_times=[dict(x.time) for x in requests], row_events=[x.events for x in requests],
            hot_stats=[dict(s.pool.stats) for s in sessions], hot_bytes=[s.pool.bytes for s in sessions],
            h_bytes=[s.hbytes() for s in sessions], h_proof_scope='row0 existing H identities; full tensor SHA before and after',
            existing_h_survivors=len(surviving), existing_h_unchanged=True,
            allocated_gib=torch.cuda.memory_allocated()/2**30, reserved_gib=torch.cuda.memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write(out/f'call_{call_index}.json', data)
        next_inputs = [list(x.ids)+extension for x in requests]
        for x in requests:
            originals.pop(id(x.generator))
        return data, next_inputs

    with torch.inference_mode():
        # Warm the same scheduler and sampling path, then discard all warmup state.
        warm = [hh.Session(reader, arm, capacity, f'warmup{i}') for i in range(case['batch'])]
        warm_rows = [hh.Request(s, corpus[:1024+1+96+(i%8)*17], 7+i) for i,s in enumerate(warm)]
        originals.update({id(x.generator): 0 for x in warm_rows})
        hh.quantum(warm_rows, set(), steps=32)
        originals.clear()
        del warm_rows, warm
        gc.collect()
        torch.cuda.empty_cache()
        sessions = [hh.Session(reader, arm, capacity, f'row{i}') for i in range(case['batch'])]
        # Shared recorded text, differing tail lengths; each session computes/owns its own H and cache.
        inputs = [corpus[:case['history_tokens']+1+96+(i%8)*17] for i in range(case['batch'])]
        write(out/'input_manifest.json', dict(recorded_request=str(request_path),
            recorded_request_sha256=protocol['recorded_request_sha256'], corpus_tokens=len(corpus),
            input_lengths=[len(x) for x in inputs], input_sha256=[digest(x) for x in inputs],
            trace_sha256=digest(trace), sampling='temperature1/top_p.95/top_k20 evaluated; common trace token consumed',
            scope='fixed-trace replay of recorded prompt excerpts; no task environments/verifiers or benchmark scores'))
        first, continuation = one_call(sessions, inputs, 544, 1)
        second, _ = one_call(sessions, continuation, 64, 2)
        write(out/'result.json', dict(case=case, status='PASS', calls=[first, second],
            numerical_boundary='PEFT BF16 merging is an authorized numerical change, not bitwise legacy equivalence.',
            total_measured_seconds=sum(x['measured_model_wall_seconds'] for x in [first,second])))
        sessions.clear()
    hh.sample = original_sample
    print(json.dumps(dict(run_id=args.run_id,status='PASS',first_tps=first['aggregate_tokens_per_second'],
        continuation_tps=second['aggregate_tokens_per_second'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
