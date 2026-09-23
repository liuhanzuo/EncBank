"""Fresh isolated Encbank probe: ragged rows, real k12/k48 sizes, warmed A/B/A."""
import copy, gc, hashlib, json, os, statistics, time, traceback
from pathlib import Path

H = Path(__file__).resolve().parent

def save(name, value):
    with (H / name).open('x') as f:
        json.dump(value, f, indent=2)

def main():
    from common import MODELS, load_model, load_state, tokenizer, torch
    from hybrid_reader import HybridReader
    from runtime_identity import resolve_configs
    from transformers.cache_utils import DynamicCache
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf
    from batch_cache import merge_caches, decode_layers
    from vllm_gdn_adapter import VllmGdnDecode
    from vendor_fused_recurrent import fused_recurrent_gated_delta_rule as vendor
    from native_layout_adapter import NativeLayoutGdn
    assert os.environ.get('ENCBANK_ISOLATED_KERNEL_PILOT') == '1'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(4203)
    torch.backends.cuda.enable_cudnn_sdp(False)
    free, total = torch.cuda.mem_get_info()
    save('gpu_preflight.json', dict(epoch=time.time(), hostname=os.uname().nodename,
        gpu=str(torch.cuda.get_device_properties(0)), free_bytes=free, total_bytes=total,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS')))
    assert total-free < 2*2**30, 'Allocated GPU is already occupied'
    plan = json.loads((H/'baseline_plan.json').read_text())
    identity, cfg = resolve_configs(plan, MODELS[1])
    assert hashlib.sha256(Path(plan['adapter_path']).read_bytes()).hexdigest() == plan['adapter_sha256']
    from native_microcheck import check_native
    with torch.inference_mode():
        save('native_microcheck.json', check_native(hf.torch_recurrent_gated_delta_rule))
    started = time.time()
    model = load_model(cfg)
    reader = HybridReader(model, cfg['j'])
    reader.attach()
    ckpt = torch.load(plan['adapter_path'], map_location='cpu', weights_only=False)
    resolve_configs(plan, identity, ckpt)
    load_state(reader, ckpt, identity)
    del ckpt
    model.requires_grad_(False)
    tok = tokenizer(cfg)
    original = hf.torch_recurrent_gated_delta_rule
    variants = dict(hf=original, native=NativeLayoutGdn(), native_norm=NativeLayoutGdn(fuse_hf_norm=True))
    save('model_ready.json', dict(epoch=time.time(), load_seconds=time.time()-started,
        torch=torch.__version__, model=cfg, adapter_sha256=plan['adapter_sha256'],
        job_id=os.environ['SLURM_JOB_ID'], benchmark_attempts=0))
    words = tok.encode('The experiment keeps each document memory immutable. A query reads selected states and then generates an answer. ', add_special_tokens=False)
    segment = (words*100)[:512]
    results = []
    with torch.inference_mode():
        memory = reader.write(segment)
        def hashed(x):
            return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        h_sha = hashed(memory)
        def prepare(batch, chunks):
            ls, us, qlens, ulens = [], [], [], []
            for row in range(batch):
                query = (words*30)[row:row+96+4*row]
                history = max(1, chunks-row%3)
                lower, upper = DynamicCache(config=reader.config), DynamicCache(config=reader.config)
                qh = reader.layers(reader.core.embed_tokens(reader.tensor(query)), 0, reader.j, cache=lower)
                reader.layers(torch.cat([memory]*history+[qh], dim=1), reader.j, reader.L, cache=upper)
                ls.append(lower); us.append(upper)
                qlens.append(len(query)); ulens.append(512*history+len(query))
            lower, lp = merge_caches(ls, qlens, reader.config, 0, reader.j, consume=True)
            upper, up = merge_caches(us, ulens, reader.config, reader.j, reader.L, consume=True)
            return (lower, upper, lp, up, torch.tensor(qlens, device='cuda'), torch.tensor(ulens, device='cuda'))
        def step(cache, token):
            lower, upper, lp, up, qp, upos = cache
            h = reader.core.embed_tokens(token)
            h = decode_layers(reader, h, 0, reader.j, lower, qp, lp)
            h = decode_layers(reader, h, reader.j, reader.L, upper, upos, up)
            logits = reader.logits(h)
            qp += 1; upos += 1
            return logits
        def run(name, initial, tokens):
            hf.torch_recurrent_gated_delta_rule = variants[name]
            cache = copy.deepcopy(initial)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = [step(cache, t) for t in tokens]
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            result = torch.cat(outputs, dim=1)
            del outputs, cache
            return result, elapsed
        def compare(actual, expected):
            a, e = actual.float(), expected.float()
            delta = (a-e).abs()
            logp, logq = e.log_softmax(-1), a.log_softmax(-1)
            kl = (logp.exp()*(logp-logq)).sum(-1)
            agreement = float((actual.argmax(-1)==expected.argmax(-1)).float().mean())
            return dict(max_logit_abs_error=float(delta.max()), mean_logit_abs_error=float(delta.mean()),
                greedy_token_agreement=agreement, max_kl=float(kl.max()), mean_kl=float(kl.mean()),
                numerical_gate_pass=agreement==1 and float(delta.max())<=.125)

        for batch, chunks in [(1,12), (8,12), (8,48)]:
            hf.torch_recurrent_gated_delta_rule = original
            start = time.perf_counter()
            initial = prepare(batch, chunks)
            tokens = [torch.tensor([[words[(i+row)%len(words)]] for row in range(batch)], device='cuda') for i in range(32)]
            torch.cuda.synchronize()
            prep_s = time.perf_counter()-start
            # Warm every shape/backend, including baseline. Never time cache copies.
            for name in variants:
                warm, _ = run(name, initial, tokens[:4])
                del warm
            expected, _ = run('hf', initial, tokens)
            times = {name: [] for name in variants}
            checks = {name: [] for name in variants}
            for order in [('hf','native','native_norm'), ('native_norm','native','hf')]:
                for name in order:
                    actual, seconds = run(name, initial, tokens)
                    times[name].append(seconds)
                    checks[name].append(compare(actual, expected))
                    del actual
            row = dict(batch=batch, max_history_chunks=chunks, max_history_tokens=512*chunks,
                distinct_ragged_queries=True, steps=32, prefill_seconds=prep_s, timing_seconds=times,
                median_seconds={n: statistics.median(v) for n,v in times.items()}, checks=checks,
                H_unchanged=hashed(memory)==h_sha, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            row['speedup'] = {n:row['median_seconds']['hf']/row['median_seconds'][n] for n in variants}
            save(f'case_b{batch}_h{chunks}.json', row)
            print(json.dumps(row), flush=True)
            results.append(row)
            del expected, initial, tokens
            gc.collect(); torch.cuda.empty_cache()

        # Inspect actual first-step GDN inputs, always returning the HF output.
        hf.torch_recurrent_gated_delta_rule = original
        initial = prepare(1,12)
        diagnostics = []
        def shadow(*args, **kwargs):
            expected, state = original(*args, **kwargs)
            entry = dict(call_index=len(diagnostics))
            for name in ['native', 'native_norm']:
                actual, other = variants[name](*args, **kwargs)
                entry[name] = dict(output_max_abs=float((actual.float()-expected.float()).abs().max()),
                    output_mean_abs=float((actual.float()-expected.float()).abs().mean()),
                    state_max_abs=float((other-state).abs().max()), state_max_scale=float(state.abs().max()))
            diagnostics.append(entry)
            return expected, state
        hf.torch_recurrent_gated_delta_rule = shadow
        step(initial, torch.tensor([[words[0]]], device='cuda'))
        save('real_gdn_diagnostics.json', diagnostics)
        hf.torch_recurrent_gated_delta_rule = original
    save('model_probe_result.json', dict(status='MEASURED', cases=results,
        job_id=os.environ['SLURM_JOB_ID'], ended_epoch=time.time(), benchmark_attempts=0,
        production_approved=False, numerical_gate='.125 maximum logit error and all greedy choices equal; unchanged from r5',
        limitation='Synthetic teacher-forced queries; not task accuracy, free-generation equivalence, or full vLLM integration.'))

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save('model_probe_failure.json', dict(epoch=time.time(), traceback=traceback.format_exc()))
        raise
