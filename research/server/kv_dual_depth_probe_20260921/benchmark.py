"""Server-only matched-graph H12/H24 dual-depth KV reconstruction benchmark."""
import collections
from concurrent.futures import ThreadPoolExecutor
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
MODEL = '/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
ADAPTER = '/srv/encbank/comem_infra_recheck_20260912/adapter'
COUNTS = [1, 2, 4, 8, 12]
WARMUPS, REPEATS = 2, 7
METHODS = ['h12_full', 'dual_serial', 'dual_streams', 'dual_threads', 'half_13_24', 'half_25_36']


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def status(phase, **kw):
    value = dict(phase=phase, at=datetime.datetime.now().astimezone().isoformat(), **kw)
    dump('status.json', value)
    print(json.dumps(value), flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires its own Slurm allocation'
    assert not (OUT / 'summary.json').exists(), 'Never overwrite completed results'
    status('IMPORTS')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from engine import Reader
    from peft import PeftModel
    from unittest.mock import patch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.manual_seed(20260921)
    assert torch.cuda.device_count() == 1
    prop = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    assert free > 80 * 2**30, 'Allocated device does not have sufficient free memory'
    torch.cuda.set_per_process_memory_fraction(64 * 2**30 / total, 0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    adapter_sha = hashlib.sha256((Path(ADAPTER) / 'adapter_model.safetensors').read_bytes()).hexdigest()
    assert adapter_sha == '1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13'
    environment = dict(job=os.environ['SLURM_JOB_ID'], pid=os.getpid(), host=platform.node(),
        gpu=prop.name, gpu_uuid=str(prop.uuid), cc=[prop.major, prop.minor], total_memory_bytes=total,
        free_before_load_bytes=free, allocator_cap_bytes=64*2**30,
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
        model=MODEL, adapter=ADAPTER, adapter_sha256=adapter_sha,
        dtype='BF16 backbone, original FP32 unmerged LoRA', depths=[12,24], layers=36,
        chunk_tokens=512, warmups=WARMUPS, repeats=REPEATS,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        attention='SDPA flash/efficient with explicit repeat-KV, same as previous probe',
        inference_only=True, model_files_downloaded=False)
    dump('environment.json', environment)
    status('LOAD_MODEL')
    begin = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao', return_value=None):
        wrapper = PeftModel.from_pretrained(model, ADAPTER, autocast_adapter_dtype=True)
    model = wrapper.base_model.model.eval()
    model.requires_grad_(False)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    assert model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
    assert model.config.num_key_value_heads == 8 and model.config.head_dim == 128
    reader = Reader(model, 12, tokenizer=tok)
    environment['model_load_seconds'] = time.perf_counter() - begin
    dump('environment.json', environment)
    docs = json.loads((ROOT / 'vendor' / 'workloads.json').read_text())
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    default = torch.cuda.current_stream()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='dual_depth')
    records, validations, preparation = [], [], []
    rng = random.Random(20260921)

    def fresh():
        return DynamicCache(config=model.config)

    def event():
        return torch.cuda.Event(enable_timing=True)

    def tensors(cache):
        return [t for i in range(12,36) for t in (cache.layers[i].keys, cache.layers[i].values)]

    def compare(actual, reference):
        max_abs, sq, refsq, numel, unequal = 0., 0., 0., 0, 0
        for x, y in zip(tensors(actual), tensors(reference)):
            assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
            d = x.float() - y.float()
            max_abs = max(max_abs, float(d.abs().max()))
            sq += float(d.square().sum()); refsq += float(y.float().square().sum()); numel += d.numel()
            unequal += int(torch.count_nonzero(x != y))
        result = dict(max_abs=max_abs, relative_rms=(sq/max(refsq,1e-30))**.5,
            rms=(sq/numel)**.5, unequal_elements=unequal, elements=numel)
        # Matched batching and attention graph: demand much tighter consistency
        # than the prior independent-batch-versus-serial experiment.
        assert result['relative_rms'] < 1e-5, result
        return result

    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with (OUT / 'measurements.jsonl').open('w', encoding='utf-8') as record_file:
            for di, doc in enumerate(docs[:3]):
                chosen = doc['queries'][0]['selected']
                ids = torch.tensor([doc['source'][i*512:(i+1)*512] for i in chosen], device='cuda')
                assert ids.shape == (12,512)
                all_h12 = reader.write_chunk(ids)
                sink_h12 = reader.write_chunk([model.config.bos_token_id])
                dump('input_'+str(di)+'.json', dict(document=di, selected=chosen,
                    selected_tokens_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()))
                for graph in ['independent_batch', 'joint_pack']:
                    for count in COUNTS:
                        status('MEASURE', document=di, graph=graph, chunks=count)
                        if graph == 'independent_batch':
                            h12 = all_h12[:count].contiguous()
                        else:
                            h12 = torch.cat([sink_h12, all_h12[:count].reshape(1,count*512,-1)], dim=1)
                        n = h12.shape[1]
                        positions = torch.arange(n,device='cuda')[None].expand(h12.shape[0],-1)
                        mask = (torch.arange(n,device='cuda')[None,:] <= positions[0,:,None])[None,None]
                        rope = reader.rotary_emb(h12, position_ids=positions)

                        def run(h, lo, hi, cache):
                            return reader._run_layers(h,slice(lo,hi),mask,positions,rope,
                                past_key_values=cache,use_cache=True)

                        # H24 is computed from the SAME graph, positions and batch
                        # as the baseline. This is an offline prerequisite, not free
                        # work included in or silently charged to reconstruction.
                        torch.cuda.synchronize()
                        prep_begin = time.perf_counter()
                        prep_cache = fresh()
                        h24 = run(h12,12,24,prep_cache)
                        torch.cuda.synchronize()
                        preparation.append(dict(document=di,graph=graph,chunks=count,
                            h24_prepare_wall_ms=(time.perf_counter()-prep_begin)*1000,
                            h12_bytes=h12.numel()*h12.element_size(),
                            additional_h24_bytes=h24.numel()*h24.element_size()))
                        del prep_cache

                        @torch.inference_mode()
                        def branch(which, cache, start, left, right):
                            with torch.cuda.device(0), torch.cuda.stream(streams[which]):
                                streams[which].wait_event(start)
                                left.record()
                                hidden = run(h12 if which == 0 else h24,
                                    12 if which == 0 else 24,24 if which == 0 else 36,cache)
                                right.record()
                                return hidden

                        def point(method):
                            torch.cuda.synchronize()
                            start, end = event(), event()
                            edges = [(event(),event()), (event(),event())]
                            begin = time.perf_counter()
                            start.record(default)
                            a = fresh()
                            extra = {}
                            if method == 'h12_full':
                                hidden = run(h12,12,36,a)
                            elif method == 'half_13_24':
                                hidden = run(h12,12,24,a)
                            elif method == 'half_25_36':
                                hidden = run(h24,24,36,a)
                            else:
                                b = fresh()
                                if method == 'dual_serial':
                                    first = run(h12,12,24,a)
                                    hidden = run(h24,24,36,b)
                                elif method == 'dual_streams':
                                    first = branch(0,a,start,*edges[0])
                                    hidden = branch(1,b,start,*edges[1])
                                elif method == 'dual_threads':
                                    fa = pool.submit(branch,0,a,start,*edges[0])
                                    fb = pool.submit(branch,1,b,start,*edges[1])
                                    first, hidden = fa.result(), fb.result()
                                else:
                                    raise AssertionError(method)
                                if method in ['dual_streams','dual_threads']:
                                    for _, finish in edges:
                                        default.wait_event(finish)
                                # Metadata-only union: each layer owns its tensors.
                                # No concatenate/copy of full per-layer KV required.
                                for i in range(24,36):
                                    a.layers[i] = b.layers[i]
                                del first, b
                            end.record(default)
                            end.synchronize()
                            wall_ms = (time.perf_counter()-begin)*1000
                            if method in ['dual_streams','dual_threads']:
                                intervals = [(start.elapsed_time(x),start.elapsed_time(y)) for x,y in edges]
                                extra = dict(branch_intervals_ms=intervals,
                                    branch_envelope_overlap_ms=max(0.,min(y for x,y in intervals)-max(x for x,y in intervals)),
                                    note='Branch envelope overlap includes stream queueing; not measured simultaneous kernel execution')
                            del hidden
                            return a,dict(wall_ms=wall_ms,cuda_ms=start.elapsed_time(end),**extra)

                        baseline, _ = point('h12_full')
                        for method in ['dual_serial','dual_streams','dual_threads']:
                            actual, _ = point(method)
                            validations.append(dict(document=di,graph=graph,chunks=count,method=method,
                                **compare(actual,baseline)))
                            del actual
                        del baseline
                        dump('validation.json',dict(passed=True,relative_rms_limit=1e-5,checks=validations,
                            scope='KV equality within the same attention graph and matched batching only'))
                        dump('preparation.json',preparation)
                        for iteration in range(-WARMUPS,REPEATS):
                            order = METHODS.copy(); rng.shuffle(order)
                            for method in order:
                                cache, measured = point(method)
                                del cache
                                if iteration >= 0:
                                    row = dict(document=di,graph=graph,chunks=count,method=method,
                                        repeat=iteration,**measured)
                                    records.append(row)
                                    record_file.write(json.dumps(row)+'\n');record_file.flush()
                        del h12,h24,mask,rope,positions
                        gc.collect()
                del all_h12,sink_h12,ids
                gc.collect();torch.cuda.empty_cache()
    pool.shutdown(wait=True)

    def stats(values):
        values = sorted(values)
        return dict(n=len(values),median=statistics.median(values),min=values[0],max=values[-1],
            p95=values[min(len(values)-1,int(.95*(len(values)-1)))])

    grouped = collections.defaultdict(list)
    for row in records:grouped[(row['graph'],row['chunks'],row['method'])].append(row)
    cells = [dict(graph=graph,chunks=count,method=method,
        wall_ms=stats([r['wall_ms'] for r in rows]),cuda_ms=stats([r['cuda_ms'] for r in rows]))
        for (graph,count,method),rows in sorted(grouped.items())]
    dump('summary.json',dict(complete=True,environment=environment,cells=cells,
        formal_observations=len(records),validation_checks=len(validations),
        reconstruction_boundary='Resident GPU H12/H24 through layer blocks with retained KV and metadata union; excludes retrieval, transfer, initial hidden writing, query and LM head',
        methods=dict(h12_full='H12 -> all 24 upper layers on default stream',
            dual_serial='H12 -> 13..24 and cached H24 -> 25..36, serial control',
            dual_streams='Two CUDA streams, one host submission thread',
            dual_threads='Two CUDA streams, two persistent host submission threads',
            half_13_24='First 12-layer segment alone',half_25_36='Second 12-layer segment alone'),
        independent_scope='Each batch row is an independent 512-token chunk, using chunk-local positions; not original cross-chunk attention',
        joint_scope='Ordered selected chunks plus sink jointly causal; H24 is valid only for that precise prefix, positions and batching; does not establish reusable per-chunk H24 for changing retrieval',
        timing_scope='CUDA-synchronized wall plus CUDA events; randomized method order, 2 warmups and 7 observations per cell per document, 3 saved PG19 inputs, one GPU process',
        hot_buffer_scope='No real agent cache-hit trace, eviction policy or current-generated-chunk insertion measured',
        source_manifest=json.loads((ROOT/'source_manifest.json').read_text())))
    status('COMPLETE',formal_observations=len(records),validation_checks=len(validations))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        dump('failure.json',dict(traceback=traceback.format_exc(),at=datetime.datetime.now().astimezone().isoformat()))
        raise
