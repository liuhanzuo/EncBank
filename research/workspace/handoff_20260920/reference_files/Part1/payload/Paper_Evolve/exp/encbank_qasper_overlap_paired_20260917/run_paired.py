"""Fresh paired Qasper answers and full-document Write timings on RTX5090."""
import argparse
import gc
import gzip
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'encbank_overlap_20260913'
sys.path.insert(0, str(PRIOR))
from common import Encbank, attach, check_paths, decode, dump, parts, score, selection, timed_stamp, torch, write_one
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/srv/encbank/legacy_workspace/models/Qwen3-8B')
    parser.add_argument('--adapter', default=str(HERE.parent / 'encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(42)
    out = HERE / 'results'
    out.mkdir(exist_ok=False)
    dump(out / 'progress.json', dict(phase='WAITING_FOR_GPU', pid=os.getpid(), examples=0, planned=200))
    sys.path.insert(0, str(HERE.parent))
    from gpu_gate import acquire_gpu
    admission = acquire_gpu(need_gb=23, cap_gb=28, idle_slack_gb=5, poll=10,
                            max_wait=6*3600, tag='encbank_qasper_overlap_paired_20260917')
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    dump(out / 'progress.json', dict(phase='LOADING', pid=os.getpid(), examples=0, planned=200))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
             attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    wrapper, model = attach(base, args.adapter)
    checks = check_paths(wrapper, model, tokenizer)
    dump(out / 'correctness.json', checks)
    with gzip.open(PRIOR / 'samples.jsonl.gz', 'rt', encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rows = [row for row in rows if row['cell'] == 'qasper']
    assert len(rows) == 200 and len({row['id'] for row in rows}) == 200
    assert all(row['budget'] == 128 for row in rows)
    with gzip.open(out / 'paired_inputs.jsonl.gz', 'wt', encoding='utf-8') as target:
        for row in rows: target.write(json.dumps(row, ensure_ascii=False)+'\n')
    dump(out / 'metadata.json', dict(model=args.model, adapter=args.adapter,
         adapter_config=json.loads((Path(args.adapter) / 'adapter_config.json').read_text()),
         j=12, widths=[0,32], bos_token_id=tokenizer.bos_token_id, max_new_tokens=128,
         torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
         gpu=torch.cuda.get_device_name(0), cap_bytes=28_000_000_000, admission=admission,
         old_predictions_reused=False, samples=200, expected_predictions=400,
         warmups_per_arm_example=1, formal_write_repetitions=3,
         inference_source=str(PRIOR / 'Encbank'), protocol='PROTOCOL_zh.md',
         timing='Synchronized wall time, full-document lower-layer encoding plus CPU-pinned store'))
    reader = Encbank(model, 12, tokenizer=tokenizer)
    rng = random.Random(20260917)
    started = time.monotonic()
    outputs = cost_count = failures = 0
    def run_one(row, chunks, query, selected, source, width, rep):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = timed_stamp()
        residuals = [write_one(reader, source, i, width).to('cpu').pin_memory()
                     for i in range(len(chunks))]
        end = timed_stamp()
        persistent = sum(value.numel()*value.element_size() for value in residuals)
        assert sum(value.shape[1] for value in residuals) == len(source)
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        assert allocated <= 28_000_000_000 and reserved <= 28_000_000_000
        cost = dict(id=row['id'], arm='w'+str(width), width=width, rep=rep, warmup=rep==-1,
                    status='ok', document_write_ms=1000*(end-begin),
                    source_positions=len(source), chunks=len(chunks),
                    processed_write_positions=sum(min(512, len(source)-i*512)+min(width,i*512) for i in range(len(chunks))),
                    persistent_bytes=persistent, peak_allocated_bytes=allocated,
                    peak_reserved_bytes=reserved, selected=selected)
        prediction = None
        if rep == 0:
            states = [residuals[i].to('cuda', non_blocking=True) for i in selected]
            sink = reader.write_chunk([tokenizer.bos_token_id])
            generated, _, _ = decode(reader, sink, states, query, tokenizer.eos_token_id, 128)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            prediction = dict(id=row['id'], cell='qasper', arm='w'+str(width),
                 generated_ids=generated, prediction=text, score=score(row, text),
                 selected=selected, cached_positions=sum(value.shape[1] for value in states),
                 max_new_tokens=128, status='ok')
            assert torch.cuda.max_memory_reserved() <= 28_000_000_000
        return cost, prediction
    with (out / 'costs.jsonl').open('w', encoding='utf-8') as costs, \
         (out / 'predictions.jsonl').open('w', encoding='utf-8') as predictions:
        for index, row in enumerate(rows):
            chunks, query = parts(row)
            selected = selection(chunks, row)
            source = torch.cat(chunks)
            bytes_by_arm = {}
            for rep in (-1,0,1,2):
                widths = [0,32]
                rng.shuffle(widths)
                for width in widths:
                    try:
                        cost, prediction = run_one(row, chunks, query, selected, source, width, rep)
                        bytes_by_arm[width] = cost['persistent_bytes']
                    except torch.cuda.OutOfMemoryError:
                        failures += 1
                        cost = dict(id=row['id'], arm='w'+str(width), rep=rep,
                                    warmup=rep==-1, status='OOM', document_write_ms=None)
                        prediction = dict(id=row['id'], cell='qasper', arm='w'+str(width),
                                          status='OOM', score=None) if rep == 0 else None
                    costs.write(json.dumps(cost, ensure_ascii=False)+'\n'); costs.flush()
                    cost_count += 1
                    if prediction is not None:
                        predictions.write(json.dumps(prediction, ensure_ascii=False)+'\n'); predictions.flush()
                        outputs += 1
                if len(bytes_by_arm) == 2: assert bytes_by_arm[0] == bytes_by_arm[32]
            progress = dict(phase='RUNNING', pid=os.getpid(), examples=index+1, planned=200,
                            predictions=outputs, cost_records=cost_count, oom_records=failures,
                            elapsed_s=time.monotonic()-started)
            dump(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert outputs == 400 and cost_count == 1600
    dump(out / 'complete.json', dict(complete=True, examples=200, predictions=outputs,
                                   cost_records=cost_count, oom_records=failures,
                                   elapsed_s=time.monotonic()-started))
    import importlib.util
    spec = importlib.util.spec_from_file_location('qasper_paired_report', HERE / 'report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    report.main()
    dump(out / 'progress.json', dict(phase='COMPLETE' if not failures else 'COMPLETE_WITH_OOM',
         pid=os.getpid(), examples=200, planned=200, predictions=outputs,
         cost_records=cost_count, oom_records=failures, elapsed_s=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        dump(HERE / 'failure.json', dict(error_type=type(error).__name__, error=str(error), pid=os.getpid()))
        raise
