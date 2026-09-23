"""Isolated PEFT conversion, FP32 rounding compatibility, and merge validation."""
import copy
import gc
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

H = Path(__file__).resolve().parent


def save(name, value):
    with (H/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def event(**value):
    print(json.dumps(dict(epoch=time.time(), **value)), flush=True)


def main():
    import hashlib
    import peft
    import transformers
    from common import MODELS, load_model, load_state, tokenizer, torch
    from hybrid_reader import HybridReader
    from runtime_identity import resolve_configs
    from transformers.cache_utils import DynamicCache
    from batch_cache import merge_caches, decode_layers
    from comem_peft import export_adapter, load_comem_peft, LegacyArithmetic, file_sha

    assert os.environ.get('COMEM_ISOLATED_KERNEL_PILOT') == '1'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    assert json.loads((H/'cpu_preflight.json').read_text())['status'] == 'PASS'
    torch.set_num_threads(2)
    torch.manual_seed(4203)
    torch.backends.cuda.enable_cudnn_sdp(False)
    free, total = torch.cuda.mem_get_info()
    save('gpu_preflight.json', dict(epoch=time.time(), hostname=os.uname().nodename,
        gpu=str(torch.cuda.get_device_properties(0)), free_bytes=free, total_bytes=total,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'),
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv'],text=True)))
    assert total-free < 2*2**30, 'Allocated GPU already occupied'
    plan = json.loads((H/'baseline_plan.json').read_text())
    protocol = json.loads((H/'protocol.json').read_text())
    identity, cfg = resolve_configs(plan, MODELS[1])
    assert file_sha(plan['adapter_path']) == plan['adapter_sha256']
    begin = time.perf_counter()
    model = load_model(cfg)
    reader = HybridReader(model, cfg['j'])
    prefix = next(n for n,m in model.named_modules() if m is reader.core)
    reader.attach()
    ckpt = torch.load(plan['adapter_path'], map_location='cpu', weights_only=False)
    resolve_configs(plan, identity, ckpt)
    load_state(reader, ckpt, identity)
    export = export_adapter(ckpt, H/'peft_adapter', prefix, cfg['path'], cfg['revision'])
    save('adapter_export.json', dict(**export, server_directory=str(H/'peft_adapter'),
        source_adapter_sha256=plan['adapter_sha256']))
    del ckpt
    original = []
    for name, wrapped in reader.modules.items():
        parent, _, child = name.rpartition('.')
        target_parent = reader.core.get_submodule(parent)
        original.append((prefix+'.'+name, target_parent, child, wrapped))
        setattr(target_parent, child, wrapped.base)
    model, peft_owner, loaded = load_comem_peft(model, H/'peft_adapter', mode='native')
    reader.model = model
    targets = []
    merge_begin = time.perf_counter()
    with torch.inference_mode():
        for full_name, parent, child, legacy in original:
            layer = loaded[full_name]
            assert torch.equal(layer.lora_A['default'].weight, legacy.a), full_name
            assert torch.equal(layer.lora_B['default'].weight, legacy.b), full_name
            assert layer.get_base_layer() is legacy.base
            merged = copy.deepcopy(layer)
            assert merged.get_base_layer().weight.data_ptr() != legacy.base.weight.data_ptr()
            merged.merge(safe_merge=True)
            merged_linear = merged.get_base_layer()
            assert merged_linear.weight.dtype == torch.bfloat16
            targets.append((parent, child, dict(legacy=legacy, peft_native=layer,
                peft_compat=LegacyArithmetic(layer), peft_merged_bf16=merged_linear)))
            del merged
    torch.cuda.synchronize()
    merge_seconds = time.perf_counter()-merge_begin
    model.requires_grad_(False)
    tok = tokenizer(cfg)
    variants = protocol['variants']

    def select(name):
        for parent, child, modules in targets:
            setattr(parent, child, modules[name])

    def hashed(x):
        return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    save('model_ready.json', dict(epoch=time.time(), load_and_convert_seconds=time.perf_counter()-begin,
        merge_seconds=merge_seconds, torch=torch.__version__, transformers=transformers.__version__,
        peft=peft.__version__, model=cfg, module_count=len(targets), core_prefix=prefix,
        adapter_tensors_equal=True, lower_layers_targeted=False, benchmark_attempts=0,
        merged_weights_saved=False, job_id=os.environ['SLURM_JOB_ID']))
    words = tok.encode('The experiment keeps each document memory immutable. A query reads selected states and then generates an answer. ', add_special_tokens=False)
    stream = words*2000
    segments = [stream[i:i+512] for i in range(48)]
    results = []
    with torch.inference_mode():
        for batch, chunks in protocol['cases']:
            case = f'b{batch}_h{chunks}'
            event(stage='prepare', case=case)
            select('legacy')
            begin = time.perf_counter()
            memories = [reader.write(ids) for ids in segments[:chunks]]
            torch.cuda.synchronize()
            write_s = time.perf_counter()-begin
            hashes = [hashed(m) for m in memories]
            queries = [stream[100+row:100+row+96+4*row] for row in range(batch)]
            histories = [max(1, chunks-row%3) for row in range(batch)]
            initial, prefill, first_logits = {}, {}, {}
            for name in variants:
                select(name)
                assert hashed(reader.write(segments[0])) == hashes[0], 'Write changed'
                torch.cuda.synchronize()
                begin = time.perf_counter()
                lower_rows, upper_rows, qlens, ulens, firsts = [], [], [], [], []
                for row, query in enumerate(queries):
                    lo, up = DynamicCache(config=reader.config), DynamicCache(config=reader.config)
                    qh = reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lo)
                    out = reader.layers(torch.cat(memories[:histories[row]]+[qh],dim=1),reader.j,reader.L,cache=up)
                    firsts.append(reader.logits(out))
                    lower_rows.append(lo);upper_rows.append(up)
                    qlens.append(len(query));ulens.append(512*histories[row]+len(query))
                    del qh,out,lo,up
                lo, lp = merge_caches(lower_rows, qlens, reader.config,0,reader.j,consume=True)
                up, upp = merge_caches(upper_rows, ulens, reader.config,reader.j,reader.L,consume=True)
                first = torch.cat(firsts,dim=0)
                initial[name] = dict(lo=lo,up=up,lp=lp,upp=upp,
                    qp=torch.tensor(qlens,device='cuda'),up_pos=torch.tensor(ulens,device='cuda'),
                    first_token=first.argmax(-1))
                first_logits[name] = first
                del lower_rows,upper_rows,firsts,lo,up,lp,upp,first
                torch.cuda.synchronize()
                prefill[name] = time.perf_counter()-begin
                event(stage='prefill_done',case=case,variant=name,seconds=prefill[name])
            tokens = [torch.tensor([[words[(i+row)%len(words)]] for row in range(batch)],device='cuda') for i in range(protocol['fixed_decode_steps'])]

            def step(cache, token):
                hidden = reader.core.embed_tokens(token)
                hidden = decode_layers(reader,hidden,0,reader.j,cache['lo'],cache['qp'],cache['lp'])
                hidden = decode_layers(reader,hidden,reader.j,reader.L,cache['up'],cache['up_pos'],cache['upp'])
                logits = reader.logits(hidden)
                cache['qp'] += 1;cache['up_pos'] += 1
                return logits

            def run(name, steps, capture=False, greedy=False):
                select(name)
                cache = copy.deepcopy(initial[name])
                token = cache['first_token']
                first = token.clone()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                outputs, chosen = [], []
                for i in range(steps):
                    logits = step(cache,token if greedy else tokens[i])
                    token = logits.argmax(-1)
                    chosen.append(token)
                    if capture:outputs.append(logits)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                ids = torch.cat(chosen,dim=1)
                result = dict(seconds=seconds,per_sequence_tps=steps/seconds,aggregate_tps=batch*steps/seconds,
                    token_ids=ids.tolist(),logits_finite=bool(torch.isfinite(logits).all()),
                    token_sha256=hashed(ids))
                assert result['logits_finite']
                if greedy:
                    all_ids=torch.cat([first,ids],dim=1)
                    result.update(first_token=first.tolist(),texts=tok.batch_decode(all_ids.tolist(),skip_special_tokens=False))
                captured=torch.cat(outputs,dim=1) if capture else None
                del cache,outputs,chosen,logits,ids,token,first
                return captured,result

            def compare(actual, expected):
                a,e=actual.float(),expected.float()
                delta=(a-e).abs()
                lp,lq=e.log_softmax(-1),a.log_softmax(-1)
                kl=(lp.exp()*(lp-lq)).sum(-1)
                agree=float((a.argmax(-1)==e.argmax(-1)).float().mean())
                return dict(exact=torch.equal(actual,expected),max_logit_abs=float(delta.max()),
                    mean_logit_abs=float(delta.mean()),greedy_agreement=agree,
                    mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                    gate_pass=agree==1 and float(delta.max())<=protocol['numerical_gate']['max_abs_logit_error'])

            for name in variants:
                run(name,8)
            expected,_=run('legacy',protocol['fixed_decode_steps'],capture=True)
            checks={}
            for name in variants:
                actual,_=run(name,protocol['fixed_decode_steps'],capture=True)
                checks[name]=dict(decode=compare(actual,expected),prefill=compare(first_logits[name],first_logits['legacy']))
                del actual
                event(stage='numerics',case=case,variant=name,checks=checks[name])
            times={name:[] for name in variants}
            for order in [variants,list(reversed(variants))]:
                for name in order:
                    _,measurement=run(name,protocol['fixed_decode_steps'])
                    times[name].append({k:v for k,v in measurement.items() if k!='token_ids'})
                    event(stage='timing',case=case,variant=name,seconds=measurement['seconds'],tps=measurement['per_sequence_tps'])
            greedy={name:run(name,protocol['greedy_steps'],greedy=True)[1] for name in variants}
            for name in variants:
                greedy[name]['exact_sequences_vs_legacy']=[a==b for a,b in zip(greedy[name]['token_ids'],greedy['legacy']['token_ids'])]
            medians={name:statistics.median(t['seconds'] for t in samples) for name,samples in times.items()}
            row=dict(case=case,batch=batch,max_history_chunks=chunks,ragged=True,
                query_lengths=[len(q) for q in queries],history_chunks=histories,
                fixed_steps=protocol['fixed_decode_steps'],write_seconds=write_s,prefill_seconds=prefill,
                checks=checks,timing_samples=times,median_seconds=medians,
                per_sequence_tps={n:protocol['fixed_decode_steps']/s for n,s in medians.items()},
                speedup={n:medians['legacy']/s for n,s in medians.items()},greedy=greedy,
                H_unchanged=[hashed(m) for m in memories]==hashes)
            save('case_'+case+'.json',row);results.append(row)
            event(stage='case_done',case=case,tps=row['per_sequence_tps'],speedup=row['speedup'])
            del expected,initial,first_logits,memories,tokens
            gc.collect();torch.cuda.empty_cache()
    gates={name:all(x['checks'][name]['decode']['gate_pass'] and x['checks'][name]['prefill']['gate_pass'] and x['H_unchanged'] for x in results) for name in variants}
    compat_exact=all(x['checks']['peft_compat']['decode']['exact'] and x['checks']['peft_compat']['prefill']['exact'] for x in results)
    save('model_probe_result.json',dict(status='MEASURED',job_id=os.environ['SLURM_JOB_ID'],
        ended_epoch=time.time(),benchmark_attempts=0,production_approved=False,
        cases=results,numerical_gates=gates,compat_exact=compat_exact,protocol=protocol,
        peft_adapter_directory=str(H/'peft_adapter'),original_adapter_unchanged=file_sha(plan['adapter_path'])==plan['adapter_sha256']))


if __name__=='__main__':
    try:main()
    except BaseException:
        save('model_probe_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        raise
