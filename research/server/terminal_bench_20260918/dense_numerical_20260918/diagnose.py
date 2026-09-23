"""Fixed-history numerical diagnosis only; never task-quality evidence."""
from pathlib import Path
import gc, hashlib, inspect, json, os, platform, subprocess, time, traceback
import torch
import transformers
from common import MODELS, load_model, dump

H = Path(__file__).resolve().parent
H.relative_to(Path('/srv/encbank').resolve())
R = H / 'run_dense'
plan = json.loads((H / 'agent_plan.json').read_text())
raw = (H / 'fixed_history.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'a55f762a1fa3448170a51443f96b3e2b5e500f4363b96d0393527e508d4fb3b7'
history = json.loads(raw)
started = time.time()
context = {}
layer_stats = []
events = (R / 'events.jsonl').open('x')

def event(kind, **data):
    row = dict(event=kind, epoch=time.time(), **data)
    line = json.dumps(row)
    events.write(line + '\n'); events.flush()
    print(line, flush=True)

def stats(t):
    finite = torch.isfinite(t)
    return dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(finite.all()),
                nonfinite=int((~finite).sum()), absmax=float(t.abs().max()))

def cache_stats(cache):
    rows = []
    if cache is None: return rows
    for i, layer in enumerate(cache.layers):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            v = getattr(layer, name, None)
            values = v if isinstance(v, (tuple, list)) else [v]
            for j, t in enumerate(values):
                if torch.is_tensor(t):
                    rows.append(dict(layer=i, field=name, index=j, **stats(t)))
    return rows

def hook(name):
    def capture(module, args, output):
        if not context.get('trace_layers'): return
        t = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(t):
            layer_stats.append((name, torch.isfinite(t).all(), t.abs().max()))
    return capture

def layer_report():
    rows = [dict(module=n, finite=bool(f), absmax=float(m)) for n, f, m in layer_stats]
    layer_stats.clear()
    return rows

class Nonfinite(RuntimeError): pass

try:
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(2); torch.manual_seed(plan['seed'])
    free, total = torch.cuda.mem_get_info()
    cap = min(80 * 2**30, int(total * .9))
    assert free >= 70 * 2**30
    torch.cuda.set_per_process_memory_fraction(cap / total)
    event('admission', free=free, total=total, cap=cap,
          nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
          processes=subprocess.check_output(['ps', '-u', 'liuhanzuo', '-o', 'pid,ppid,comm'], text=True))
    model = load_model(MODELS[1]); model.requires_grad_(False)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    core = getattr(model.model, 'language_model', model.model)
    for i, block in enumerate(core.layers):
        block.register_forward_hook(hook(f'layer.{i}'))
        for n in ('linear_attn', 'self_attn', 'mlp'):
            if hasattr(block, n): getattr(block, n).register_forward_hook(hook(f'layer.{i}.{n}'))
    import transformers.models.qwen3_5.modeling_qwen3_5 as impl
    source = Path(inspect.getsourcefile(impl))
    dump(R / 'worker_ready.json', dict(pid=os.getpid(), job=os.environ['SLURM_JOB_ID'],
         host=platform.node(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         transformers=transformers.__version__, impl_path=str(source),
         impl_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), input_sha256=hashlib.sha256(raw).hexdigest(),
         calls={n: str(getattr(impl,n)) for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']},
         purpose='diagnosis only; saved answers teacher-forced, no benchmark score'))

    def reset():
        for obj in (model, model.model):
            if hasattr(obj, 'rope_deltas'): obj.rope_deltas = None

    def forward(ids, cache, full_length=None):
        layer_stats.clear()
        args = dict(input_ids=torch.tensor([ids], device='cuda'), past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        if full_length is not None:
            args['attention_mask'] = torch.ones((1,full_length), device='cuda', dtype=torch.long)
        out = model(**args)
        logits = out.logits
        rows = layer_report()
        if not bool(torch.isfinite(logits).all()):
            event('nonfinite', context=dict(context), logits=stats(logits),
                  layers=rows, cache=cache_stats(out.past_key_values))
            raise Nonfinite(json.dumps(context))
        if context['phase'] != 'decode':
            event('finite_prefill', context=dict(context), logits=stats(logits), layers=rows,
                  cache=cache_stats(out.past_key_values))
        return logits, out.past_key_values

    def sample(logits):
        scores = logits[0,-1].float() / plan['temperature']
        values, indices = scores.topk(plan['top_k'])
        cumulative = values.softmax(-1).cumsum(-1)
        mask = cumulative > plan['top_p']; mask[1:] = mask[:-1].clone(); mask[0] = False
        values[mask] = -float('inf'); probs = values.softmax(-1)
        if not bool(torch.isfinite(probs).all() & (probs >= 0).all() & (probs.sum() > 0)):
            event('invalid_sampler', context=dict(context), probabilities=stats(probs))
            raise Nonfinite('sampler')
        return int(indices[torch.multinomial(probs,1)].item())

    results = {}; cache = None; logits = None
    with torch.inference_mode():
        reset(); torch.manual_seed(plan['seed']); matches=0; compared=0
        try:
            for row in history['steps']:
                step = row['step']; context.update(mode='incremental', step=step, phase='prefill', token=0, trace_layers=True)
                assert (cache.get_seq_length() if cache is not None else 0) == row['cached_prefix_tokens']
                logits, cache = forward(row['new_prefill_ids'], cache, len(row['ledger_ids']))
                assert cache.get_seq_length() == len(row['ledger_ids'])
                saved = row.get('saved_generated_ids')
                if saved is not None:
                    for j, token in enumerate(saved):
                        context.update(phase='decode', token=j, trace_layers=False)
                        sampled=sample(logits); matches += int(sampled==token); compared += 1
                        if j != len(saved)-1:
                            logits, cache = forward([token], cache)
                    event('saved_step_replayed', step=step, saved_tokens=len(saved), sample_matches=matches,
                          sample_compared=compared, cache_tokens=cache.get_seq_length())
                else:
                    generated=[]
                    stop=model.generation_config.eos_token_id
                    stop=set(stop if isinstance(stop,list) else [stop])
                    for j in range(plan['max_new_tokens']):
                        context.update(phase='decode', token=j, trace_layers=True)
                        token=sample(logits); generated.append(token)
                        if token in stop: break
                        logits, cache=forward([token],cache)
                    results['failed_request_generated_ids']=generated
            results['incremental']='finite_through_replay'
        except Nonfinite as exc:
            results['incremental']='nonfinite_reproduced'; results['failure_context']=dict(context)
        results.update(sample_matches=matches, sample_compared=compared)
        cache=None; logits=None; gc.collect(); torch.cuda.empty_cache()
        reset()
        context.update(mode='fresh', step=10, phase='full_prefill', token=0, trace_layers=True)
        try:
            logits, cache=forward(history['steps'][10]['ledger_ids'],None,len(history['steps'][10]['ledger_ids']))
            results['fresh_full_prefill']='finite'; results['fresh_logits']=stats(logits)
        except Nonfinite:
            results['fresh_full_prefill']='nonfinite'
        results.update(elapsed_seconds=time.time()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       diagnostic_only=True, root_cause_not_automatically_established=True)
        dump(R/'diagnostic_result.json',results); event('diagnostic_complete',**results)
except BaseException:
    dump(R/'worker_failure.json',dict(traceback=traceback.format_exc(),context=context)); raise
finally:
    events.close()
