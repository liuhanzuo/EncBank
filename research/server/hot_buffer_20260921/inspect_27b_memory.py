"""Read-only server evidence collection; never load model tensors or run inference."""
import collections, glob, json, math, pathlib, struct, subprocess

HOME = pathlib.Path('/srv/encbank')
MODEL = HOME/'qcomem_runtime_20260911/models/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ROOT = HOME/'qcomem_align_codex_20260911/terminal_bench_full89_20260919'
out = {'config': json.loads((MODEL/'config.json').read_text())}
sizes = collections.Counter(); lora = 0; tensors = 0
for path in MODEL.glob('*.safetensors'):
    with path.open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    for name, info in header.items():
        if name == '__metadata__': continue
        count = math.prod(info['shape']); tensors += 1
        group = 'vision' if 'visual' in name else 'text_or_other'
        sizes[group] += count * 2  # all parameters hypothetically loaded as BF16
        parts = name.split('.')
        if 'layers' in parts and group != 'vision':
            i = parts.index('layers')
            if int(parts[i+1]) >= 21 and len(info['shape']) == 2 and name.endswith('.weight'):
                lora += 32 * sum(info['shape']) * 4
out['weight_bf16_bytes'] = dict(sizes)
out['suffix_lora_fp32_bytes_from_linear_shapes'] = lora
out['tensor_count'] = tensors
sources = glob.glob(str(HOME/'Paper_Evolve/.venv/lib/python*/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'))
out['model_source'] = sources
out['state_source_excerpt'] = []
for src in sources:
    lines = pathlib.Path(src).read_text().splitlines()
    for i, line in enumerate(lines):
        if any(s in line for s in ['recurrent_state', 'conv_state', 'self.key_dim', 'self.value_dim', 'self.conv_dim', 'dtype=torch.float32']):
            out['state_source_excerpt'].append([i+1, line])
def stats(values):
    v=sorted(values)
    if not v:return None
    return {'n':len(v),'min':v[0],'median':v[len(v)//2],'p90':v[min(len(v)-1,int(len(v)*.9))],'max':v[-1]}
for tag,rel in [('old_k12','comem_k12_no_task_deadline_r6_20260920/run_comem'),('old_k48','comem_k48_no_task_deadline_r6_20260920/run_comem'),('dense','server_control_20260920/dense_unbounded_20260921/run_dense')]:
    r=ROOT/rel; rows=[]
    for path in (r/'mailbox').glob('*.response.json'):
        try: rows.append(json.loads(path.read_text()))
        except (ValueError,OSError):pass
    keys=['full_history_tokens','prompt_tokens','query_tokens','generated_tokens','persistent_H_bytes','peak_allocated_bytes','current_allocated_bytes','shared_batch_cache_bytes','independent_prefill_cache_bytes','all_sessions_H_bytes']
    out[tag]={'responses':len(rows),'stats':{k:stats([x[k] for x in rows if isinstance(x.get(k),(float,int))]) for k in keys}}
    out[tag]['sample_keys']=list(rows[0]) if rows else []
    ev=r/'events.jsonl'
    if ev.exists():
        events=[]
        for line in ev.read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        out[tag]['event_kinds']=dict(collections.Counter(x.get('event') for x in events))
        starts=[x for x in events if x.get('event')=='request_start']
        out[tag]['event_prompt_tokens']=stats([x['prompt_tokens'] for x in starts if 'prompt_tokens' in x])
        out[tag]['latest_start']=starts[-1:] 
    complete=r/'worker_complete.json'
    if complete.exists():out[tag]['worker_complete']=json.loads(complete.read_text())
print(json.dumps(out,indent=2))
