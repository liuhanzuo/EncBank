"""Read existing request metadata. No inference, model loads, or benchmark mutation."""
import collections
import datetime
import hashlib
import json
import statistics
from pathlib import Path

S = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')


def distribution(values):
    v = sorted(x for x in values if isinstance(x, (int, float)))
    if not v:
        return None
    return {'n': len(v), 'median': statistics.median(v), 'p90': v[int(.9*(len(v)-1))], 'sum': sum(v)}


def main():
    registry = json.loads((S/'unbounded_20260921/scale4_20260921/registry.json').read_text())
    rows, sources = [], {}
    fields = ['status','generated_tokens','prompt_tokens','query_tokens','full_history_tokens',
              'request_seconds','queue_seconds','decode_tokens_per_second','decode_seconds',
              'ttft_seconds','selection_seconds','write_seconds','independent_prefill_seconds',
              'batch_prefill_wait_seconds','full_worker_ttft_seconds','newly_written_tokens',
              'reused_written_tokens','cached_prefix_tokens','shared_batch_cache_bytes',
              'persistent_H_bytes','batch_initial_size','actual_selected_chunks']
    for spec in registry['runs']:
        root = Path(spec['root'])
        plan = json.loads((root/'plan.json').read_text())
        box = Path(plan['rpc_root'])/plan['arm']
        sources[spec['id']] = {'root': str(root), 'plan_sha256': hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()}
        for f in box.glob('*.response.json'):
            r = json.loads(f.read_text())
            if r.get('task') not in spec['tasks']:
                raise ValueError('Noncanonical response '+str(f))
            x = {k: r.get(k) for k in fields}
            x.update(arm=spec['arm'], run=spec['id'], task=r['task'], step=r.get('step'),
                     response_path=str(f), response_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            if x['decode_tokens_per_second'] is None and (r.get('decode_seconds') or 0)>0:
                x['decode_tokens_per_second'] = max(0,r.get('generated_tokens',0)-1)/r['decode_seconds']
            if r.get('full_prompt_sha256'):
                x['full_prompt_sha256'] = r['full_prompt_sha256']
            elif r.get('prompt_ids'):
                x['full_prompt_sha256'] = hashlib.sha256(json.dumps(r['prompt_ids'],separators=(',',':')).encode()).hexdigest()
            if x['step'] is None:
                req = f.with_name(f.name.replace('.response.','.request.'))
                if req.exists():x['step']=json.loads(req.read_text()).get('step')
            rows.append(x)
    arms = {}
    for arm in ['dense','k12','k48']:
        a = [r for r in rows if r['arm']==arm]
        timed = [r for r in a if r['status']=='ok' and (r['generated_tokens'] or 0)>=128]
        arms[arm] = {'responses':len(a),'status_counts':dict(collections.Counter(r['status'] for r in a)),
                     'timed_ok_at_least_128_tokens':len(timed)}
        for field in ['decode_tokens_per_second','generated_tokens','request_seconds','selection_seconds',
                      'write_seconds','independent_prefill_seconds','queue_seconds','batch_prefill_wait_seconds',
                      'shared_batch_cache_bytes','cached_prefix_tokens']:
            arms[arm][field]=distribution([r[field] for r in timed])
        arms[arm]['by_generated_length']={}
        for lo,hi in [(128,1024),(1024,8192),(8192,32768),(32768,262145)]:
            group=[r for r in timed if lo<=r['generated_tokens']<hi]
            arms[arm]['by_generated_length'][f'{lo}-{hi-1}']={
                'decode_tps':distribution([r['decode_tokens_per_second'] for r in group]),
                'request_s':distribution([r['request_seconds'] for r in group])}
    first=collections.defaultdict(dict)
    for r in rows:
        if r['step']==0 and r.get('full_prompt_sha256'):
            first[(r['task'],r['full_prompt_sha256'])][r['arm']]=r
    matches=[{'task':k[0],'prompt_sha256':k[1],'arms':{a:{f:r[f] for f in ['generated_tokens','decode_tokens_per_second','request_seconds','write_seconds','independent_prefill_seconds']} for a,r in g.items()}} for k,g in first.items() if 'dense' in g and len(g)>1]
    result={'checked_at':datetime.datetime.now().astimezone().isoformat(),'arms':arms,'same_initial_prompt':matches,
            'caveats':['Observed completed requests, not a controlled speed benchmark.',
                       'Different tasks, prompt lengths, generation lengths and concurrent cohorts; incomplete long requests are censored.',
                       'Encbank decode time includes pauses for other requests prefill/refill; stage totals overlap between requests.',
                       'Matching initial prompts do not imply matching generated token sequences or engine versions.'],
            'sources':sources,'request_metadata':rows}
    print(json.dumps(result))


if __name__=='__main__':main()
