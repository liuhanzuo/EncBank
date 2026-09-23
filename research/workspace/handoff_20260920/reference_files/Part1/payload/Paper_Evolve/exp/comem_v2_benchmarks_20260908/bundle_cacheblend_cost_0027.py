"""Copy only this task's small completed metadata for remote CPU audit."""
from pathlib import Path
import json
import tarfile
import datetime as dt

root = Path(__file__).resolve().parent
out = root / 'results/protocol/cacheblend_cost_0027'
out.mkdir(parents=True, exist_ok=True)
files = {root / 'heartbeat_cost_20260908_2202.json', root / 'heartbeat_cacheblend_cost_20260909_0005.json',
         root / 'results/local/bootstrap_cacheblend_serving/status.json'}
jobs, stats = [], {}
for n in (32768, 131072):
    cb = root / f'results/local/cacheblend_serving_reuse/full/cacheblend16_{n}/attempts/0001'
    if not (cb / 'COMPLETED.json').is_file():
        continue
    entries = [('cacheblend16', cb)]
    for arm in ('pub', 'pub_sink', 'pub_lora', 'j0', 'fix_all'):
        prefix = 'pub_lora/' if arm == 'pub_lora' else ''
        attempt = '0002' if arm == 'pub_sink' and n == 32768 else '0001'
        entries.append((arm, root / f'results/local/serving_reuse/{prefix}full/{arm}_{n}/attempts/{attempt}'))
    for arm, folder in entries:
        for pattern in ('*.json', '*.jsonl'):
            files.update(folder.glob(pattern))
        store = folder / f'store_{n}_{arm}'
        files.update([store / 'tokens.pt', store / 'store.json'])
        stats[str(store.relative_to(root))] = {p.name: p.stat().st_size for p in store.iterdir() if p.is_file()}
        jobs.append({'arm': arm, 'length': n, 'relative_path': str(folder.relative_to(root)).replace('\\', '/')})
smoke = root / 'results/local/cacheblend_serving_reuse/smoke/cacheblend16_1024/attempts/0001'
files.update(smoke.glob('*.json'))
files.update(smoke.glob('*.jsonl'))
meta = out / 'bundle_metadata.json'
meta.write_text(json.dumps({'created_at': dt.datetime.now().astimezone().isoformat(), 'jobs': jobs,
    'actual_store_file_bytes': stats, 'copy_scope': 'JSON/JSONL + small tokens.pt only, no model or KV tensor file'}, indent=2) + '\n')
files.add(meta)
with tarfile.open(out / 'metadata.tar', 'w') as tar:
    for path in sorted(files):
        tar.add(path, arcname=str(path.relative_to(root)).replace('\\', '/'), recursive=False)
print(json.dumps({'files': len(files), 'tar_bytes': (out/'metadata.tar').stat().st_size, 'jobs': jobs}, indent=2))
