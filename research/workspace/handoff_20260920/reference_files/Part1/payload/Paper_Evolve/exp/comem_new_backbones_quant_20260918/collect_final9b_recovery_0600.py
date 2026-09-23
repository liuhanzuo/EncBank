"""Collect completed generation and real recovery receipts without changing remote jobs."""
import collections, datetime, hashlib, json, shlex, subprocess, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'
OUT = ROOT / 'delivery/qwen35_final_generation'
OUT.mkdir(exist_ok=True)
files = ['maintenance_history/storage-canary-20260919-0600/recovery.json']
for shard in range(4):
    files.extend(f'results/large-final/Qwen3.5-9B/shard{shard}/{name}'
                 for name in ['predictions.jsonl', 'complete.json', 'protocol.json'])
    files.extend(f'runs/large-final-m0-s{shard}/{name}'
                 for name in ['submission.json', 'parent_exit.json'])
archive = OUT / 'saved_outputs.tar.gz'
command = 'timeout -k 5s 45s tar -czf - -C ' + shlex.quote(REMOTE) + ' ' + ' '.join(map(shlex.quote, files))
with archive.open('wb') as f:
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1', command],
                             stdout=f, stderr=subprocess.PIPE, timeout=55)
assert process.returncode == 0, process.stderr.decode()
rows, receipts, parts = [], [], []
with tarfile.open(archive, 'r:gz') as tar:
    def read(path):
        return json.load(tar.extractfile(path))
    recovery = read(files[0])
    for shard in range(4):
        folder = f'results/large-final/Qwen3.5-9B/shard{shard}'
        run = f'runs/large-final-m0-s{shard}'
        complete, protocol = read(folder + '/complete.json'), read(folder + '/protocol.json')
        parent, submission = read(run + '/parent_exit.json'), read(run + '/submission.json')
        data = tar.extractfile(folder + '/predictions.jsonl').read()
        part = [json.loads(line) for line in data.splitlines()]
        assert len(part) == 1809 and complete['records'] == 1809
        assert complete['generation_complete'] and complete['failures'] == {'ok': 1809}
        assert protocol['rank'] == protocol['alpha'] == 128 and protocol['step'] == 8000
        assert parent['actual_wait'] and parent['returncode'] == 0 and parent['completion_exists']
        assert parent['job'] == submission['job']
        if shard == 3:
            previous = ROOT / 'delivery/storage_failure_20260919_recurrence/qwen35_final_shard3_predictions.jsonl'
            assert data == previous.read_bytes(), 'Finalization must preserve all previously saved answers exactly'
            assert hashlib.sha256(data).hexdigest() == '2c912152e1c425c86ed075abadf3d7ad06b96571c6be78aed5df258abacad6de'
        rows.extend(part)
        parts.append(data)
        receipts.append(parent)
assert len(rows) == len({(r['id'], r['arm']) for r in rows}) == 7236
assert all(r['status'] == 'ok' and r['rank'] == 128 and r['step'] == 8000 for r in rows)
counts = dict(collections.Counter(r['benchmark'] for r in rows))
assert counts == dict(ruler=1500, longeval=500, longbench=1150, babilong=2100, locomo=1986)
jobs = [r['job'] for r in receipts]
acct = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
                       'sacct -j ' + ','.join(jobs) + ' -n -P -o JobIDRaw,State,ExitCode,End'],
                      text=True, capture_output=True, timeout=30)
assert acct.returncode == 0, acct.stderr
assert all(any(line.startswith(job + '|COMPLETED|0:0|') for line in acct.stdout.splitlines()) for job in jobs)
summary = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), model='Qwen3.5-9B',
               rank=128, alpha=128, step=8000, records=7236, counts=counts,
               generation_complete=True, final_shard_answers_unchanged=True,
               judge_complete=False, full_five_benchmark_complete=False,
               parent_exits=receipts, sacct=acct.stdout, collection_actual_returncode=process.returncode)
(OUT / 'predictions.jsonl').write_bytes(b''.join(parts))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(OUT / 'recovery.json').write_text(json.dumps(recovery, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary))
