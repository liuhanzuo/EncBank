"""Apply the user's four-benchmark priority while evaluation is still queued."""
import datetime,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
launch=json.loads((root/'launch.json').read_text());job=launch['evaluation_array']
q=subprocess.run(['squeue','-j',job,'-h','-o','%T'],text=True,capture_output=True,check=True)
assert q.stdout.split() and set(q.stdout.split())=={'PENDING'}, 'Evaluation is no longer entirely pending; inspect before changing scope'
assert not list((root/'results').glob('*/shard*/predictions.jsonl')), 'Existing evaluation outputs must be preserved under their original scope'
with tarfile.open(root/'four_benchmark_scope.tar.gz') as t:
    assert all((root/m.name).resolve().is_relative_to(root.resolve()) and m.isfile() for m in t.getmembers())
    t.extractall(root,filter='data')
scope=json.loads((root/'evaluation_scope.json').read_text())
assert scope['samples_per_model']==5250 and scope['total_records']==73500
launch.update(active_scope_id=scope['scope_id'],active_benchmarks=scope['benchmarks'],deferred_benchmarks=scope['deferred'],
              active_samples_per_model=5250,active_records_per_model=36750,total_active_records=73500)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluation_array=job,
             state_before='PENDING',duplicate_jobs_submitted=False,training_changed=False,scope=scope)
(root/'scope_change.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
