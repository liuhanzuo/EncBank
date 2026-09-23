"""Verify all four completed 27B generation shards and save the complete local dataset."""
import collections, datetime, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/qwen38_final_generation';OUT.mkdir(exist_ok=True)
stage=ROOT/'delivery/node_local_recovery_20260919_0630'
inventory=json.loads((stage/'recovery.json').read_text(encoding='utf-8'))
monitor=json.loads((stage/'monitor_state.json').read_text(encoding='utf-8'))
assert not monitor['queue']
payload=[];parts=[];evidence=[]
for task in sorted(inventory['tasks'],key=lambda t:t['shard']):
    job,shard=task['job'],task['shard'];folder=stage/job
    raw=(folder/'predictions.jsonl').read_bytes()
    files={name:json.loads((folder/name).read_text(encoding='utf-8')) for name in
           ['stage.json','start.json','parent_exit.json','evaluation_complete.json','evaluation_protocol.json']}
    parent=files['parent_exit.json'];complete=files['evaluation_complete.json'];protocol=files['evaluation_protocol.json']
    assert any(l.startswith(job+'|COMPLETED|0:0|') for l in monitor['sacct'].splitlines())
    assert parent['actual_wait'] and parent['returncode']==0 and parent['job']==job and parent['completion_exists']
    assert complete['generation_complete'] and complete['records']==1809 and complete['failures']=={'ok':1809}
    assert protocol['rank']==protocol['alpha']==128 and protocol['step']==8000 and protocol['model']['j']==21
    original=ROOT/'delivery/storage_failure_20260919_0617'/('qwen38_shard%d_predictions.jsonl'%shard)
    assert raw.startswith(original.read_bytes())
    payload.append(dict(task=task,files=files,predictions=raw.decode('utf-8')))
    parts.append(raw);evidence.append(dict(job=job,shard=shard,parent_exit=parent,complete=complete,protocol=protocol))
with tarfile.open(ROOT/'delivery/qwen38_final_shard3/saved_outputs.tar.gz') as t:
    folder='results/large-final/Qwen3.8-27B/shard3'
    raw=t.extractfile(folder+'/predictions.jsonl').read()
    complete=json.load(t.extractfile(folder+'/complete.json'))
    protocol=json.load(t.extractfile(folder+'/protocol.json'))
    parent=json.load(t.extractfile('runs/large-final-m1-s3/parent_exit.json'))
    parts.append(raw);evidence.append(dict(job='106640',shard=3,parent_exit=parent,complete=complete,protocol=protocol))
old=json.loads((ROOT/'delivery/qwen38_final_shard3/summary.json').read_text())
assert '106640|COMPLETED|0:0|' in old['sacct']
rows=[json.loads(line) for part in parts for line in part.splitlines()]
assert len(rows)==len({(r['id'],r['arm']) for r in rows})==7236
assert all(r['status']=='ok' and r['rank']==128 and r['step']==8000 for r in rows)
counts=dict(collections.Counter(r['benchmark'] for r in rows))
assert counts==dict(ruler=1500,longeval=500,longbench=1150,babilong=2100,locomo=1986)
assert len({e['protocol']['binding'] for e in evidence})==1
(OUT/'predictions.jsonl').write_bytes(b''.join(parts))
summary=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Qwen3.8-27B',rank=128,alpha=128,step=8000,j=21,
             generation_complete=True,records=7236,counts=counts,errors=0,judge_complete=False,
             full_five_benchmark_complete=False,evidence=evidence,sacct=monitor['sacct']+old['sacct'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(OUT/'canonical_sync_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(generation_complete=True,records=7236,counts=counts,errors=0,judge_complete=False)))
