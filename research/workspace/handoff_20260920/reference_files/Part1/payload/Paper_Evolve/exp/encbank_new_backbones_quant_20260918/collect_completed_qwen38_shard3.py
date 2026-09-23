"""Collect one truly completed shard; never present its scores as full benchmarks."""
import collections,datetime,json,subprocess,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent
remote='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
folder='results/large-final/Qwen3.8-27B/shard3';task='runs/large-final-m1-s3'
members=[folder+'/'+n for n in ['predictions.jsonl','protocol.json','complete.json']]+[task+'/'+n for n in ['submission.json','parent_exit.json']]
out=r/'delivery/qwen38_final_shard3';out.mkdir(exist_ok=True)
archive=out/'saved_outputs.tar.gz'
with archive.open('wb') as stdout:
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','timeout -k 5s 45s tar -czf - -C '+remote+' '+' '.join(members)],stdout=stdout,stderr=subprocess.PIPE,timeout=55)
assert p.returncode==0,p.stderr.decode()
with tarfile.open(archive,'r:gz') as t:
    def read(n):return json.load(t.extractfile(n))
    complete=read(folder+'/complete.json');protocol=read(folder+'/protocol.json');parent=read(task+'/parent_exit.json');sub=read(task+'/submission.json')
    raw=t.extractfile(folder+'/predictions.jsonl').read()
rows=[json.loads(l) for l in raw.splitlines()]
assert len(rows)==1809 and len({(x['id'],x['arm']) for x in rows})==1809
assert all(x['status']=='ok' and x['rank']==128 and x['step']==8000 for x in rows)
assert protocol['rank']==128 and protocol['alpha']==128 and protocol['step']==8000
assert complete['generation_complete'] and complete['records']==1809 and complete['failures']=={'ok':1809}
assert parent['actual_wait'] and parent['returncode']==0 and parent['completion_exists'] and parent['job']==sub['job']=='106640'
assert raw.startswith((r/'delivery/storage_failure_20260919_recurrence/qwen38_final_shard3_live_0541.jsonl').read_bytes())
a=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','sacct -j 106640 -n -P -o JobIDRaw,State,ExitCode,End'],text=True,capture_output=True,timeout=25)
assert a.returncode==0 and any(l.startswith('106640|COMPLETED|0:0|') for l in a.stdout.splitlines())
summary=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Qwen3.8-27B',mode='large-final',shard=3,
    rank=128,alpha=128,step=8000,records=1809,unique=1809,counts=dict(collections.Counter(x['benchmark'] for x in rows)),
    generation_shard_complete=True,parent_exit=parent,sacct=a.stdout,collection_actual_returncode=p.returncode,
    judge_complete=False,full_five_benchmark_complete=False,scope='One of four generation shards only; no final benchmark scores.')
(out/'predictions.jsonl').write_bytes(raw)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
