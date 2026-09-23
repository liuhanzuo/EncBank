"""Read node-local recovery state and mirror valid answers to the workstation; no GPU submission."""
import json, shlex, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='20260919_0625', choices=['20260919_0625', '20260919_0630'])
args = parser.parse_args()
OUT = ROOT/('delivery/node_local_recovery_' + args.tag)
READ_STAGE = r'''
import json,os
from pathlib import Path
items=ITEMS
result=[]
for item in items:
 root=Path(item['stage']);folder=root/'results/large-final/Qwen3.8-27B'/('shard%d'%item['shard'])
 record=dict(job=item['job'],shard=item['shard'],stage=str(root),exists=root.exists(),files={})
 for name in ['stage.json','start.json','parent_exit.json']:
  f=root/name
  if f.exists():record['files'][name]=json.loads(f.read_text())
 for name in ['progress.json','protocol.json','complete.json']:
  f=folder/name
  if f.exists():record['files']['evaluation_'+name]=json.loads(f.read_text())
 for name in ['child.stderr.log','child.stdout.log']:
  f=root/name
  if f.exists():
   with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-2200));record[name]=stream.read().decode('utf-8','replace')
 f=folder/'predictions.jsonl'
 if f.exists():
  data=f.read_bytes();end=data.rfind(b'\n');data=data[:end+1]
  record['predictions']=data.decode('utf-8')
 else:record['predictions']=''
 for ext in ['out','err']:
  f=Path('/tmp')/('qcm-q18-local-20021-'+item['job']+'.'+ext)
  if f.exists():record['bootstrap_'+ext]=f.read_text()[-2200:]
 result.append(record)
print(json.dumps(result))
'''
REMOTE = r'''
import datetime,json,shlex,subprocess,sys
p=json.load(sys.stdin)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=45).stdout
ids=','.join(t['job'] for t in p['tasks'])
acct=run(['sacct','-j',ids,'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,End'])
queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'])
jobs=[line.split('|') for line in queue.splitlines() if line.split('|')[1].startswith('qcm-q18-')]
assert sum(int(j[3].split(':')[-1]) for j in jobs)<=4
records=[]
for node in sorted({t['node'] for t in p['tasks']}):
 tasks=[t for t in p['tasks'] if t['node']==node]
 active=next((t['job'] for t in tasks if any(j[0]==t['job'] and j[2]=='RUNNING' for j in jobs)),None)
 if active:
  command=['srun','--jobid='+active,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none']
 elif all(any(j[0]==t['job'] and j[2]=='PENDING' for j in jobs) for t in tasks):
  records.extend(dict(job=t['job'],shard=t['shard'],stage=t['stage'],exists=False,files={},predictions='',not_started=True) for t in tasks)
  continue
 else:
  command=['srun','--partition=gpu','--nodelist='+node,'--job-name=qcm-local-readout','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','--time=00:03:00']
 code=p['read_code'].replace('ITEMS',repr(tasks))
 records.extend(json.loads(run(command+['/usr/bin/python3','-I','-B','-c',code])))
print(json.dumps(dict(at=datetime.datetime.utcnow().isoformat()+'Z',sacct=acct,queue=jobs,records=records,cpu_only=True)))
'''

def main():
    recovery=json.loads((OUT/'recovery.json').read_text(encoding='utf-8'))
    payload=dict(tasks=recovery['tasks'],read_code=READ_STAGE)
    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
                            '/usr/bin/python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload),
                           capture_output=True,text=True,timeout=58)
    assert process.returncode==0,process.stderr
    report=json.loads(process.stdout)
    for record in report['records']:
        raw=record.pop('predictions').encode('utf-8')
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==len({(r['id'],r['arm']) for r in rows})
        assert all(r['status']=='ok' and r['rank']==128 and r['step']==8000 for r in rows)
        dest=OUT/record['job'];dest.mkdir(exist_ok=True)
        old=dest/'predictions.jsonl'
        if old.exists():assert raw.startswith(old.read_bytes()), 'Saved answer prefix changed'
        if rows:
            original=ROOT/'delivery/storage_failure_20260919_0617'/('qwen38_shard%d_predictions.jsonl'%record['shard'])
            assert raw.startswith(original.read_bytes())
            old.write_bytes(raw)
        record['mirrored_records']=len(rows)
        for name,value in record['files'].items():
            (dest/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        for name in ['child.stderr.log','child.stdout.log','bootstrap_out','bootstrap_err']:
            if name in record:(dest/name).write_text(record[name],encoding='utf-8')
    (OUT/'monitor_state.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
