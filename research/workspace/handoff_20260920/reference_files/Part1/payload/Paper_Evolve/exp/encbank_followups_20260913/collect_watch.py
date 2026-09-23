"""Quiet collection of this task's submitted jobs; no messages or new GPU jobs."""
from pathlib import Path
import json,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_followups_20260913'
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True,encoding='utf-8')
def main():
    begin=time.time()
    while time.time()-begin<8*3600:
        try:
            r=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpu-node1',f'/srv/encbank/Paper_Evolve/.venv/bin/python {REMOTE}/status_remote.py'])
            status=json.loads(r.stdout)
            for group in ('kv_quality','clean_quality'):
                if status[group]['complete'] and not (HERE/f'{group}_collected.json').exists():
                    run(['ssh','-o','BatchMode=yes','gpu-node1',f'cd {REMOTE} && tar -czf {group}_results.tar.gz {group}'])
                    run(['scp','-o','BatchMode=yes',f'gpu-node1:{REMOTE}/{group}_results.tar.gz',str(HERE/f'{group}_results.tar.gz')])
                    with tarfile.open(HERE/f'{group}_results.tar.gz') as t:t.extractall(HERE,filter='data')
                    (HERE/f'{group}_collected.json').write_text(json.dumps({'complete':True,'collected_local_time':time.strftime('%Y-%m-%d %H:%M:%S')}))
                    script='aggregate_clean.py' if group=='clean_quality' else 'aggregate_kv.py'
                    log=run([sys.executable,str(HERE/script)]);(HERE/f'{group}_aggregate.log').write_text(log.stdout,encoding='utf-8')
            status['local_cost']=[{'process':i,'complete':(HERE/'kv_cost'/f'process_{i:02d}'/'complete.json').exists(),'progress':json.loads(p.read_text()) if (p:=HERE/'kv_cost'/f'process_{i:02d}'/'progress.json').exists() else None} for i in (1,2,3)]
            if all(r['complete'] for r in status['local_cost']) and not (HERE/'kv_cost_summary.json').exists():run([sys.executable,str(HERE/'aggregate_kv.py')])
            status['all_measurements_complete']=all(status[g]['complete'] for g in ('kv_quality','clean_quality')) and all(r['complete'] for r in status['local_cost'])
            status['checked_local_time']=time.strftime('%Y-%m-%d %H:%M:%S')
            p=HERE/'STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2),encoding='utf-8');q.replace(p)
            if status['all_measurements_complete']:
                (HERE/'ALL_MEASUREMENTS_COMPLETE.txt').write_text(status['checked_local_time']);return
        except Exception as error:
            with (HERE/'collection_errors.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+str(error)+'\n')
        time.sleep(60)
    raise SystemExit('collection window expired; inspect task logs')
if __name__=='__main__':main()
