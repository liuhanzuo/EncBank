"""Inspect remote work without submitting/restarting; collect final eval when all jobs complete."""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_distillation_sweep_20260917'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',action='store_true');args=ap.parse_args()
    result=subprocess.run(['ssh','gpu-node1',f'/srv/encbank/Paper_Evolve/.venv/bin/python {REMOTE}/remote_status.py'],capture_output=True,text=True,encoding='utf-8',check=True)
    status=json.loads(result.stdout)
    (ROOT/'remote_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in status['runs']:
        last=row.get('last_train') or {};worker=row.get('worker') or {};recipe=worker.get('arm') or {}
        print(json.dumps(dict(name=row['name'],phase=worker.get('phase'),step=last.get('step'),steps=recipe.get('steps'),
            loss=last.get('loss'),elapsed_s=last.get('elapsed_s'),grad_norm=last.get('grad_norm'),
            complete=bool(row['complete']),failure=bool(row['failure'])),ensure_ascii=False))
    print(status['queue'])
    if args.results:
        assert len(status['runs'])==5 and all(r['complete'] for r in status['runs']), 'All five branches must complete first'
        assert len(status['evaluations'])==8 and all(v['complete'] for v in status['evaluations'].values())
        subprocess.run(['scp','-r',f'gpu-node1:{REMOTE}/evaluation',str(ROOT)],check=True)
        subprocess.run(['scp','-r',f'gpu-node1:{REMOTE}/data',str(ROOT)],check=True)
        subprocess.run([sys.executable,str(ROOT/'report.py')],check=True)


if __name__=='__main__':
    main()
