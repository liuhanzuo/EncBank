import argparse, json, os, subprocess, sys
import config
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and a.j in config.DEPTHS and a.seed in config.SEEDS
    out=config.ROOT/('smoke_training' if a.smoke else 'training')/config.tag(a.j,a.seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    cmd=[sys.executable,'-u','-B',str(config.ROOT/'controlled_train_core.py'),
        '--model',config.MODEL,'--data',config.DATA,'--token-cache',config.TOKENS,
        '--out',str(out),'--devices','cuda:0','--j',str(a.j),'--adapter-start','24',
        '--rank','32','--alpha','32','--steps','4000','--seed',str(a.seed),'--save-every','1000']
    if a.smoke:cmd+=['--stop-after','2']
    if (out/'last.pt').exists():cmd+=['--resume',str(out/'last.pt')]
    child=subprocess.Popen(cmd,env=os.environ.copy());code=child.wait()
    (out/'training_parent_exit.json').write_text(json.dumps(dict(returncode=code,actual_wait=True)))
    assert code==0,code
    status=json.loads((out/'status.json').read_text())
    assert status['step']==(2 if a.smoke else 4000)
    if not a.smoke:assert status['complete']
    (out/'complete.json').write_text(json.dumps(dict(complete=True,smoke=a.smoke,step=status['step'],actual_wait=True)))

if __name__=='__main__':main()
