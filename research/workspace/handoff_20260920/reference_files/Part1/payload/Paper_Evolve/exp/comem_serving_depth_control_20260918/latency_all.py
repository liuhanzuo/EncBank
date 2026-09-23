import json,random,subprocess,sys
import config

def main():
    out=config.ROOT/'latency_depth';out.mkdir(parents=True,exist_ok=True)
    arms=[(j,s) for j in config.DEPTHS for s in config.SEEDS]
    random.Random(52918).shuffle(arms);records=[]
    for j,seed in arms:
        cmd=[sys.executable,'-B','-u',str(config.ROOT/'latency.py'),'--j',str(j),'--seed',str(seed)]
        code=subprocess.Popen(cmd).wait();records.append(dict(j=j,seed=seed,returncode=code,actual_wait=True))
        (out/'processes.json').write_text(json.dumps(records,indent=2));assert code==0
    (out/'complete.json').write_text(json.dumps(dict(complete=True,arms=18,single_gpu_allocation=True)))

if __name__=='__main__':main()
