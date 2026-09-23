import argparse,json,subprocess,sys
import config

def main():
    p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();out=config.ROOT/'latency_depth'/config.tag(a.j,a.seed);out.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for length in config.SOURCE_LENGTHS:
        cmd=[sys.executable,'-B','-u',str(config.ROOT/'serving.py'),'--j',str(a.j),'--seed',str(a.seed),'--length',str(length)]
        code=subprocess.Popen(cmd).wait();receipts.append(dict(command=cmd,returncode=code,actual_wait=True))
        (out/'processes.json').write_text(json.dumps(receipts,indent=2));assert code==0
        assert json.loads((out/str(length)/'complete.json').read_text())['complete']
    (out/'complete.json').write_text(json.dumps(dict(complete=True,actual_wait=True,lengths=config.SOURCE_LENGTHS)))

if __name__=='__main__':main()
