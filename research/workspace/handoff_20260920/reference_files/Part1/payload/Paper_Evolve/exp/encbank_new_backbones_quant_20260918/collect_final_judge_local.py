"""Read local-disk Judge progress, then preserve completed outputs without new calls."""
import datetime, json, shlex, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/final_judge_local'
STAGE='/tmp/qcm-q18-final-judge-20021-20260919-0840'
REMOTE=r'''
import datetime,json,tarfile
from pathlib import Path
p=Path('/tmp/qcm-q18-final-judge-20021-20260919-0840')
def read(name):
    f=p/name
    return json.loads(f.read_text()) if f.exists() else None
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=str(p),
       launch=read('launch.json'),child=read('child_start.json'),
       progress=read('judge_gpt6_astra/progress.json'),parent_exit=read('parent_exit.json'),
       complete=read('complete.json'))
for label,data in [('parent',r['launch']),('child',r['child'])]:
    c=Path('/proc')/str(data['pid'])/'cmdline' if data else None
    r[label+'_alive']=bool(c and c.exists() and b'final_judge_local_runner.py' in c.read_bytes())
r['stderr']={n:(p/n).read_text()[-2500:] for n in ['parent.stderr.log','child.stderr.log'] if (p/n).exists()}
if r['parent_exit'] and r['parent_exit']['returncode']==0:
    assert r['complete']['complete'] and r['complete']['decisions']==11916
    status=read('judge_gpt6_astra/status.json')
    assert status['available_complete'] and status['decisions']==11916 and not status['errors']
    r['status']=status
    dest=p/'saved_judge_outputs.tar.gz'
    if not dest.exists():
        with tarfile.open(str(dest)+'.tmp','w:gz') as t:
            for f in p.iterdir():
                if f.name not in ('tmp','saved_judge_outputs.tar.gz','saved_judge_outputs.tar.gz.tmp'):
                    t.add(str(f),arcname=f.name)
        Path(str(dest)+'.tmp').rename(dest)
    r['archive']=str(dest)
print(json.dumps(r))
'''
def main():
    command='/usr/bin/python3 -I -B -c '+shlex.quote(REMOTE)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',command],encoding='utf-8',capture_output=True,timeout=55)
    assert result.returncode==0,result.stderr[-2500:]
    report=json.loads(result.stdout)
    (OUT/'monitor.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if report.get('archive'):
        archive=OUT/'saved_judge_outputs.tar.gz'
        if not archive.exists():
            subprocess.run(['scp','-q','-o','BatchMode=yes','gpu-node1:'+report['archive'],str(archive)],check=True,timeout=55)
        with tarfile.open(archive) as t:
            raw=t.extractfile('judge_gpt6_astra/judge_decisions.jsonl').read()
            decisions=[json.loads(line) for line in raw.splitlines()]
            assert len(decisions)==11916
            keys={(r['cohort'],r['arm'],r['id']) for r in decisions}
            assert len(keys)==11916
            selected=[r for r in decisions if r['cohort']=='Qwen3.8-27B' and r['arm']=='cache_r128_s8000_h16']
            assert len(selected)==1986 and len({r['id'] for r in selected})==1986
            final=ROOT/'delivery/qwen38_final_generation'
            (final/'judge_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
            (final/'judge_completion_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('at','parent_alive','child_alive','progress','parent_exit','complete','stderr')},ensure_ascii=False))
if __name__=='__main__':main()
