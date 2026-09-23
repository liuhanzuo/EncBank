"""Test the exact failed create paths while their owners are stopped; never launches jobs."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,json,os,socket
from pathlib import Path
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
for pid,needle in [(3759947,b'control/coordinator.py'),(3817318,b'judge_watch.py')]:
    p=Path('/proc')/str(pid)/'cmdline';assert not (p.exists() and needle in p.read_bytes())
paths=['cache/tmp-106701/torchinductor_liuhanzuo','cache/tmp-106702/torchinductor_liuhanzuo',
       'judge_gpt6_astra/watch_status.json.tmp','results/large-final/Qwen3.5-9B/shard3/complete.json.tmp',
       'PAUSE_SUBMISSIONS.persistent-storage.tmp']
results=[]
for i,name in enumerate(paths):
    p=r/name;v=dict(path=name)
    try:
        assert not p.exists(),'Existing path: leave untouched and inspect separately'
        if i<2:
            os.makedirs(str(p),exist_ok=True);p.rmdir();v['operation']='mkdir exact original torchinductor path'
        else:
            q=p.with_name(p.name+'.diagnostic-renamed');assert not q.exists()
            with p.open('xb') as f:f.write(b'original-path-diagnostic\n');f.flush();os.fsync(f.fileno())
            assert p.read_bytes()==b'original-path-diagnostic\n';p.replace(q)
            assert q.read_bytes()==b'original-path-diagnostic\n';q.unlink()
            v['operation']='create exact failed tmp, fsync, read, rename to diagnostic filename, remove; canonical output untouched'
        v['ok']=True
    except (OSError,AssertionError) as e:v.update(ok=False,errno=getattr(e,'errno',None),error=str(e))
    results.append(v)
print(json.dumps(dict(at=datetime.datetime.utcnow().isoformat()+'Z',host=socket.gethostname(),checks=results,
    all_passed=all(v['ok'] for v in results),canonical_outputs_untouched=True,no_gpu_or_judge_started=True)))
'''
def main():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        'timeout -k 5s 35s /usr/bin/python3 -c '+shlex.quote(CODE)],text=True,capture_output=True,timeout=45)
    assert p.returncode==0,p.stderr
    x=json.loads(p.stdout);out=ROOT/'delivery/storage_failure_20260919_recurrence'
    (out/'original_path_check_0555.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps(x))
if __name__=='__main__':main()
