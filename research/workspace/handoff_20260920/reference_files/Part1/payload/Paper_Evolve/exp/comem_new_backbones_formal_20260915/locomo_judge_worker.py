"""Collect newly verified LoCoMo shards and judge them without waiting for all GPUs."""
import json,msvcrt,os,subprocess,time,traceback
from pathlib import Path
from astra_judge_batch import ROOT,run_batch,write_json
REMOTE='/srv/encbank/comem_new_backbones_formal_20260915'
OUT=ROOT/'judge_gpt6_astra'/'formal_new_models'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'worker.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    protocol=json.loads((ROOT/'judge_protocol.json').read_text())
    assert protocol['status']=='READY' and protocol['model']=='gpt-6-astra'
    local=OUT/'inputs';local.mkdir(exist_ok=True)
    errors=0
    while not (OUT/'STOP').exists():
        try:
            cmd=f'/srv/encbank/Paper_Evolve/.venv/bin/python -B {REMOTE}/export_locomo_judge.py'
            r=subprocess.run(['ssh','-o','ConnectTimeout=20','gpu-node1',cmd],text=True,capture_output=True,check=True,timeout=90)
            manifest=json.loads(r.stdout)
            for item in manifest['files']:
                name=item['file'];assert Path(name).name==name
                p=local/name
                if not p.exists():
                    tmp=p.with_suffix('.download')
                    subprocess.run(['scp','gpu-node1:'+REMOTE+'/judge_inputs/'+name,str(tmp)],check=True,capture_output=True,timeout=120)
                    assert sum(1 for _ in tmp.open(encoding='utf-8'))==item['records']
                    tmp.replace(p)
            inputs=[local/item['file'] for item in manifest['files']]
            if inputs:
                write_json(OUT/'worker_status.json',dict(phase='JUDGING',pid=os.getpid(),at=time.time(),
                    verified_shards=manifest['verified_shards'],model='gpt-6-astra'))
                result=run_batch(inputs,OUT,protocol,workers=8)
                if result['errors']:
                    write_json(OUT/'worker_status.json',dict(phase='JUDGE_FAILURE',pid=os.getpid(),at=time.time(),errors=result['errors']))
                    return
                if manifest['verified_shards']==8 and result['available_complete']:
                    assert result['decisions']==27804
                    write_json(OUT/'worker_status.json',dict(phase='COMPLETE',pid=os.getpid(),at=time.time(),records=27804,oom=result['oom']))
                    return
            write_json(OUT/'worker_status.json',dict(phase='WAITING_FOR_VERIFIED_GENERATIONS',pid=os.getpid(),at=time.time(),
                verified_shards=manifest['verified_shards'],decisions=result['decisions'] if inputs else 0,model='gpt-6-astra'))
            errors=0
        except Exception:
            errors+=1
            write_json(OUT/'worker_status.json',dict(phase='COLLECTION_ERROR',pid=os.getpid(),at=time.time(),failures=errors,traceback=traceback.format_exc()))
            if errors>=3:return
        time.sleep(60)
    write_json(OUT/'worker_status.json',dict(phase='STOPPED',pid=os.getpid(),at=time.time()))
if __name__=='__main__':main()
