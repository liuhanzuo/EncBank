"""Collect training read-only from BeeGFS, using login-local archive storage."""
import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/node_local_quality_20260919_1255'
CODE=r'''
import json,tarfile
from pathlib import Path
p=Path('/srv/encbank/encbank_c3_c4_20260919')
control=Path('/tmp/qcm-c34-control-20021-20260919-1255');assert control.is_dir()
archives=[]
for j in [6,12,18]:
 train=p/'training'/('j%d_s42'%j)
 complete=json.loads((train/'complete.json').read_text());assert complete['step']==4000
 wait=json.loads((train/'training_parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==0
 dest=control/('training-j%d.tar'%j)
 if not dest.exists():
  with tarfile.open(str(dest)+'.tmp','w') as tar:
   for filename in ['complete.json','training_parent_exit.json','status.json','metadata.json','trainable_parameters.json','train.jsonl','final/adapter_model.safetensors','final/adapter_config.json']:
    f=train/filename;tar.add(str(f),arcname=str(f.relative_to(p)))
  Path(str(dest)+'.tmp').rename(dest)
 archives.append(dict(j=j,path=str(dest),sha256=complete['adapter_sha256']))
print(json.dumps(archives))
'''
def collect_training():
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],capture_output=True,encoding='utf-8',timeout=300)
    assert result.returncode==0,result.stderr[-1500:]
    archives=json.loads(result.stdout)
    for item in archives:
        path=OUT/('training-j%d.tar'%item['j']);marker=OUT/('training-j%d-collected.json'%item['j'])
        if not path.exists():
            temp=path.with_suffix('.tmp')
            subprocess.run(['scp','-q','-o','BatchMode=yes','gpu-node1:'+item['path'],str(temp)],check=True,timeout=300);temp.replace(path)
        if not marker.exists():
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    target=(ROOT/member.name).resolve();assert target.is_relative_to((ROOT/'training').resolve())
                    assert not member.issym() and not member.islnk()
                tar.extractall(ROOT,filter='data')
            adapter=ROOT/'training'/('j%d_s42'%item['j'])/'final/adapter_model.safetensors'
            assert hashlib.sha256(adapter.read_bytes()).hexdigest()==item['sha256']
            marker.write_text(json.dumps(item,indent=2))
    return archives
def main():
    state=json.loads((OUT/'monitor_state.json').read_text());assert state['phase']=='COMPLETE'
    for rec in state['records']:
        assert rec['mirrored_records']==800 and rec['slurm'][1:3]==['COMPLETED','0:0']
        quality=ROOT/'quality'/('j%d_s42'%rec['j'])
        complete=json.loads((quality/'complete.json').read_text());wait=json.loads((quality/'parent_exit.json').read_text())
        assert complete['records']==800 and complete['errors']==0 and wait['actual_wait'] and wait['returncode']==0
        assert len((quality/'predictions.jsonl').read_bytes().splitlines())==800
    archives=collect_training()
    print(json.dumps(dict(collected=[x['j'] for x in archives],quality_source='node-local mirrored complete results with real waits')))
if __name__=='__main__':main()
