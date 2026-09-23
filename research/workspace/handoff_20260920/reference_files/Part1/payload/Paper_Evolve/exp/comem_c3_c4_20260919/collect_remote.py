"""Mirror only completed outputs/checkpoints for the local same-GPU depth timing."""
import json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/comem_c3_c4_20260919'
CODE=r'''
import json,tarfile
from pathlib import Path
p=Path('/srv/encbank/comem_c3_c4_20260919')
archives=[]
for j in [6,12,18]:
    train=p/'training'/('j%d_s42'%j);quality=p/'quality'/('j%d_s42'%j)
    if not (train/'complete.json').exists() or not (quality/'complete.json').exists():continue
    assert json.loads((train/'complete.json').read_text())['step']==4000
    assert json.loads((quality/'complete.json').read_text())['records']==800
    for f in [train/'training_parent_exit.json',quality/'parent_exit.json']:
        wait=json.loads(f.read_text());assert wait['actual_wait'] and wait['returncode']==0
    dest=p/('delivery-j%d.tar.gz'%j)
    if not dest.exists():
        with tarfile.open(str(dest)+'.tmp','w:gz') as tar:
            for filename in ['complete.json','training_parent_exit.json','status.json','metadata.json','trainable_parameters.json','train.jsonl']:
                f=train/filename;tar.add(str(f),arcname=str(f.relative_to(p)))
            for filename in ['adapter_model.safetensors','adapter_config.json']:
                f=train/'final'/filename;tar.add(str(f),arcname=str(f.relative_to(p)))
            tar.add(str(quality),arcname=str(quality.relative_to(p)))
        Path(str(dest)+'.tmp').rename(dest)
    archives.append(dict(j=j,path=str(dest)))
print(json.dumps(archives))
'''
def main():
    if (ROOT/'delivery/node_local_quality_20260919_1255/recovery.json').exists():
        from collect_node_quality import main as collect_recovered
        return collect_recovered()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],capture_output=True,encoding='utf-8',timeout=55)
    assert result.returncode==0,result.stderr[-1000:]
    archives=json.loads(result.stdout);dest=ROOT/'delivery';dest.mkdir(exist_ok=True)
    for item in archives:
        path=dest/('j%d.tar.gz'%item['j'])
        if not path.exists():
            temp=path.with_suffix('.tmp')
            subprocess.run(['scp','-q','-o','BatchMode=yes','gpu-node1:'+item['path'],str(temp)],check=True,timeout=180)
            temp.replace(path)
        marker=dest/('j%d_collected.json'%item['j'])
        if not marker.exists():
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    target=(ROOT/member.name).resolve();assert target.is_relative_to(ROOT.resolve())
                    assert not member.issym() and not member.islnk()
                tar.extractall(ROOT,filter='data')
            marker.write_text(json.dumps(dict(complete=True,j=item['j'],archive=str(path))))
    print(json.dumps(dict(collected=[item['j'] for item in archives])))
if __name__=='__main__':main()
