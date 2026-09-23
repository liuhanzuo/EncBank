"""Collect verified generation evidence without weights or Judge API requests."""
import argparse,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/comem_new_backbones_formal_20260915'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['four','locomo'],default='four');args=parser.parse_args()
    script=r'''
import json,os,tarfile
from pathlib import Path
root=Path(REMOTE);scope=SCOPE
directory='results' if scope=='four' else 'results_locomo'
expected_scope='four-benchmarks-locomo-deferred-20260915' if scope=='four' else 'locomo-resumed-astra-20260916'
files=[];verified=0;records=0;oom=0
for model in ('Qwen3.5-9B','Qwen3.8-27B'):
    for shard in range(4):
        p=root/directory/model/f'shard{shard}'
        if not (p/'verified_summary.json').exists():continue
        v=json.loads((p/'verified_summary.json').read_text())
        assert v['verified'] and v['scope_id']==expected_scope
        verified+=1;records+=v['records'];oom+=v['oom_records']
        for name in ('predictions.jsonl','protocol.json','correctness.json','complete.json','verified_summary.json'):
            f=p/name;assert f.is_file();files.append(f)
if scope=='four':
    assert verified==8 and records==73500,(verified,records)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['generation_verified_complete'] and summary['scope']['scope_id']==expected_scope
    files += [root/'summary.json',root/'evaluation_scope.json']
else:
    assert verified>0,'No verified LoCoMo shard to collect yet'
    files += [root/'locomo_scope.json']
delivery=root/'delivery';delivery.mkdir(exist_ok=True)
archive=delivery/f'{scope}_verified_{verified}shards.tar.gz'
if not archive.exists():
    temp=archive.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as t:
        for file in files:t.add(file,arcname=str(file.relative_to(root)),recursive=False)
    os.replace(temp,archive)
print(json.dumps(dict(archive=str(archive),scope=scope,verified_shards=verified,records=records,oom=oom,
    files=[str(f.relative_to(root)) for f in files],bytes=archive.stat().st_size)))
'''.replace('REMOTE',repr(REMOTE)).replace('SCOPE',repr(args.scope))
    r=subprocess.run(['ssh','gpu-node1','python3 -'],input=script,text=True,capture_output=True,check=True)
    info=json.loads(r.stdout)
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    archive=delivery/Path(info['archive']).name
    if not archive.exists():
        temporary=archive.with_suffix('.download')
        subprocess.run(['scp','-q','gpu-node1:'+info['archive'],str(temporary)],check=True)
        assert temporary.stat().st_size==info['bytes'];temporary.replace(archive)
    assert archive.stat().st_size==info['bytes']
    with tarfile.open(archive,'r:gz') as t:
        assert {m.name for m in t.getmembers()}==set(info['files'])
        for m in t.getmembers():
            assert m.isfile() and not Path(m.name).is_absolute()
            target=(ROOT/m.name).resolve();assert target.is_relative_to(ROOT.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            payload=t.extractfile(m).read();assert len(payload)==m.size
            if target.exists():
                assert target.read_bytes()==payload, f'Existing local evidence differs: {m.name}'
            else:
                temporary=target.with_suffix(target.suffix+'.download');temporary.write_bytes(payload);temporary.replace(target)
    (delivery/f'{args.scope}_collection.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(info|dict(local_archive=str(archive))))

if __name__=='__main__':main()
