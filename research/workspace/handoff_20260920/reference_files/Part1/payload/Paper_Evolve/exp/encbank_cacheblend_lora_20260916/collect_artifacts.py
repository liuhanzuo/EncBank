"""Download completed scientific outputs and final adapter, not optimizer states."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_cacheblend_lora_20260916'
script='''import json
from pathlib import Path
root=Path(REMOTE)
assert json.loads((root/'COMPLETE.json').read_text())['verified_records']==2000
files=['results.csv','per_sample_scores.csv','training_curve.csv','summary.json','COMPLETE.json',
'cpu_checks.json','launch.json','submission.json','evaluation/predictions.jsonl','evaluation/protocol.json',
'evaluation/correctness_cacheblend.json','evaluation/complete.json','training/metadata.json',
'training/correctness_initial.json','training/correctness_adapted.json','training/complete.json','training/train.jsonl']
files += [str(p.relative_to(root)) for p in (root/'training/final').iterdir() if p.is_file()]
assert all((root/f).is_file() for f in files)
print(json.dumps(files))
'''.replace('REMOTE',repr(REMOTE))
r=subprocess.run(['ssh','gpu-node1','python3 -'],input=script,text=True,capture_output=True,encoding='utf-8',check=True)
files=json.loads(r.stdout)
for name in files:
    target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.download')
    subprocess.run(['scp','-q',f'gpu-node1:{REMOTE}/{name}',str(temporary)],check=True)
    temporary.replace(target)
print(json.dumps(dict(collected=len(files),root=str(ROOT))))
