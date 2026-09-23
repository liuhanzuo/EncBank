"""Explicit code-only deployment; never overwrite an already submitted experiment."""
import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_cacheblend_lora_20260916'
files=['config.py','train_support.py','kv_reference.py','encbank_training_metadata.json',
 'encbank_adapter_config.json','trainable_cacheblend.py','test_trainable.py','train_cacheblend.py',
 'evaluate.py','report.py','pipeline.slurm','release.slurm','release_slot.py','submit_remote.py',
 'status_remote.py','PROTOCOL_zh.md']
payload={name:base64.b64encode((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).decode() for name in files}
script='''import base64,json
from pathlib import Path
root=Path(REMOTE)
root.mkdir(parents=True,exist_ok=True)
assert not (root/'launch.json').exists() and not (root/'submission.json').exists(), 'Already submitted; inspect instead of deploying'
assert not (root/'training').exists() and not (root/'evaluation').exists(), 'Existing results; inspect before replacing code'
for name,encoded in PAYLOAD.items():
    assert '/' not in name and '\\\\' not in name
    temp=root/(name+'.tmp');temp.write_bytes(base64.b64decode(encoded));temp.replace(root/name)
print(json.dumps(dict(deployed=len(PAYLOAD),root=str(root))))
'''.replace('REMOTE',repr(REMOTE)).replace('PAYLOAD',repr(payload))
r=subprocess.run(['ssh','gpu-node1','python3 -'],input=script,text=True,capture_output=True,encoding='utf-8')
print(r.stdout);print(r.stderr);r.check_returncode()
