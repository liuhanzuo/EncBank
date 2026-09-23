"""Recover inspected shard0 failure on node8; retain the two live node4 jobs unchanged."""
import hashlib, json, shlex, subprocess
from pathlib import Path
from submit_node_local_recovery import REMOTE
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/node_local_recovery_20260919_0645'
assert not OUT.exists(), 'One-shot already attempted; inspect receipts before retry'
source=ROOT/'delivery/storage_failure_20260919_0617'
data=(source/'qwen38_shard0_predictions.jsonl').read_bytes()
proof=json.loads((source/'shard0_node8_probe_0645.json').read_text(encoding='utf-8'))
assert proof['actual_returncode']==0 and proof['node']=='gpu-host' and proof['saved_records']==808
task=dict(task='large-final-m1-s0',shard=0,previous_job='106877',
          sha256=hashlib.sha256(data).hexdigest(),saved_records=808)
replacements={
    'qcm-q18-control-20021-20260919-0625':'qcm-q18-control-20021-20260919-0645',
    "p['proof']['node']=='gpu-host'":"p['proof']['node']=='gpu-host'",
    "'106878,106879'":"'106877'",
    "for job in ['106878','106879']:":"for job in ['106877']:",
    '/usr/bin/python3 -u - ':'/usr/bin/python3 -I -B -u - ',
    "'--nodelist=gpu-node4'":"'--nodelist=gpu-node8'",
    "node='gpu-node4'":"node='gpu-node8'",
}
for old,new in replacements.items():
    assert REMOTE.count(old)==1,(old,REMOTE.count(old))
    REMOTE=REMOTE.replace(old,new)
payload=dict(tasks=[task],proof=proof,script=(ROOT/'control/node_local_evaluation.py').read_text(encoding='utf-8'))
OUT.mkdir()
(OUT/'local_intent.json').write_text(json.dumps(payload['tasks'],indent=2),encoding='utf-8')
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
                       '/usr/bin/python3 -I -B -c '+shlex.quote(REMOTE)],input=json.dumps(payload),
                      capture_output=True,text=True,timeout=58)
(OUT/'submission_stdout.txt').write_text(result.stdout,encoding='utf-8')
(OUT/'submission_stderr.txt').write_text(result.stderr,encoding='utf-8')
assert result.returncode==0,result.stderr
report=json.loads(result.stdout)
(OUT/'recovery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
# Add the actual accepted submission to the combined read-only monitor inventory.
inventory=ROOT/'delivery/node_local_recovery_20260919_0630/recovery.json'
original=inventory.read_bytes()
combined=json.loads(original)
assert {t['job'] for t in combined['tasks']}=={'106933','106934'}
assert all(t['shard']!=0 for t in combined['tasks'])
(inventory.parent/'recovery_before_shard0.json').write_bytes(original)
combined['tasks'].extend(report['tasks'])
combined['additional_recoveries']=[dict(receipt=str(OUT/'recovery.json'),control=report['control'])]
inventory.write_text(json.dumps(combined,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
