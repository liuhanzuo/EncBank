"""Inspected bootstrap-only retry: isolate stdlib imports from another user's /tmp/subprocess.py."""
import json
from pathlib import Path
import submit_node_local_recovery as base

root=Path(__file__).resolve().parent
previous=json.loads((root/'delivery/node_local_recovery_20260919_0625/monitor_state.json').read_text(encoding='utf-8'))
for job in ['106929','106930']:
    assert any(line.startswith(job+'|FAILED|1:0|') for line in previous['sacct'].splitlines())
    row=next(r for r in previous['records'] if r['job']==job)
    assert row['mirrored_records']==0 and "module 'subprocess' has no attribute 'run'" in row['bootstrap_err']
    assert '/tmp/subprocess.py' in row['bootstrap_err']
proof=json.loads((root/'delivery/storage_failure_20260919_0617/node_local_isolated_probe.json').read_text(encoding='utf-8'))
assert proof['actual_returncode']==0 and proof['probe']
base.OUT=root/'delivery/node_local_recovery_20260919_0630'
assert base.REMOTE.count('qcm-q18-control-20021-20260919-0625')==1
assert base.REMOTE.count('/usr/bin/python3 -u - ')==1
base.REMOTE=base.REMOTE.replace('qcm-q18-control-20021-20260919-0625','qcm-q18-control-20021-20260919-0630')
base.REMOTE=base.REMOTE.replace('/usr/bin/python3 -u - ','/usr/bin/python3 -I -B -u - ')
base.REMOTE=base.REMOTE.replace("acct=run(['sacct','-j','106878,106879'", "acct=run(['sacct','-j','106878,106879,106929,106930'")
base.REMOTE=base.REMOTE.replace("for job in ['106878','106879']:","for job in ['106878','106879','106929','106930']:")
base.main()
