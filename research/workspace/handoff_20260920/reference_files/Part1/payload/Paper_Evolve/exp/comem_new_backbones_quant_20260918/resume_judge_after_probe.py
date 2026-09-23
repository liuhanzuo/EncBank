"""Prepare one inspected judge recovery after a valid actual Codex response."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE_CODE=r'''
import datetime,fcntl,json,sys
from pathlib import Path
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
out=r/'judge_gpt6_astra'
lock=(out/'watch.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
payload=json.load(sys.stdin)
launch=json.loads((out/'launch.json').read_text())
cmd=Path('/proc')/str(launch['pid'])/'cmdline'
assert not (cmd.exists() and b'judge_watch.py' in cmd.read_bytes())
proof=json.loads((out/'latest_service_check.json').read_text())
assert proof['at']=='20260918T120832Z' and proof['judgment_ok'] and proof['returncode']==0
assert proof['decision_saved_for_formal_reuse'] and not proof['errors']
assert (out/'cache'/proof['stimulus_digest']/'decision.json').exists()
assert not (out/'transport_recovery.json').exists()
status=json.loads((out/'status.json').read_text())
assert len(status['errors'])==2 and status['decisions']==129
grants={}
for e in status['errors']:
    digest=e['stimulus_digest'];entry=out/'cache'/digest
    assert not (entry/'decision.json').exists()
    attempts=[json.loads(f.read_text()) for f in sorted(entry.glob('attempt-*/**/result.json'))]
    assert len(attempts)==3 and all(not a['ok'] for a in attempts)
    messages=' '.join(json.dumps(a.get('errors',[])).lower() for a in attempts)
    assert '503' in messages and not any(x in messages for x in ['status 401','status 403','unauthorized','quota'])
    grants[digest]=dict(previous_attempts=len(attempts),maximum_additional_attempts=1)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
history=out/'recovery_history'/('actual-codex-'+stamp);history.mkdir(parents=True)
(history/'judge_watch_before.py').write_bytes((r/'judge_watch.py').read_bytes())
record=dict(id=stamp,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),proof=proof,
    extra_attempts_by_stimulus=grants,workers=2,
    reason='User requested actual Codex retry; successful uncached formal judgment confirms Responses route works despite models403. Only two prior503 failures get one new attempt.',
    completed_decisions_unchanged=True,judge_model_and_prompt_unchanged=True)
temp=r/'judge_watch.py.recovery.tmp';temp.write_text(payload['code']);temp.replace(r/'judge_watch.py')
for name in ['launch.json','watch_status.json','watch_failure.json','status.json','progress.json','watch.stdout.log','watch.stderr.log']:
    f=out/name
    if f.exists():f.rename(history/name)
temp=out/'transport_recovery.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out/'transport_recovery.json')
(history/'recovery.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(prepared=True,history=str(history),extra_requests_maximum=len(grants),proof_at=proof['at'])))
'''


def main():
    payload=dict(code=(ROOT/'judge_watch.py').read_text(encoding='utf-8'))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        '/srv/encbank/Paper_Evolve/.venv/bin/python -B -c '+shlex.quote(REMOTE_CODE)],
        input=json.dumps(payload),text=True,capture_output=True,timeout=45)
    print(result.stdout,end='');print(result.stderr,end='')
    if result.returncode==0:
        (ROOT/'delivery/judge_recovery_prepared.json').write_text(result.stdout,encoding='utf-8')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
