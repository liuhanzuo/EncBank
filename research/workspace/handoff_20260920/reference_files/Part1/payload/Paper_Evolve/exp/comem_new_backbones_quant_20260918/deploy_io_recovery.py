"""Deploy prepared control-only recovery once storage is writable and owner is idle."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ['coordinator.py', 'recovery_worker.py', 'recovery.sh', 'io_resilience.py',
         'durable_runner.py', 'recover_storage_1722.py', 'checkpoint_verified_1843.json']
REMOTE_CODE = r'''
import fcntl,json,os,subprocess,sys,uuid
from pathlib import Path
root=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
payload=json.load(sys.stdin)
lock=(root/'coordinator.lock').open('a+b')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
queue=subprocess.run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T'],universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=20).stdout
assert not any('qcm-q18-' in row for row in queue.splitlines()),queue
probe=root/'control'/('health-'+uuid.uuid4().hex)
with probe.open('wb') as f:
    f.write(b'io-ok\n');f.flush();os.fsync(f.fileno())
renamed=probe.with_suffix('.renamed');probe.rename(renamed)
assert renamed.read_bytes()==b'io-ok\n';renamed.unlink()
allowed={'coordinator.py','recovery_worker.py','recovery.sh','io_resilience.py','durable_runner.py','recover_storage_1722.py','checkpoint_verified_1843.json'}
assert set(payload)==allowed
for name,content in payload.items():
    p=root/'control'/name
    temp=p.with_name(p.name+'.upload.tmp')
    with temp.open('w',encoding='utf-8') as f:
        f.write(content);f.flush();os.fsync(f.fileno())
    temp.replace(p)
    assert p.read_text(encoding='utf-8')==content
print(json.dumps({'deployed':sorted(payload),'owner_idle':True,'scientific_files_changed':False}),flush=True)
'''


def main():
    payload = {name: ((ROOT / 'delivery/recovery_checkpoint_20260918_1843.json')
                     if name == 'checkpoint_verified_1843.json' else (ROOT / 'control' / name)).read_text(encoding='utf-8')
               for name in FILES}
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
          'gpu-node1', '/usr/bin/python3 -c ' + shlex.quote(REMOTE_CODE)],
          input=json.dumps(payload), text=True, capture_output=True, timeout=45)
    print(result.stdout, end='')
    print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
