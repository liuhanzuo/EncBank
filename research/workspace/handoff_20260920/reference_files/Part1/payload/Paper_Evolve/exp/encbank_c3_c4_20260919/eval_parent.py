import json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
j=int(sys.argv[1]);assert j in [6,12,18]
dest=ROOT/'quality'/('j%d_s42'%j);dest.mkdir(parents=True,exist_ok=True)
local=Path(os.environ['QCM_LOCAL_CACHE_ROOT']).resolve()
assert str(local).startswith('/tmp/qcm-c34-') and local.is_dir()
from triton.runtime.cache import FileCacheManager
cache=FileCacheManager('c34-local-filesystem-check')
probe=Path(cache.put(b'c34-local-cache-write-check','probe.bin',binary=True))
assert probe.resolve().is_relative_to(local) and probe.read_bytes()==b'c34-local-cache-write-check'
(dest/'cache_location.json').write_text(json.dumps(dict(root=str(local),node=platform.node(),
    job=os.environ.get('SLURM_JOB_ID'),triton=os.environ['TRITON_CACHE_DIR'],
    original_put_and_cleanup_passed=True,scientific_configuration_unchanged=True)))
child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'evaluate.py'),'--j',str(j)],cwd=ROOT)
code=child.wait()
(dest/'parent_exit.json').write_text(json.dumps(dict(actual_wait=True,returncode=code,child_pid=child.pid)))
raise SystemExit(code)
