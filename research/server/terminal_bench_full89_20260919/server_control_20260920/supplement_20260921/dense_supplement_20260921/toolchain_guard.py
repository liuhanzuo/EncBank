"""Verify resolved compiler executables/cache roots on allocated node before model."""
from pathlib import Path
import json,os,shutil,subprocess,time
H=Path(__file__).resolve().parent;approved=Path('/srv/encbank').resolve();rows={}
for key in ['FLASHINFER_WORKSPACE_BASE','TVM_FFI_CACHE_DIR']:
 p=Path(os.environ[key]).resolve();p.relative_to(approved);assert p==H or p==H/'tvm_ffi';p.mkdir(exist_ok=True)
for name in ['ninja','nvcc','g++','gcc']:
 p=shutil.which(name);assert p,name
 r=subprocess.run([p,'--version'],capture_output=True,text=True,timeout=15);assert r.returncode==0
 rows[name]=dict(path=p,version=r.stdout)
assert Path(rows['ninja']['path']).resolve()==Path('/srv/encbank/venvs/rllm/bin/ninja').resolve()
(H/'toolchain_guard.json').write_text(json.dumps(dict(status='PASS',epoch=time.time(),tools=rows),indent=2)+'\n')
