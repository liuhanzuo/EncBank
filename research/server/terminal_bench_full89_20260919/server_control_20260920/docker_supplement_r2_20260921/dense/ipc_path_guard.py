"""Check both short Unix socket roots on the allocated node before loading."""
from pathlib import Path
import json,os,tempfile,time
approved=Path('/srv/encbank').resolve()
expected=Path(json.loads(Path('plan.json').read_text())['ipc_root']).resolve()
rows={}
for key in ['TMPDIR','VLLM_RPC_BASE_PATH']:
 p=Path(os.environ[key]).resolve();p.relative_to(approved);assert p==expected and p.is_dir()
 length=len(os.fsencode(p/('0'*36)));assert length<=107
 rows[key]=dict(path=str(p),uuid_endpoint_bytes=length)
assert Path(tempfile.gettempdir()).resolve()==expected
Path('ipc_path_guard.json').write_text(json.dumps(dict(status='PASS',epoch=time.time(),roots=rows),indent=2)+'\n')
