"""Minimal remote metadata-only allocation protocol; no model or kernel imports."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE_ROOT='/srv/encbank/qcomem_align_codex_20260911/repo'
ROOT=Path(os.environ.get('QCOMEM_REPO_ROOT','/srv/encbank/legacy_workspace')).resolve()
PYTHON='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
RESOURCE={'allocator_cap_bytes':200*2**30,'minimum_free_bytes':220*2**30,'stable_idle_seconds':45}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);temporary=p.with_name(p.name+'.tmp');temporary.write_bytes((json.dumps(v,indent=2)+'\n').encode());temporary.replace(p)
def local(relative):
 p=(ROOT/relative).resolve();p.relative_to(ROOT)
 if ROOT==Path(REMOTE_ROOT):p.relative_to(Path('/srv/encbank').resolve())
 return p
def preflight(expected):
 assert sha(HERE/'plan.json')==expected
 plan=read(HERE/'plan.json');assert plan['resource_policy']==RESOURCE
 assert plan['model_allowed'] is False and plan['tensor_or_kernel_tests_allowed'] is False and plan['training_allowed'] is False
 for name,digest in plan['source_sha256'].items():assert sha(local(name))==digest,name
 return plan
