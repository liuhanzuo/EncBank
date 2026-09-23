"""No CUDA: fail before submission/startup if the exact retained model path is unavailable."""
from pathlib import Path
import hashlib,json,time
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());M=json.loads((H/'model_recovery_manifest.json').read_text())
BOUND=Path('/srv/encbank').resolve();model=Path(P['model']).resolve(strict=True);model.relative_to(BOUND)
assert str(model)==M['model'] and P['model_revision']==M['revision'] and M['identity_matches_prior']
checked=[]
for row in M['files']:
    p=model/row['file'];assert p.is_file() and not p.is_symlink(),p
    s=p.stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),p
    if not p.name.endswith('.safetensors'):assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
    checked.append(p.name)
if P['arm']=='encbank':
    cp=Path(P['adapter_path']).resolve(strict=True);cp.relative_to(BOUND)
    assert str(cp)==M['adapter'] and hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']==M['adapter_sha256']
print(json.dumps(dict(status='PASS',epoch=time.time(),model=str(model),files=len(checked),small_files_SHA_rechecked=True,
    full_shards_SHA_reference='model_recovery_manifest.json',full_shards_current_size_mtime_match=True,model_calls=0)))
