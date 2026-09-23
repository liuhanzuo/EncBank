"""Copy Harbor's backend for a diagnostic-only compatibility experiment."""
import hashlib
import json
from pathlib import Path
import harbor.environments.singularity.singularity as backend

root = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
target = root / 'container_backend_candidate'
target.mkdir(exist_ok=True)
manifest = {}
for name in ['singularity.py', 'server.py', 'bootstrap.sh']:
    data = (Path(backend.__file__).parent / name).read_bytes()
    before = hashlib.sha256(data).hexdigest()
    if name == 'bootstrap.sh':
        text = data.decode()
        old = 'export DEBIAN_FRONTEND=noninteractive'
        new = '''export DEBIAN_FRONTEND=noninteractive
# Diagnostic compatibility fix for root-mapped Apptainer package downloads.
mkdir -p /tmp/comem-apt/partial
printf 'APT::Sandbox::User "root";\\nDir::Cache::archives "/tmp/comem-apt";\\n' > /tmp/comem-apt.conf
export APT_CONFIG=/tmp/comem-apt.conf'''
        assert text.count(old) == 1
        text = text.replace(old, new)
        assert text.count('_SYS_PY=/usr/bin/python3') == 1
        text = text.replace('_SYS_PY=/usr/bin/python3', '_SYS_PY="$(command -v python3 || echo /usr/bin/python3)"')
        data = text.encode()
    (target / name).write_bytes(data)
    manifest[name] = dict(original_sha256=before, candidate_sha256=hashlib.sha256(data).hexdigest())
(target / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
