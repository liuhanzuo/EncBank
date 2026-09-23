import ast,json
from common import ROOT,save,sha
assert not (ROOT/'source_manifest.json').exists()
for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
manifest={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.py','.md'] or p.name in ['plan.json','task_manifest.json','environment_template.json','qualification_reuse.json'])}
save(ROOT/'source_manifest.json',manifest)
print(json.dumps(dict(frozen_sources=len(manifest))))
