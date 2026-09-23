"""Export only code, logs and results; exclude all model and compiled caches."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent;T=B/'hot_buffer_terminal_20260921';files={}
def add(p):
    if p.is_file() and not p.is_symlink():files[str(p.relative_to(B))]=p.read_bytes()
for root in [H]+[Path(j['root']) for j in json.loads((T/'submissions.json').read_text())]:
    for rel,expected in json.loads((root/'source_manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected
    for p in root.rglob('*'):
        parts=p.relative_to(root).parts
        if any(x in ['cache','compile_cache','__pycache__'] for x in parts):continue
        if p.suffix in ['.py','.md','.json','.log','.txt','.out','.err']:add(p)
for p in T.glob('*'):
    if p.is_file():add(p)
old=B/'hidden_reader_terminal_r5_20260921'
for root in [Path(p) for p in json.loads((old/'roots.json').read_text()).values()]:
    for sub in ['pairs','jobs','results']:
        for p in (root/sub).rglob('*.json'):
            if p.name in ['parent_exit.json','task_controller_parent_exit.json','scope_retirement.json','integrated_complete.json','execution_receipt.json']:add(p)
stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
manifest=dict(at=datetime.datetime.now().astimezone().isoformat(),files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()},
    note='GPU replay is complete; Terminal-Bench files may be a live snapshot. No weights/adapters/checkpoints/caches included.')
files['EXPORT_MANIFEST.json']=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
dest=H/('delivery_'+stamp+'.tar.gz')
with tarfile.open(dest,'w:gz') as tar:
    for name,data in sorted(files.items()):info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
receipt=dict(archive=str(dest),files=len(files),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(H/'latest_export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
