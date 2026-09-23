"""Export code and result evidence only; never include model/cache directories."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
A=Path(__file__).resolve().parent;B=A.parent
names=['hidden_reader_terminal_20260921','hidden_reader_terminal_r2_20260921','hidden_reader_terminal_r3_20260921','hidden_reader_terminal_r4_20260921']
roots=[B/n for n in names]+[Path(v) for v in json.loads((B/'hidden_reader_terminal_r5_20260921'/'roots.json').read_text()).values()]
files={}
def add(path):
    if path.is_file() and not path.is_symlink():files[str(path.relative_to(B))]=path.read_bytes()
for root in roots:
    for p in root.iterdir():
        if p.suffix in ['.py','.md','.json']:add(p)
    for d in ['vendor','jobs','configs','pairs','mailbox','results','qualification','logs']:
        for p in (root/d).rglob('*'):
            parts=p.relative_to(root).parts
            if any(x in ['compile_cache','__pycache__','cache'] for x in parts):continue
            if p.suffix in ['.py','.json','.log','.txt','.out','.err']:add(p)
for p in A.iterdir():
    if p.suffix in ['.py','.md','.json','.log']:add(p)
for p in (A/'controls').glob('*/*'):
    if p.suffix in ['.py','.json'] and not p.is_symlink():add(p)
for p in (B/'hidden_reader_terminal_r5_20260921').glob('*.json'):add(p)
for root in [B/'hidden_reader_terminal_r2_20260921'/'bandtb_node4_qualification_r2_20260921']:
    for d in ['jobs','qualification']:
        for p in (root/d).rglob('*.json'):add(p)
stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
manifest=dict(snapshot_time=datetime.datetime.now().astimezone().isoformat(),live_snapshot=True,
    files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
    exclusions=['model weights','adapters','checkpoints','HF caches','compiled binaries'],
    note='Atomic result JSON and immutable sources; jobs may still be running. Never interpret a missing result as reward zero.')
files['EXPORT_MANIFEST.json']=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
dest=A/('delivery_'+stamp+'.tar.gz')
with tarfile.open(dest,'w:gz') as tar:
    for name,data in sorted(files.items()):
        info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
receipt=dict(archive=str(dest),files=len(files),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(A/'latest_export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
