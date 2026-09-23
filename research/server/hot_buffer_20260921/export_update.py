import datetime,hashlib,io,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent;files={}
def add(p):
    if p.exists():files[str(p.relative_to(B))]=p.read_bytes()
for name in ['READOUT_zh.md','summary.json','terminal_snapshot.json','CURRENT_RUN.md','report.py','terminal_status.py']:add(H/name)
for j in json.loads((B/'hot_buffer_terminal_20260921'/'submissions.json').read_text()):
    R=Path(j['root']);task=j['task'];arm=j['arm']
    for name in ['environment.json','qualification_parent_exit.json','parent_exit.json','task_controller_parent_exit.json','complete.json','status.json']:
        add(R/'pairs'/task/name)
    for name in ['actual_result.json','execution_receipt.json']:add(R/'results'/(task+'--'+arm)/name)
    for p in (R/'mailbox'/task).glob('*.json'):add(p)
    for p in (R/'jobs').glob('*.json'):add(p)
stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S');manifest=dict(at=stamp,files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()})
files['EXPORT_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
dest=H/('update_'+stamp+'.tar.gz')
with tarfile.open(dest,'w:gz') as tar:
    for n,d in sorted(files.items()):info=tarfile.TarInfo(n);info.size=len(d);tar.addfile(info,io.BytesIO(d))
print(json.dumps(dict(path=str(dest),files=len(files),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))
