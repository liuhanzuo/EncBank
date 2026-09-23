"""Package complete scores, exact evaluated token inputs, and executed code."""
from pathlib import Path
import json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
assert json.loads((HERE/'summary.json').read_text())['complete']
dest=ROOT/'output/Encbank_frozen_j12_accuracy.zip'
items=[]
for p in HERE.iterdir():
    if p.is_file() and p.suffix in ('.py','.md','.json','.slurm','.txt') and p.name!='NEXT_STEPS.md': items.append(p)
for part in ('Encbank','vendor','data','results'):
    items.extend(p for p in (HERE/part).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for p in (HERE/'collected').glob('slurm-*'): items.append(p)
items.extend(p for p in (HERE/'collected/smoke').rglob('*') if p.is_file())
with zipfile.ZipFile(dest,'w') as z:
    for p in sorted(set(items)):
        z.write(p,p.relative_to(HERE).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.gz' else zipfile.ZIP_DEFLATED,compresslevel=None if p.suffix=='.gz' else 3)
with zipfile.ZipFile(dest) as z: assert z.testzip() is None
print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'files':len(items)}))
