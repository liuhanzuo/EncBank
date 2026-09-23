import hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
verification=json.loads((R/'verification.json').read_text());assert verification['all_actual_wait_zero'] and verification['all_slurm_completed_zero']
jobs=[]
for phase in ['screen','confirm']:jobs+=json.loads((R/('submissions_'+phase+'.json')).read_text())
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt','.png'] and p.name!='delivery_manifest.json']
for rel in json.loads((R/'source_manifest.json').read_text()):
    files.append(R/rel)
for j in jobs:
    out=R/j['run']/'results';files.extend(out.glob('*.json'));files.extend(out.glob('*.jsonl'))
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(R/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(R/'results_for_local.tar.gz','w:gz') as tar:
    for name in list(manifest)+['delivery_manifest.json']:tar.add(R/name,arcname=name)
print(json.dumps(dict(files=len(manifest),archive_bytes=(R/'results_for_local.tar.gz').stat().st_size)))
