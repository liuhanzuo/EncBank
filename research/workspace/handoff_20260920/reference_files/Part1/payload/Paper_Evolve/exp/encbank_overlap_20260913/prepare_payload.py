from pathlib import Path
import shutil,tarfile
HERE=Path(__file__).resolve().parent
assert (HERE/'samples.jsonl.gz').exists() and (HERE/'sample_summary.json').exists()
for name in ('encbank','eval'):
    dest=HERE/'Encbank'/name;dest.mkdir(parents=True,exist_ok=True)
    for p in (HERE.parents[1]/'Encbank'/name).glob('*.py'):shutil.copy2(p,dest/p.name)
with tarfile.open(HERE/'payload.tar.gz','w:gz',compresslevel=3) as tar:
    for name in ('common.py','run_quality.py','run_quality.slurm','samples.jsonl.gz','sample_summary.json','PROTOCOL_zh.md','Encbank'):
        tar.add(HERE/name,arcname=name,filter=lambda i:None if '__pycache__' in i.name else i)
print('Payload bytes:',(HERE/'payload.tar.gz').stat().st_size)
