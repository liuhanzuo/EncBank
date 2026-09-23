from pathlib import Path
import shutil,tarfile,json
ROOT=Path('F:/Paper_Evolve')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'exp/encbank_v2_benchmarks_20260908'
for part in ('encbank','eval'):
    dest=HERE/'Encbank'/part
    dest.mkdir(parents=True,exist_ok=True)
    for f in (ROOT/'Encbank'/part).glob('*.py'): shutil.copy2(f,dest/f.name)
shutil.copy2(SOURCE/'prepare_babilong.py',HERE/'prepare_babilong.py')
for part in ('prompts.py','metrics.py'):
    dest=HERE/'vendor/babilong/babilong'/part
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SOURCE/'vendor/babilong/babilong'/part,dest)
dest=HERE/'data/longbench'; dest.mkdir(parents=True,exist_ok=True)
for f in (SOURCE/'data/longbench').glob('*.jsonl'): shutil.copy2(f,dest/f.name)
for task in ('qa1','qa2','qa5'):
    dest=HERE/'data/babilong'/task; dest.mkdir(parents=True,exist_ok=True)
    for f in (SOURCE/'data/babilong'/task).glob('*.json'): shutil.copy2(f,dest/f.name)
shutil.copy2(ROOT/'exp/data/pg19_essay.txt',HERE/'data/pg19_essay.txt')
with tarfile.open(HERE/'payload.tar.gz','w:gz',compresslevel=3) as tar:
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f.name!='payload.tar.gz' and '__pycache__' not in f.parts: tar.add(f,arcname=f.relative_to(HERE))
print(json.dumps({'payload_bytes':(HERE/'payload.tar.gz').stat().st_size,'formal_samples':5250,'shards':4}))
