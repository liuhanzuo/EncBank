"""Download selected reference files over SSH; never download model weights."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib, io, json, sys, tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
base = Path('/srv/encbank/Encbank_Migration_20260920')
pe = 'Part1/payload/Paper_Evolve/'
qc = 'Part2/payload/qencbank/'
text_ext = {'.py','.md','.tex','.bib','.sty','.bst','.cls','.json','.yaml','.yml','.toml','.txt','.sh','.slurm','.cfg','.ini'}
asset_ext = {'.pdf','.png','.jpg','.jpeg','.svg','.eps'}
explicit = {
 'Part1/CONTINUE_HERE_zh.md', 'Part1/README_恢复说明.md',
 qc+'paper_autonomous_multifork_iteration/state/decision_log.md',
 qc+'paper_autonomous_multifork_iteration/state/codex_experiment_status.json',
 qc+'paper_autonomous_multifork_iteration/evidence/experiment_registry.json',
 qc+'paper_autonomous_multifork_iteration/paper_state.json',
 qc+'paper_autonomous_multifork_iteration/state/paper_state.json',
 qc+'paper_autonomous_multifork_iteration/main_v78_edge9.tex',
 qc+'paper_autonomous_multifork_iteration/main_v78_edge9.pdf',
}
def selected(name, size):
 p = PurePosixPath(name)
 if size > 10*1024*1024: return False
 if any(s in {'.git','.venv','__pycache__','models','checkpoints','.runtime','node_modules'} for s in p.parts): return False
 if p.suffix.lower() not in text_ext | asset_ext: return False
 if name in explicit: return True
 if name.startswith(pe+'Encbank/'): return True
 if name.startswith(pe+'exp/'):
  sub = PurePosixPath(name[len(pe+'exp/'):])
  return p.suffix.lower() in text_ext and (len(sub.parts)<=2 or (len(sub.parts)==3 and sub.parts[1]=='delivery'))
 if name.startswith(qc):
  sub=PurePosixPath(name[len(qc):])
  if len(sub.parts)==1 and p.suffix.lower()=='.md': return True
  if str(sub).startswith('paper_autonomous_multifork_iteration/evidence/encbank_honly_formal_20260911/runtime/') and p.suffix=='.py': return True
 return False
report={'source':str(base),'checked_at':datetime.now().astimezone().isoformat(),'files':[],'unavailable':[],'scope':'Selected code, manuscript sources/assets, reports and state. No checkpoints, model weights, credentials or task environments.'}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 with open(base/'_transfer/UPLOAD_MANIFEST.jsonl',encoding='utf-8') as manifest:
  for line in manifest:
   row=json.loads(line); name=row['path']
   if not selected(name,row['bytes']): continue
   p=base/name
   if not p.is_file() or p.is_symlink():
    report['unavailable'].append({'path':name,'reason':'not uploaded'});continue
   if p.stat().st_size!=row['bytes']:
    report['unavailable'].append({'path':name,'reason':'size differs from upload manifest'});continue
   data=p.read_bytes()
   if hashlib.sha256(data).hexdigest()!=row['sha256']:
    report['unavailable'].append({'path':name,'reason':'SHA256 differs from upload manifest'});continue
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
   tf.addfile(info,io.BytesIO(data));report['files'].append(row)
 info=tarfile.TarInfo('PULL_MANIFEST.json')
 data=json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8');info.size=len(data);info.mode=0o644
 tf.addfile(info,io.BytesIO(data))
'''

increment = ROOT / 'source_increment'
manifest = json.loads((increment / 'HANDOFF_SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
for row in manifest['files']:
    data = (increment / 'qencbank' / row['path']).read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
print(f"Source increment: {len(manifest['files'])} files SHA256 PASS", flush=True)

archive = ROOT / 'reference_files.tar.gz'
with archive.open('wb') as output:
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpu-node1', 'python3', '-'], input=REMOTE.encode('utf-8'), stdout=output, check=True)
destination = ROOT / 'reference_files'
destination.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
    assert all(m.isfile() for m in tf.getmembers())
    tf.extractall(destination, filter='data')
report = json.loads((destination / 'PULL_MANIFEST.json').read_text(encoding='utf-8'))
for row in report['files']:
    data = (destination / row['path']).read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
print(json.dumps({'verified_files':len(report['files']), 'verified_bytes':sum(r['bytes'] for r in report['files']), 'unavailable_files':len(report['unavailable']), 'archive_bytes':archive.stat().st_size},indent=2))
