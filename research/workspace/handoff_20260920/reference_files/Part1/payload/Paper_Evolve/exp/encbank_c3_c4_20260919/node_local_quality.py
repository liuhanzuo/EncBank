"""Node-local IO recovery only; copy and execute unchanged scientific evaluator."""
import argparse,datetime,hashlib,json,os,platform,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--j',type=int,required=True);p.add_argument('--previous-job',required=True);p.add_argument('--saved-sha256',required=True);a=p.parse_args()
assert a.j in [6,12,18] and os.getuid()==20021
canonical=Path('/srv/encbank/encbank_c3_c4_20260919')
job=os.environ['SLURM_JOB_ID'];root=Path('/tmp/qcm-c34-quality-liuhanzuo-'+job)
root.mkdir(mode=0o700)
def dump(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temp.replace(path)
hashes={}
for name in ['config.py','controlled_train_core.py','evaluate.py','data/quality.jsonl.gz','data/quality_spec.json']:
    data=(canonical/name).read_bytes();dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    hashes[name]=hashlib.sha256(data).hexdigest();assert dest.read_bytes()==data
assert hashes['data/quality.jsonl.gz']=='6d22191d46f5925596a99c8da172bfccae050a785a59de22902f1c3540e0b04e'
(root/'training').symlink_to(canonical/'training',target_is_directory=True)
quality=root/'quality'/('j%d_s42'%a.j);quality.mkdir(parents=True)
saved=(canonical/'quality'/('j%d_s42'%a.j)/'predictions.jsonl').read_bytes()
assert hashlib.sha256(saved).hexdigest()==a.saved_sha256 and saved.endswith(b'\n')
rows=[json.loads(s) for s in saved.splitlines()];assert len(rows)==len({(r['id'],r['arm']) for r in rows})
(quality/'predictions.jsonl').write_bytes(saved)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',CPATH='/srv/encbank/.cache/python/include/python3.12')
env['PYTHONPATH']=':'.join([str(root),'/srv/encbank/encbank_infra_recheck_20260912/deps','/srv/encbank/encbank_followups_20260913/Encbank','/srv/encbank/encbank_frozen_j12_20260912'])
for key,folder in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
    target=root/folder;target.mkdir();env[key]=str(target)
dump(root/'stage.json',dict(job=job,j=a.j,node=platform.node(),root=str(root),previous_job=a.previous_job,saved_records=len(rows),saved_sha256=a.saved_sha256,files_sha256=hashes,scientific_code_unchanged=True,shared_paths_read_only=True))
python='/srv/encbank/Paper_Evolve/.venv/bin/python'
probe="from triton.runtime.cache import FileCacheManager; from pathlib import Path; import os; p=Path(FileCacheManager('c34-local-quality-probe').put(b'ok','probe.bin',binary=True)); assert str(p).startswith(os.environ['TRITON_CACHE_DIR']) and p.read_bytes()==b'ok'"
with (root/'child.stdout.log').open('ab') as stdout,(root/'child.stderr.log').open('ab') as stderr:
    check=subprocess.run([python,'-B','-c',probe],cwd=str(root),env=env,stdout=stdout,stderr=stderr)
    assert check.returncode==0
    dump(quality/'cache_location.json',dict(root=str(root),node=platform.node(),job=job,original_put_and_cleanup_passed=True,scientific_configuration_unchanged=True,outputs_also_node_local=True))
    child=subprocess.Popen([python,'-B','-u',str(root/'evaluate.py'),'--j',str(a.j)],cwd=str(root),env=env,stdout=stdout,stderr=stderr)
    dump(root/'start.json',dict(child_pid=child.pid,at=datetime.datetime.utcnow().isoformat()+'Z'))
    code=child.wait()
dump(quality/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=child.pid,job=job,node=platform.node(),at=datetime.datetime.utcnow().isoformat()+'Z'))
raise SystemExit(code)
