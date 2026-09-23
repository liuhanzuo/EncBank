import datetime,json,os,subprocess,sys,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent
ENV=Path('/srv/encbank/qencbank_runtime_20260911/appworld_cpu_20260917')
ENV.resolve().relative_to(Path('/srv/encbank').resolve())
assert not ENV.exists()
tmp=H/'tmp';tmp.mkdir(exist_ok=True)
os.environ.update(TMPDIR=str(tmp),PIP_CACHE_DIR=str(H/'pip_cache'),APPWORLD_ROOT=str(H/'appworld_public'),APPWORLD_CACHE=str(H/'app_cache'),CUDA_VISIBLE_DEVICES='-1')
src=H/'appworld_source';src.mkdir(exist_ok=False)
with zipfile.ZipFile(H/'appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5.zip') as z:
    for info in z.infolist():
        if info.is_dir():continue
        rel=Path(*Path(info.filename).parts[1:]);target=src/rel;target.resolve().relative_to(src.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
for name in ['apps','tests']:(src/f'src/appworld/.source/{name}.bundle').write_bytes((H/f'{name}.bundle').read_bytes())
py=str(ENV/'bin/python');commands=[[sys.executable,'-m','venv',str(ENV)],[py,'-m','pip','install','--disable-pip-version-check',str(src)],[py,'-m','appworld.cli','install'],[py,'-m','pip','check']]
records=[]
for i,argv in enumerate(commands):
    with (H/f'env_{i}.stdout.log').open('wb') as out,(H/f'env_{i}.stderr.log').open('wb') as err:
        child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=H);rc=child.wait()
    records.append(dict(argv=argv,pid=child.pid,actual_exit_code=rc,actual_parent_wait=True))
    (H/'environment_receipts.json').write_text(json.dumps(records,indent=2))
    assert rc==0,'Environment infrastructure failure; no automatic retry'
(H/'environment_complete.json').write_text(json.dumps(dict(complete=True,python=py,at=datetime.datetime.now().astimezone().isoformat())))
