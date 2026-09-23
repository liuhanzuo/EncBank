"""CPU-only exact identity recovery into thread-owned storage, without changing the cache."""
from pathlib import Path
import datetime,hashlib,json,os,time,traceback
H=Path(__file__).resolve().parent;BOUND=Path('/srv/encbank').resolve();H.resolve().relative_to(BOUND)
M=Path('/srv/encbank/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
D=Path('/srv/encbank/qencbank_runtime_20260911/models/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
A=Path('/srv/encbank/qencbank_runtime_20260911/models/Qwen3.8-27B-adapter-final4000-9253fbb9.pt')
ADAPTER=Path('/srv/encbank/encbank_new_backbones_formal_20260915/training/Qwen3.8-27B/adapter-final.pt')
expected=json.loads((H/'expected_model.json').read_text());rows=[]
def dump(n,x):
    p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def digest(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):d.update(block)
    return d.hexdigest()
try:
    assert not D.exists() and not A.exists(),'Do not overwrite an existing model recovery'
    for name,e in expected['files'].items():
        src=(M/name).resolve(strict=True);src.relative_to(BOUND);s=src.stat();assert s.st_size==e['bytes'],name
        start=time.monotonic();actual=digest(src);after=src.stat()
        assert (s.st_size,s.st_mtime_ns)==(after.st_size,after.st_mtime_ns),name
        assert actual==e['sha256'],name
        row=dict(file=name,source=str(src),bytes=s.st_size,sha256=actual,mtime_ns=s.st_mtime_ns,seconds=time.monotonic()-start)
        rows.append(row);dump('hash_progress.json',dict(files=rows,model_calls=0));print(name+' identity PASS',flush=True)
    ap=ADAPTER.resolve(strict=True);ap.relative_to(BOUND);assert digest(ap)==expected['adapter_sha256']
    D.parent.resolve().relative_to(BOUND);D.mkdir(parents=True);D.resolve().relative_to(BOUND)
    # Content-addressed cache blobs are retained by hard links, surviving cache-name removal.
    # No chmod/write on linked content; runtime model files are treated as immutable.
    for row in rows:
        src=Path(row['source']);target=D/row['file'];os.link(src,target)
        assert src.stat().st_ino==target.stat().st_ino and src.stat().st_dev==target.stat().st_dev
    os.link(ap,A);A.resolve().relative_to(BOUND)
    record=dict(status='PASS',at=datetime.datetime.now().astimezone().isoformat(),model=str(D),adapter=str(A),
        source_snapshot=str(M),revision=expected['revision'],files=rows,adapter_sha256=expected['adapter_sha256'],
        original_model=str(expected['original_model']),identity_matches_prior=True,link_kind='hardlink',
        source_content_modified=False,model_calls=0,allocated_GPU_jobs=0,
        omitted_nonruntime_metadata=['download_complete.json'],boundary='Hard links retain immutable cache data if original names are removed. They do not protect against in-place modification; full identity is recorded and launch rechecks apply.')
    dump('model_recovery.json',record);print(json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)
except BaseException:
    dump('model_recovery_failure.json',dict(error=traceback.format_exc(),verified_files=len(rows),model_calls=0));raise
