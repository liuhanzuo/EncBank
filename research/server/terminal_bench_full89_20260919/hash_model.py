"""CPU-only full shard identity; do not instantiate the model or read evaluators."""
from pathlib import Path
import datetime,hashlib,json,os,time,traceback
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());M=Path(P['model']).resolve()
M.relative_to(Path('/srv/encbank').resolve());H.resolve().relative_to(Path('/srv/encbank').resolve())
O=H/'model_hashes';O.mkdir(exist_ok=True)
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
assert not (O/'owner.json').exists();dump('owner.json',dict(pid=os.getpid(),epoch=time.time()))
try:
    expected=json.loads((H/'cpu_preflight_remote.json').read_text())['files'];rows=[]
    for path in sorted(M.glob('*.safetensors')):
        before=path.stat();assert before.st_size==expected[path.name]['bytes'] and before.st_mtime_ns==expected[path.name]['mtime_ns']
        sha=hashlib.sha256();start=time.monotonic()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*2**20),b''):sha.update(block)
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        row=dict(file=path.name,bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=sha.hexdigest(),seconds=time.monotonic()-start)
        dump(path.name+'.json',row);rows.append(row);print(path.name,'hashed',flush=True)
    assert rows
    dump('completion.json',dict(status='PASS',shards=rows,at=datetime.datetime.now().astimezone().isoformat(),gpu_model_calls=0))
except BaseException:dump('failure.json',dict(error=traceback.format_exc()));raise
