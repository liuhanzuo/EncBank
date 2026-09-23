"""Isolated research-run state, never alters the 27B benchmark owners."""
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text()) if (ROOT/'plan.json').exists() else {}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}-{time.time_ns()}')
    with temp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)
def verify_sources():
    for rel,expected in json.loads((ROOT/'source_manifest.json').read_text()).items():
        assert sha(ROOT/rel)==expected,rel
