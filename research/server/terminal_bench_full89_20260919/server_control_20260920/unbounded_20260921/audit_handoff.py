"""Verify uploaded scientific evidence; never read or copy model artifacts."""
import hashlib,json,tarfile,time
from pathlib import Path

F=Path('/srv/encbank/COMem_Migration_20260920/final_handoff_20260921')
U=Path(__file__).resolve().parent
O=U/'verified_handoff'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ready=json.loads((F/'FINAL_READY.json').read_text())
    assert ready['status']=='PASS',ready
    sums=[]
    for line in (F/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(None,1);name=name.lstrip('*')
        p=(F/name).resolve();assert p.is_relative_to(F)
        assert sha(p)==digest,name
        sums.append(name)
    manifest=json.loads((F/'FILE_SHA256.json').read_text())
    expected={r['archive_path']:r for r in manifest['files']}
    O.mkdir(exist_ok=True)
    seen=set();total=0
    with tarfile.open(F/'LOCAL_RAW_EVIDENCE.tar.gz','r|gz') as t:
        for m in t:
            if m.isdir():continue
            assert m.isfile() and m.name in expected,m.name
            r=expected[m.name];assert m.size==r['bytes']
            p=(O/m.name).resolve();assert p.is_relative_to(O)
            raw=t.extractfile(m).read()
            assert hashlib.sha256(raw).hexdigest()==r['sha256'],m.name
            if p.exists():assert p.read_bytes()==raw
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            seen.add(m.name);total+=len(raw)
    assert seen==set(expected),(len(seen),len(expected))
    report=dict(status='PASS',epoch=time.time(),top_level_files=len(sums),raw_files=len(seen),
                raw_bytes=total,manifest_sha256=sha(F/'FILE_SHA256.json'),
                archive_sha256=sha(F/'LOCAL_RAW_EVIDENCE.tar.gz'),root=str(O),
                no_model_artifacts=True)
    (U/'handoff_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
