from pathlib import Path
import hashlib,json,os,platform,time
h=Path(__file__).resolve().parent;h.relative_to(Path('/srv/encbank').resolve())
assert not (h/'storage_preflight.json').exists()
d=h/'storage_preflight';d.mkdir(exist_ok=False);rows=[]
for i in range(8):
 data=(('qencbank filesystem preflight '+str(i)+'\n')*2048).encode();p=d/(str(i)+'.tmp');out=d/(str(i)+'.bin');start=time.monotonic()
 with p.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.replace(out);assert out.read_bytes()==data
 rows.append(dict(index=i,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),seconds=time.monotonic()-start))
(h/'storage_preflight.json').write_text(json.dumps(dict(status='PASS',host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),rows=rows,model_calls=0),indent=2)+'\n')
