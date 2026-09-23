"""Preserve completed in-memory old-worker replies before authorized shutdown."""
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

B = Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919')
OUT = B/'server_control_20260920/unbounded_20260921/old_worker_snapshots'

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, str(path)
        return
    for attempt in range(5):
        try:
            path.write_bytes(data)
            assert path.read_bytes() == data
            return
        except OSError:
            if attempt == 4: raise
            time.sleep(2**attempt)

def main():
    reports = []
    for arm, job in [('k12','112400'),('k48','112403')]:
        p = B/f'comem_{arm}_no_task_deadline_r6_20260920'
        w = p/'run_comem'
        d = OUT/arm
        spec = importlib.util.spec_from_file_location('old_mailbox',p/'live_mailbox.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        endpoint = json.loads((w/'transport_endpoint.json').read_text())
        events = [json.loads(x) for x in (w/'events.jsonl').read_text().splitlines() if x.strip()]
        starts = {x['request_id']:x for x in events if x['event']=='request_start'}
        done = {x['request_id'] for x in events if x['event']=='request_complete'}
        released = {x['task_id'] for x in events if x['event']=='session_release'}
        candidates = done | {rid for rid in starts if rid.rsplit('_',1)[0] in released}
        def fetch(rid):
            dest = d/'responses'/(rid+'.response.json')
            if dest.exists(): raw=dest.read_bytes()
            else:
                proof=module.client(endpoint,{'action':'wait','id':rid},timeout=30)
                raw=base64.b64decode(proof['payload_base64'],validate=True)
                assert len(raw)==proof['bytes'] and hashlib.sha256(raw).hexdigest()==proof['sha256']
                save(dest,raw)
            r=json.loads(raw)
            assert r['request_id']==rid
            return dict(request_id=rid,task=r['task'],step=r['step'],status=r['status'],
                        generated_tokens=r.get('generated_tokens',0),hit_generation_cap=r.get('hit_generation_cap',False),
                        request_seconds=r.get('request_seconds'),state_hashes_unchanged=r.get('state_hashes_unchanged'),
                        bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),path=str(dest))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(fetch,sorted(candidates)))
        stamp=str(time.time_ns())
        for name in ['events.jsonl','worker_ready.json','process_start.json']:
            save(d/(stamp+'_'+name),(w/name).read_bytes())
        report=dict(arm=arm,job_id=job,epoch=time.time(),source=str(p),responses=rows,
                    active_or_unfinished=[v for k,v in starts.items() if k not in candidates],
                    source_plan_sha256=hashlib.sha256((p/'plan.json').read_bytes()).hexdigest())
        save(d/(stamp+'_snapshot.json'),(json.dumps(report,indent=2)+'\n').encode())
        reports.append({k:v for k,v in report.items() if k!='responses'})
        print(json.dumps(dict(arm=arm,preserved=len(rows),tokens=sum(r['generated_tokens'] for r in rows),
                              capped=sum(r['hit_generation_cap'] for r in rows),snapshot=str(d/(stamp+'_snapshot.json')))),flush=True)
    save(OUT/(str(time.time_ns())+'_index.json'),(json.dumps(reports,indent=2)+'\n').encode())

if __name__=='__main__': main()
