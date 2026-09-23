"""Parent owns the worker process group, records real wait and aggregate NVML."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from persistence import dump
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
P=json.loads((H/'plan.json').read_text());R=H/'run_dense';assert not R.exists();(R/'mailbox').mkdir(parents=True)
args=[P['engine_python'],'-u',str(H/'agent_worker.py')]
with (R/'worker.stdout.log').open('wb') as out,(R/'worker.stderr.log').open('wb') as err:
    p=subprocess.Popen(args,stdout=out,stderr=err,cwd=H,start_new_session=True)
    dump(R/'process_start.json',dict(pid=p.pid,pgid=p.pid,parent_pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],argv=args,started_epoch=time.time()))
    cap_triggered=False;boot_started=time.monotonic();boot_failed=False
    with (R/'owned_nvml.jsonl').open('w') as mon:
        while p.poll() is None:
            if not (R/'worker_ready.json').exists() and time.monotonic()-boot_started>1200 and not boot_failed:
                boot_failed=True;dump(R/'bootstrap_timeout.json',dict(classification='INFRASTRUCTURE_PRE_MODEL_READINESS_TIMEOUT',epoch=time.time(),seconds=1200))
                os.killpg(p.pid,signal.SIGTERM)
            ps=subprocess.run(['ps','-e','-o','pid=,pgid='],capture_output=True,text=True,timeout=20)
            owned={int(line.split()[0]) for line in ps.stdout.splitlines() if len(line.split())==2 and int(line.split()[1])==p.pid}
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
            rows=[];mib=0
            for line in q.stdout.splitlines():
                parts=[x.strip() for x in line.split(',')]
                if len(parts)==3 and parts[0].isdigit() and int(parts[0]) in owned:
                    rows.append(line)
                    if parts[1].isdigit():mib+=int(parts[1])
            mon.write(json.dumps(dict(epoch=time.time(),owned_pids=sorted(owned),rows=rows,aggregate_mib=mib,query_exit=q.returncode))+'\n');mon.flush()
            if mib>P['device_cap_gib']*1024 and not cap_triggered:
                cap_triggered=True;dump(R/'memory_cap_failure.json',dict(aggregate_mib=mib,cap_gib=P['device_cap_gib'],epoch=time.time(),owned_pgid=p.pid))
                os.killpg(p.pid,signal.SIGTERM)
            time.sleep(2)
    code=p.wait();receipt=dict(exit_code=code,actual_parent_wait=True,pid=p.pid,pgid=p.pid,ended_epoch=time.time(),memory_cap_triggered=cap_triggered)
    print(json.dumps(dict(event='actual_worker_wait',**receipt)),flush=True);dump(R/'process_receipt.json',receipt)
sys.exit(code if code>=0 else 128-code)
