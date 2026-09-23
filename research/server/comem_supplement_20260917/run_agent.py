"""Single parent owns worker/actor, records actual waits and never retries failed runs."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H/'agent_qwen27b_attempt1';R.mkdir(exist_ok=False);(R/'mailbox').mkdir()
GPU='/srv/encbank/Paper_Evolve/.venv/bin/python'
CPU='/srv/encbank/qcomem_runtime_20260911/appworld_cpu_20260917'+"/bin/python"
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
commands={n:[py,'-X','utf8','-u',str(H/file),'--run',str(R)] for n,py,file in [('worker',GPU,'agent_worker.py'),('actor',CPU,'agent_actor.py'),('evaluator',CPU,'evaluate_agent.py')]}
save(R/'registration.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),commands=commands,
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in H.glob('*.py')},plan_sha256=hashlib.sha256((H/'agent_plan.json').read_bytes()).hexdigest()))
receipts={};worker=None
def start(name):
    stdout=(R/(name+'.stdout.log')).open('wb');stderr=(R/(name+'.stderr.log')).open('wb')
    child=subprocess.Popen(commands[name],stdout=stdout,stderr=stderr,cwd=H)
    receipts[name]=dict(pid=child.pid,started_at=datetime.datetime.now().astimezone().isoformat(),exit_code=None)
    save(R/'process_receipts.json',receipts);stdout.close();stderr.close();return child
def wait(name,child):
    rc=child.wait();receipts[name].update(exit_code=rc,actual_parent_wait=True,ended_at=datetime.datetime.now().astimezone().isoformat());save(R/'process_receipts.json',receipts);return rc
try:
    worker=start('worker');deadline=time.monotonic()+7*3600
    while not (R/'worker_ready.json').exists():
        if worker.poll() is not None: assert wait('worker',worker)==0,'Worker failed before readiness'
        if time.monotonic()>deadline: raise TimeoutError('GPU admission/readiness deadline')
        time.sleep(2)
    actor=start('actor');arc=wait('actor',actor)
    save(R/'mailbox/stop.json',dict(reason='actor exited',actual_exit_code=arc))
    wrc=wait('worker',worker);assert arc==wrc==0, 'Actor or worker failed; partial outputs retained'
    evaluator=start('evaluator');erc=wait('evaluator',evaluator);assert erc==0,'Official scorer infrastructure failure; not zero success'
    save(R/'complete.json',dict(complete=True,finished_at=datetime.datetime.now().astimezone().isoformat(),actual_exit_codes=dict(actor=arc,worker=wrc,evaluator=erc)))
finally:
    if worker is not None and worker.poll() is None:
        save(R/'mailbox/stop.json',dict(reason='parent finalization'))
        try: worker.wait(timeout=900)
        except subprocess.TimeoutExpired: worker.terminate();worker.wait()
        receipts['worker'].update(exit_code=worker.returncode,actual_parent_wait=True);save(R/'process_receipts.json',receipts)
