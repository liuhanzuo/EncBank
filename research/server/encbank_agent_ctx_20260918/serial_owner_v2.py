"""One CPU owner: resume the diagnosed path failure, one formal arm at a time."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, subprocess, time, traceback

H = Path(__file__).resolve().parent
H.relative_to(Path('/srv/encbank').resolve())
assert 'memory' not in str(H)
OLD = H.parent / 'encbank_agent_memory_20260917'
PREFIXES = ('qcm-', 'qencbank-', 'encbank-', 'midcache-', 'cm-')
OUT = H / 'serial_queue_v2'
OUT.mkdir(exist_ok=True)
lock = (OUT / 'owner.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def now(): return datetime.datetime.now().astimezone().isoformat()
def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
def run(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, (argv, p.returncode, p.stderr)
    return p.stdout
def queue():
    raw = run(['squeue', '-r', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b'])
    jobs = []
    for line in raw.splitlines():
        jid, name, state, tres = line.split('|')
        if not name.startswith(PREFIXES): continue
        matches = re.findall(r'(?:^|,)(?:gres/)?gpu(?::[^,:]+)*:(\d+)(?:\([^)]*\))?(?=,|$)', tres)
        assert 'gpu' not in tres or matches, ('Unknown GPU request syntax', line)
        jobs.append(dict(job_id=jid, name=name, state=state, tres=tres, gpus=sum(map(int, matches))))
    return jobs
def state(status, **extra):
    save(OUT / 'status.json', dict(status=status, observed_at=now(), owner_pid=os.getpid(), **extra))
def verify_package():
    manifest = json.loads((H / 'manifest.json').read_text())
    for name, expected in manifest.items():
        path = (H / name).resolve(); path.relative_to(H)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    proof = json.loads((H / 'cpu_world_preflight.json').read_text())
    assert proof['actual_exit_code'] == 0 and proof['actual_parent_wait']
    worlds = json.loads(proof['stdout'])
    assert worlds['status'] == 'SIX_WORLDS_INITIALIZE_EXECUTE_CLOSE_PASS'
    assert [r['task_id'] for r in worlds['rows']] == plan['task_ids']
    for arm in plan['arms']:
        receipts = json.loads((OLD / ('run_' + arm) / 'process_receipts.json').read_text())
        assert receipts['worker']['exit_code'] == 0 and receipts['worker']['actual_parent_wait']
        assert receipts['actor']['exit_code'] == 1 and receipts['actor']['actual_parent_wait']
        assert json.loads((OLD / ('run_' + arm) / 'worker_complete.json').read_text())['requests'] == 0

plan = json.loads((H / 'agent_plan.json').read_text())
assert hashlib.sha256((H / 'agent_plan.json').read_bytes()).hexdigest() == 'b227ff5b9d568ce1a8a5ad36ca6fe7875a49b3083842b3a38ee9fea880c0cd12'
assert not (OUT / 'registration.json').exists(), 'Do not restart an existing owner without reconciliation'
save(OUT / 'registration.json', dict(started_at=now(), owner_pid=os.getpid(),
    proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    arms=plan['arms'], gpu_request_limit=4, prefixes_counted=PREFIXES,
    scientific_plan_sha256=hashlib.sha256((H / 'agent_plan.json').read_bytes()).hexdigest()))
try:
    verify_package()
    closed = []
    for arm in plan['arms']:
        receipt = OUT / (arm + '_submission.json')
        target = H / ('run_' + arm)
        assert not receipt.exists() and not target.exists(), 'Existing attempt: stop, never duplicate'
        while True:
            jobs = queue()
            assert not any('qencbank-agentmem-' in j['name'] for j in jobs), 'Another AppWorld job is active'
            count = sum(j['gpus'] for j in jobs)
            if count < 4: break
            state('waiting_resource_budget', next_arm=arm, prior_gpu_requests=count, jobs=jobs, closed_arms=closed)
            time.sleep(45)
        verify_package()
        # Admission must be fresh after verification, which reads all frozen input hashes.
        jobs = queue(); count = sum(j['gpus'] for j in jobs)
        if count >= 4: raise RuntimeError('Budget changed at admission; stopped without submission')
        assert not any('qencbank-agentmem-' in j['name'] for j in jobs)
        argv = ['sbatch', '--parsable', str(H / (arm + '.slurm'))]
        record = dict(at=now(), arm=arm, argv=argv, prior_jobs=jobs, prior_gpu_requests=count, status='submission_intent')
        save(receipt, record)
        # A timeout or uncertain return stops the owner; the persisted intent prevents a blind retry.
        child = subprocess.run(argv, cwd=H, capture_output=True, text=True, timeout=30)
        record.update(actual_exit_code=child.returncode, actual_parent_wait=True, stdout=child.stdout, stderr=child.stderr)
        save(receipt, record)
        assert child.returncode == 0, child.stderr
        jid = child.stdout.strip().split(';')[0]; assert jid.isdigit()
        record.update(job_id=jid, status='submitted'); save(receipt, record)
        missing = 0
        while True:
            active = run(['squeue', '-h', '-j', jid, '-o', '%i|%T|%R']).strip()
            if active:
                state('slurm_active', arm=arm, job_id=jid, scheduler=active, closed_arms=closed)
                time.sleep(45); continue
            accounting = run(['sacct', '-j', jid, '-n', '-P', '--format=JobID,State,ExitCode,Start,End,Elapsed'])
            rows = [line.split('|') for line in accounting.splitlines() if line]
            parent = next((r for r in rows if r[0] == jid), None)
            if parent is None or parent[1] in ('PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING'):
                missing += 1; assert missing <= 40, 'Accounting unavailable after 30 minutes; inspect rather than resubmit'
                state('waiting_accounting', arm=arm, job_id=jid, closed_arms=closed)
                time.sleep(45); continue
            save(OUT / (arm + '_accounting.json'), dict(observed_at=now(), raw=accounting))
            assert parent[1] == 'COMPLETED' and parent[2] == '0:0', ('Terminal failure; no auto retry', parent)
            complete = json.loads((target / 'complete.json').read_text())
            receipts = json.loads((target / 'process_receipts.json').read_text())
            for name in ('worker', 'actor', 'evaluator'):
                assert complete['actual_exit_codes'][name] == 0
                assert receipts[name]['exit_code'] == 0 and receipts[name]['actual_parent_wait']
            actor = json.loads((target / 'actor_results.json').read_text())
            scores = json.loads((target / 'official_scores.json').read_text())
            assert actor['status'] == 'completed' and scores['status'] == 'complete'
            assert [r['task_id'] for r in actor['rows']] == plan['task_ids']
            assert [r['task_id'] for r in scores['rows']] == plan['task_ids']
            closed.append(arm)
            state('arm_closed', arm=arm, job_id=jid, closed_arms=closed, analysis_pending=True)
            break
    state('all_four_arms_closed_analysis_pending', closed_arms=closed)
except BaseException:
    state('stopped_failure_no_retry', error=traceback.format_exc())
    raise
