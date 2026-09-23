"""Probe Docker-free Apptainer instances and delegated resource limits on the server."""
import json, os, subprocess, time, select
from pathlib import Path

ROOT = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
job = os.environ['SLURM_JOB_ID']
out = ROOT / ('instance-probe-' + job)
out.mkdir(mode=0o700)
env = dict(os.environ, APPTAINER_CONFIGDIR=str(out / 'config'), APPTAINER_TMPDIR=str(ROOT/'tmp'),
           APPTAINER_CACHEDIR=str(ROOT/'apptainer_cache'), TMPDIR=str(ROOT/'tmp'))
name = 'comem-check-' + job
report = dict(job=job, model_calls=0, checks={})
group = None
network = None
def run(key, args, timeout=60):
    def join():
        (group/'cgroup.procs').write_text(str(os.getpid()))
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout, env=env,
                       preexec_fn=join if group and args[:2] == ['apptainer','exec'] else None)
    report['checks'][key] = dict(code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(key, p.returncode, p.stdout[-1500:], p.stderr[-1500:], flush=True)
    return p
image = ROOT/'sif/alexgshaw_cancel-async-tasks_20251031.sif'
try:
    cmd = ['systemd-run','--user','--scope','--quiet','--unit='+name,'-p','MemoryMax=512M','-p','CPUQuota=100%',
           'apptainer','instance','start','--fakeroot','--containall','--writable-tmpfs',
           '--no-mount','home,tmp,bind-paths','--net','--network','none','--dns','192.0.2.1',
           str(image),name]
    p = run('start_with_scope',cmd)
    if not p.returncode:
        listing = run('list',['apptainer','instance','list','--json',name])
        pid = json.loads(listing.stdout)['instances'][0]['pid']
        cgroup = run('group_path',['systemctl','--user','show',name+'.scope','-p','ControlGroup','--value']).stdout.strip()
        group = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
        run('host_limits',['cat',str(group/'memory.max'),str(group/'cpu.max')])
        report['network_namespace'] = os.readlink(f'/proc/{pid}/ns/net')
        report['host_network_namespace'] = os.readlink('/proc/self/ns/net')
        ready_read, ready_write = os.pipe()
        log = (out/'slirp.log').open('wb')
        network = subprocess.Popen(['slirp4netns','--configure','--disable-host-loopback',
            '--userns-path='+f'/proc/{pid}/ns/user','--ready-fd='+str(ready_write),str(pid),'tap0'],
            stdout=log,stderr=log,pass_fds=(ready_write,),env=env)
        os.close(ready_write)
        assert select.select([ready_read],[],[],20)[0] and os.read(ready_read,1) == b'1', 'network not ready'
        os.close(ready_read)
        run('first_exec',['apptainer','exec','--userns','instance://'+name,'sh','-c',
            'echo ok > /tmp/persist; id; readlink /proc/self/ns/net; cat /proc/self/cgroup'])
        run('second_exec',['apptainer','exec','--userns','instance://'+name,'cat','/tmp/persist'])
        run('scope',['systemctl','--user','show',name+'.scope','-p','ControlGroup','-p','MemoryMax','-p','CPUQuotaPerSecUSec'])
        run('internet',['apptainer','exec','--userns','instance://'+name,'python3','-c',
            'import urllib.request; print(urllib.request.urlopen("https://example.com",timeout=15).status)'])
finally:
    if network:
        network.terminate()
        network.wait(timeout=10)
    run('stop',['apptainer','instance','stop','--force',name])
    run('list_after',['apptainer','instance','list','--json',name])
