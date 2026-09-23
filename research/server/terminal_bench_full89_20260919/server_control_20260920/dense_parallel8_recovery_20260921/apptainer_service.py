"""Per-task user service: all container commands inherit one memory cgroup."""
import json
import fcntl
import os
import select
import signal
import socketserver
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def start_ticks(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path(spec['root'])
    root.resolve().relative_to(Path('/srv/encbank'))
    os.umask(0o077)
    os.sched_setaffinity(0, spec['affinity'])
    env = dict(os.environ, APPTAINER_CONFIGDIR=str(root/'config'),
               APPTAINER_TMPDIR=spec['tmp'], APPTAINER_CACHEDIR=spec['cache'], TMPDIR=spec['tmp'])
    name = spec['name']
    shutdown = threading.Event()
    network = None
    server = None
    executor = None
    def cli(args, timeout=90):
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        if p.returncode:
            raise RuntimeError(p.stderr[-8000:] or p.stdout[-8000:])
        return p.stdout
    def identity_alive():
        try:
            return start_ticks(spec['owner_pid']) == spec['owner_ticks']
        except (FileNotFoundError, ProcessLookupError):
            return False
    def watch():
        while not shutdown.wait(1):
            if not identity_alive():
                shutdown.set()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    threading.Thread(target=watch, daemon=True).start()
    report = {'state':'STARTING','pid':os.getpid(),'name':name}
    def save():
        (root/'service.tmp').write_text(json.dumps(report,indent=2)+'\n')
        (root/'service.tmp').replace(root/'service.json')
    try:
        group_name = next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        group = Path('/sys/fs/cgroup')/group_name.lstrip('/')
        report.update(cgroup=group_name,memory_max=(group/'memory.max').read_text().strip(),
                      affinity=sorted(os.sched_getaffinity(0)), cpu_quota_enforced=False)
        assert int(report['memory_max']) == spec['memory_mb']*1024*1024
        overlay = root/'overlay.img'
        cli(['apptainer','overlay','create','--fakeroot','--sparse','--size',str(spec['storage_mb']),str(overlay)],timeout=120)
        cmd = ['apptainer','instance','start','--fakeroot','--containall','--overlay',str(overlay),
               '--no-mount','home,tmp,bind-paths','--net','--network','none','--dns','192.0.2.1']
        (root/'resolv.conf').write_text('nameserver 192.0.2.1\noptions timeout:3 attempts:2\n')
        cmd.extend(['-B',str(root/'resolv.conf')+':/etc/resolv.conf:ro'])
        for source,target in spec['binds']:
            Path(source).resolve().relative_to(Path('/srv/encbank'))
            cmd.extend(['-B',source+':'+target])
        # Serialize mount startup only; model requests and task execution remain concurrent.
        with Path(spec['startup_lock']).open('a') as startup_lock:
            fcntl.flock(startup_lock,fcntl.LOCK_EX)
            cli(cmd+[spec['image'],name])
        instance = json.loads(cli(['apptainer','instance','list','--json',name]))['instances'][0]
        pid = instance['pid']
        report.update(instance_pid=pid, instance_ticks=start_ticks(pid),
                      network=os.readlink(f'/proc/{pid}/ns/net'), host_network=os.readlink('/proc/self/ns/net'))
        assert report['network'] != report['host_network']
        if spec['internet']:
            reader,writer = os.pipe()
            with (root/'network.log').open('wb') as log:
                network = subprocess.Popen(['slirp4netns','--configure','--disable-host-loopback',
                    '--userns-path='+f'/proc/{pid}/ns/user','--ready-fd='+str(writer),str(pid),'tap0'],
                    pass_fds=(writer,),stdout=log,stderr=log,env=env)
            os.close(writer)
            try:
                assert select.select([reader],[],[],20)[0] and os.read(reader,1)==b'1', 'network startup failed'
            finally:
                os.close(reader)
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                nonlocal executor
                try:
                    request = json.loads(self.rfile.readline())
                    if request['op']=='exec':
                        task_env=dict(TMPDIR='/tmp',TMP='/tmp',TEMP='/tmp',XDG_CACHE_HOME='/tmp/.cache')
                        task_env.update(spec.get('proxy_env',{}))
                        task_env.update(request.get('env') or {})
                        request['env']=task_env
                        if executor is not None:
                            assert executor.poll() is None,'Persistent command parent exited'
                            fd=os.open(root/'staging',os.O_RDONLY|os.O_DIRECTORY)
                            try:
                                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                                    client.settimeout(None if request.get('timeout') is None else request['timeout']+15)
                                    client.connect(f'/proc/self/fd/{fd}/executor.sock')
                                    client.sendall((json.dumps(request)+'\n').encode())
                                    result=json.loads(client.makefile('rb').readline())
                            finally:
                                os.close(fd)
                            self.wfile.write((json.dumps(result)+'\n').encode())
                            return
                        args = ['apptainer','exec','--userns','--pwd',request.get('cwd') or spec['workdir'],
                                'instance://'+name,'env']
                        args.extend(key+'='+value for key,value in task_env.items())
                        seconds = request.get('timeout')
                        if seconds is not None:
                            args.extend(['timeout','--signal=TERM','--kill-after=5',str(seconds)])
                        args.extend(['bash','-c',request['command']])
                        p = subprocess.run(args,capture_output=True,text=True,env=env,
                                           timeout=None if seconds is None else seconds+30)
                        result = dict(stdout=p.stdout,stderr=p.stderr,return_code=p.returncode)
                    elif request['op']=='start_executor':
                        assert executor is None
                        with (root/'executor.log').open('wb') as log:
                            executor=subprocess.Popen(['apptainer','exec','--userns','--pwd',spec['workdir'],
                                'instance://'+name,'env','TMPDIR=/tmp','python3','/staging/.comem_executor.py'],
                                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                        for _ in range(100):
                            assert executor.poll() is None,'Executor startup failed; see executor.log'
                            if (root/'staging/executor.sock').exists():
                                break
                            time.sleep(.1)
                        else:
                            raise TimeoutError('Executor socket not created')
                        result={'executor_pid':executor.pid}
                    elif request['op']=='status':
                        result = report
                    else:
                        raise ValueError('Unknown operation')
                except Exception as error:
                    result = {'error':repr(error)}
                self.wfile.write((json.dumps(result)+'\n').encode())
        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
        server = Server(spec['socket'],Handler)
        server.timeout = 1
        report['state']='READY'
        save()
        while not shutdown.is_set():
            if network and network.poll() is not None:
                raise RuntimeError('Container network process exited')
            if executor and executor.poll() is not None:
                raise RuntimeError('Persistent command parent exited')
            server.handle_request()
    except Exception as error:
        report.update(state='FAILED',error=repr(error))
        save()
        raise
    finally:
        if server:
            server.server_close()
        Path(spec['socket']).unlink(missing_ok=True)
        if network and network.poll() is None:
            network.terminate()
        stopped = subprocess.run(['apptainer','instance','stop','--force',name],env=env,capture_output=True,text=True,timeout=30)
        report.update(state='STOPPED' if report['state']!='FAILED' else 'FAILED',stop_code=stopped.returncode)
        save()


if __name__=='__main__':
    main()
