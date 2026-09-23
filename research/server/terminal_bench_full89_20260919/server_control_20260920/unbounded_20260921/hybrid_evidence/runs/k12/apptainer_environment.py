"""Docker-free Harbor backend using managed Apptainer instances, without a container HTTP server."""
import asyncio
import json
import os
import shlex
import socket
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentResourceCapabilities, EnvironmentCapabilities

ROOT = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')


class ManagedApptainerEnvironment(SingularityEnvironment):
    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities(memory_limit=True)

    @property
    def capabilities(self):
        return EnvironmentCapabilities(mounted=True,disable_internet=True)

    async def start(self, force_build=False):
        self._name = 'qcomem-ai-'+uuid.uuid4().hex[:12]
        self._root = ROOT/'managed_instances'/self._name
        self._root.mkdir(parents=True,mode=0o700)
        self._staging_dir = self._root/'staging'
        self._staging_dir.mkdir()
        self._sif_path = Path(self._docker_image) if self._is_sif_image else await self._convert_docker_to_sif(self._docker_image,force_pull=force_build)
        sockets = ROOT.parent/'ai'
        sockets.mkdir(mode=0o700,exist_ok=True)
        self._socket = sockets/(self._name+'.sock')
        binds = [[str(self._staging_dir),'/staging']]
        for mount in self._mounts:
            if mount.get('type')=='bind':
                Path(mount['source']).mkdir(parents=True,exist_ok=True)
                binds.append([mount['source'],mount['target']])
        affinity = sorted(os.sched_getaffinity(0))[:max(1,int(self._effective_cpus or 1))]
        spec = dict(root=str(self._root),name=self._name,image=str(self._sif_path),binds=binds,
                    workdir=self._workdir,socket=str(self._socket),memory_mb=self._effective_memory_mb or 2048,
                    storage_mb=self._effective_storage_mb or 10240,
                    affinity=affinity,internet=self.task_env_config.allow_internet,
                    tmp=str(ROOT/'tmp'),cache=str(ROOT/'apptainer_cache'),owner_pid=os.getpid(),
                    owner_ticks=(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
        spec['proxy_env']={key:os.environ[key] for key in ['http_proxy','https_proxy','all_proxy','no_proxy',
                           'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'] if os.environ.get(key)}
        for key in ['http_proxy','https_proxy','all_proxy','no_proxy']:
            if key not in spec['proxy_env'] and key.upper() in spec['proxy_env']:
                spec['proxy_env'][key]=spec['proxy_env'][key.upper()]
        (self._root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        command = ['systemd-run','--user','--quiet','--collect','--unit='+self._name,
                   '-p','Type=exec','-p','Restart=no','-p','KillMode=control-group','-p','TimeoutStopSec=35',
                   '-p','RuntimeMaxSec=36h','-p','MemoryMax='+str(spec['memory_mb'])+'M',
                   str(ROOT/'harbor_env/bin/python'),str(Path(__file__).with_name('apptainer_service.py')),str(self._root/'spec.json')]
        result = await asyncio.to_thread(subprocess.run,command,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        try:
            for _ in range(150):
                if (self._root/'service.json').exists():
                    record=json.loads((self._root/'service.json').read_text())
                    if record['state']=='FAILED':
                        raise RuntimeError(record.get('error'))
                    if record['state']=='READY':
                        break
                await asyncio.sleep(1)
            else:
                raise TimeoutError('Apptainer instance startup deadline exceeded')
            diagnostic = await self.exec('cat /etc/resolv.conf; cat /proc/net/route; readlink /proc/self/ns/net; cat /proc/self/cgroup',timeout_sec=15)
            (self._root/'network_diagnostic.json').write_text(diagnostic.model_dump_json(indent=2))
            bootstrap = '''set -e
mkdir -p /tmp/comem-apt/partial /logs/agent /logs/verifier
printf 'APT::Sandbox::User "root";\\nDir::Cache::archives "/tmp/comem-apt";\\n' > /etc/apt/apt.conf.d/99comem-runtime
if ! command -v tmux >/dev/null || ! command -v asciinema >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux asciinema
fi
command -v tmux
command -v asciinema
'''
            result = await self.exec(bootstrap,timeout_sec=300)
            (self._root/'bootstrap.json').write_text(result.model_dump_json(indent=2))
            assert result.return_code==0,result.stderr[-5000:]
            shutil.copy2(Path(__file__).with_name('apptainer_executor.py'),self._staging_dir/'.comem_executor.py')
            await self._rpc({'op':'start_executor'},30)
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(True)
            raise

    async def exec(self,command,cwd=None,env=None,timeout_sec=None,user=None):
        resolved = self._resolve_user(user)
        if resolved is not None:
            command='su '+shlex.quote(str(resolved))+' -s /bin/bash -c '+shlex.quote(command)
        request=dict(op='exec',command=command,cwd=cwd,env=self._merge_env(env),timeout=timeout_sec)
        return ExecResult(**(await self._rpc(request,None if timeout_sec is None else timeout_sec+45)))

    async def _rpc(self,request,timeout):
        def call():
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
                stream.settimeout(timeout)
                stream.connect(str(self._socket))
                stream.sendall((json.dumps(request)+'\n').encode())
                with stream.makefile('rb') as response:
                    result=json.loads(response.readline())
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result
        return await asyncio.to_thread(call)

    async def stop(self,delete=True):
        if not hasattr(self,'_name'):
            return
        result = await asyncio.to_thread(subprocess.run,['systemctl','--user','stop',self._name+'.service'],
                                        capture_output=True,text=True,timeout=50)
        state = await asyncio.to_thread(subprocess.run,['systemctl','--user','is-active',self._name+'.service'],
                                       capture_output=True,text=True,timeout=10)
        assert state.stdout.strip() not in {'active','activating','deactivating'},state.stdout
        status_path=self._root/'service.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        group=Path('/sys/fs/cgroup')/status.get('cgroup','missing').lstrip('/')
        events=(group/'cgroup.events').read_text() if (group/'cgroup.events').exists() else 'populated 0'
        assert 'populated 0' in events, 'Container service still has live processes'
        (self._root/'closure.json').write_text(json.dumps(dict(service_state=state.stdout.strip(),stop_code=result.returncode,
                                                             cgroup_empty=True))+'\n')
