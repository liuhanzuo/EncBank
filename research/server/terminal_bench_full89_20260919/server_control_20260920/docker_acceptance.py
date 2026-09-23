"""Server-side Docker acceptance check; operates only on its own bounded container."""
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')


def main():
    assert os.name == 'posix'
    token = uuid.uuid4().hex[:12]
    output = ROOT / ('docker-acceptance-' + token)
    output.mkdir(mode=0o700)
    name = 'qcomem-container-check-' + token
    container_id = None
    report = dict(status='RUNNING', hostname=socket.gethostname(), epoch=time.time(),
                  model_calls=0, benchmark_attempts=0, container_name=name)
    def docker(*args, timeout=60):
        proc = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        return proc.stdout.strip()
    try:
        server = json.loads(docker('version', '--format', '{{json .Server}}'))
        assert isinstance(server, dict) and server.get('Version')
        report['server_version'] = server['Version']
        report['compose_version'] = docker('compose', 'version')
        image = 'alexgshaw/cancel-async-tasks:20251031'
        docker('pull', image, timeout=300)
        container_id = docker('create', '--name', name, '--label', 'comem.acceptance=' + token,
            '--memory', '512m', '--cpus', '1', '--pids-limit', '128',
            '--mount', 'type=bind,src=' + str(output) + ',dst=/probe', image,
            'sh', '-c', 'sleep 120')
        docker('start', container_id)
        docker('exec', container_id, 'sh', '-c',
               'printf server-only > /probe/roundtrip; chmod 644 /probe/roundtrip')
        assert (output / 'roundtrip').read_text() == 'server-only'
        report['file_roundtrip'] = 'PASS'
        report['container_network_namespace'] = docker('exec', container_id, 'readlink', '/proc/self/ns/net')
        report['host_network_namespace'] = os.readlink('/proc/self/ns/net')
        assert report['container_network_namespace'] != report['host_network_namespace']
        report['memory_limit'] = docker('exec', container_id, 'cat', '/sys/fs/cgroup/memory.max')
        assert int(report['memory_limit']) == 512 * 1024 * 1024
        report['cpu_limit'] = docker('exec', container_id, 'cat', '/sys/fs/cgroup/cpu.max')
        quota, period = report['cpu_limit'].split()
        assert quota != 'max' and int(quota) / int(period) == 1
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = repr(error)
    finally:
        if container_id:
            try:
                label = docker('inspect', '--format', '{{index .Config.Labels "comem.acceptance"}}', container_id)
                assert label == token
                docker('rm', '-f', container_id)
                report['container_removed'] = True
            except Exception as error:
                report['cleanup_error'] = repr(error)
                report['status'] = 'FAIL'
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
