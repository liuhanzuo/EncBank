"""Model-free acceptance of the two supplement images on this computer's Docker."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import time
import uuid

ROOT=Path('/srv/encbank/workspace/local_docker_20260921')


def command(args, timeout=60):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout.strip()


def probe(task):
    tag=uuid.uuid4().hex[:12]
    output=ROOT/('check-'+task+'-'+tag)
    output.mkdir()
    name='comem-local-check-'+tag
    cid=None
    report=dict(task=task,model_calls=0,benchmark_attempts=0,started_epoch=time.time(),status='RUNNING')
    gateway=command(['ip','route','show','default']).split()[2]
    proxy='http://'+gateway+':7897'
    def docker(*args,timeout=60):return command(['docker',*args],timeout)
    try:
        image='alexgshaw/'+task+':20251031'
        docker('pull',image,timeout=600)
        report['image']=image
        report['image_digest']=json.loads(docker('image','inspect',image))[0]['RepoDigests']
        cid=docker('create','--name',name,'--label','comem.acceptance='+tag,
                   '--memory','2g','--cpus','1','--pids-limit','512',
                   '-e','http_proxy='+proxy,'-e','https_proxy='+proxy,
                   '-e','HTTP_PROXY='+proxy,'-e','HTTPS_PROXY='+proxy,
                   image,'bash','-c','sleep 1200')
        docker('start',cid)
        report['memory_max']=docker('exec',cid,'cat','/sys/fs/cgroup/memory.max')
        report['cpu_max']=docker('exec',cid,'cat','/sys/fs/cgroup/cpu.max')
        assert int(report['memory_max'])==2*1024**3
        quota,period=report['cpu_max'].split()
        assert int(quota)==int(period)
        install='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux asciinema; command -v tmux; command -v asciinema'
        (output/'install.log').write_text(docker('exec',cid,'bash','-lc',install,timeout=600)+'\n')
        docker('exec',cid,'bash','-lc',"tmux -L comem-check new-session -d -s check 'sleep 120'")
        docker('exec',cid,'bash','-lc','tmux -L comem-check has-session -t check; tmux -L comem-check kill-server')
        report['terminal_and_persistent_tmux']='PASS'
        (output/'upload.txt').write_text('comem-local-docker-roundtrip\n')
        docker('cp',str(output/'upload.txt'),cid+':/tmp/comem-upload.txt')
        docker('cp',cid+':/tmp/comem-upload.txt',str(output/'download.txt'))
        assert (output/'upload.txt').read_bytes()==(output/'download.txt').read_bytes()
        report['file_transfer']='PASS'
        report['network_namespace']=docker('exec',cid,'readlink','/proc/self/ns/net')
        assert report['network_namespace']!=os.readlink('/proc/self/ns/net')
        report['https_status']=docker('exec',cid,'python3','-c','import urllib.request; print(urllib.request.urlopen("https://example.com",timeout=30).status)')
        assert report['https_status']=='200'
        report['status']='PASS'
    except Exception as error:
        report.update(status='FAIL',error=repr(error))
    finally:
        if cid:
            try:
                label=docker('inspect','--format','{{index .Config.Labels "comem.acceptance"}}',cid)
                assert label==tag
                docker('rm','-f',cid)
                report['owned_container_removed']=True
            except Exception as error:report.update(status='FAIL',cleanup_error=repr(error))
        report['ended_epoch']=time.time()
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    assert os.name=='posix'
    report=dict(server=json.loads(command(['docker','version','--format','{{json .Server}}'])),
                compose=command(['docker','compose','version']),model_calls=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        report['images']=list(pool.map(probe,['regex-chess','vulnerable-secret']))
    report['status']='PASS' if all(row['status']=='PASS' for row in report['images']) else 'FAIL'
    (ROOT/'ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)
