"""Read a Python stack from this campaign's worker; no shared environment edits."""
import hashlib, json, pathlib, subprocess, urllib.request, zipfile, io

root = pathlib.Path(__file__).resolve().parent
out = root / 'startup_diagnostic_109631'
out.mkdir(exist_ok=True)
meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/py-spy/0.4.2/json', timeout=20))
artifact = next(x for x in meta['urls'] if 'manylinux' in x['filename'] and 'x86_64' in x['filename'])
wheel = urllib.request.urlopen(artifact['url'], timeout=30).read()
assert hashlib.sha256(wheel).hexdigest() == artifact['digests']['sha256']
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    name = next(n for n in archive.namelist() if n.endswith('/scripts/py-spy'))
    (out / 'py-spy').write_bytes(archive.read(name))
(out / 'profiler_source.json').write_text(json.dumps({k: artifact[k] for k in ('filename','url','digests')}, indent=2))
remote = '/tmp/qcm-q38-tb-liuhanzuo-109631'
subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', str(out / 'py-spy'), 'gpu-node1:' + remote + '/py-spy'], check=True, timeout=30)
command = 'chmod 700 {r}/py-spy && timeout 15s {r}/py-spy dump --pid 745085 --nonblocking'.format(r=remote)
proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',command], capture_output=True,text=True,timeout=25)
(out / 'python_stack.txt').write_text(proc.stdout + '\nSTDERR:\n' + proc.stderr, encoding='utf8')
print(json.dumps(dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)))
