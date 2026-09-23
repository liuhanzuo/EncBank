"""Compile the exact Triton helper on CPU, with the restored include path."""
from pathlib import Path
import hashlib,json,os,subprocess,sysconfig,time
H=Path(__file__).resolve().parent;H.resolve().relative_to(Path('/srv/encbank').resolve())
D=H/'compile_check';D.mkdir(exist_ok=False)
T=Path('/srv/encbank/venvs/rllm/lib/python3.12/site-packages/triton/backends/nvidia')
INCLUDE=Path('/srv/encbank/.cache/python/include/python3.12').resolve();INCLUDE.relative_to(Path('/srv/encbank').resolve())
source=(T/'driver.c').read_bytes();(D/'cuda_utils.c').write_bytes(source)
env=os.environ.copy();env['CPATH']=str(INCLUDE);env['TMPDIR']=str(H/'tmp')
argv=['/usr/bin/gcc',str(D/'cuda_utils.c'),'-O3','-shared','-fPIC','-Wno-psabi','-o',str(D/'cuda_utils.cpython-312-x86_64-linux-gnu.so'),'-l:libcuda.so.1','-L'+str(T/'lib'),'-L/lib64','-L/lib','-I'+str(T/'include'),'-I'+str(D),'-I/usr/include/python3.12']
start=time.monotonic();p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
result=dict(status='PASS' if p.returncode==0 else 'FAIL',argv=argv,CPATH=str(INCLUDE),exit_code=p.returncode,actual_parent_wait=True,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-start,
    source_sha256=hashlib.sha256(source).hexdigest(),Python_h_sha256=hashlib.sha256((INCLUDE/'Python.h').read_bytes()).hexdigest(),patchlevel=(INCLUDE/'patchlevel.h').read_text(),gpu_model_calls=0,compiled_binary_loaded=False)
if p.returncode==0:result['binary_sha256']=hashlib.sha256((D/'cuda_utils.cpython-312-x86_64-linux-gnu.so').read_bytes()).hexdigest()
(H/'header_compile_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='patchlevel'}));assert p.returncode==0
