"""Compile the failed FlashInfer sampling module on CPU; never load a model."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
H=Path(__file__).resolve().parent;H.resolve().relative_to(Path('/srv/encbank').resolve())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.update(PATH='/srv/encbank/venvs/rllm/bin:/usr/local/cuda/bin:'+os.environ['PATH'],CUDA_HOME='/usr/local/cuda',CPATH='/srv/encbank/.cache/python/include/python3.12',FLASHINFER_CUDA_ARCH_LIST='10.3',FLASHINFER_WORKSPACE_BASE=str(H),TVM_FFI_CACHE_DIR=str(H/'tvm_ffi'),MAX_JOBS='2',TMPDIR='/srv/encbank/qencbank_runtime_20260911/t89i4',VLLM_RPC_BASE_PATH='/srv/encbank/qencbank_runtime_20260911/t89i4',XDG_CACHE_HOME=str(H/'cache'),HF_HOME=str(H/'hf_cache'),PYTHONDONTWRITEBYTECODE='1')
def save(n,d):(H/n).write_text(json.dumps(d,indent=2)+'\n')
started=time.time();versions={}
try:
 for name in ['ninja','nvcc','g++','gcc']:
  path=shutil.which(name);assert path,name
  p=subprocess.run([path,'--version'],capture_output=True,text=True,timeout=15);assert p.returncode==0
  versions[name]=dict(path=path,version=p.stdout,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
 save('toolchain_versions.json',dict(versions=versions,env={k:os.environ[k] for k in ['PATH','CUDA_HOME','CPATH','FLASHINFER_CUDA_ARCH_LIST','FLASHINFER_WORKSPACE_BASE','TVM_FFI_CACHE_DIR','MAX_JOBS','TMPDIR','CUDA_VISIBLE_DEVICES']}))
 import torch
 assert not torch.cuda.is_initialized()
 from flashinfer.jit import gen_sampling_module
 spec=gen_sampling_module();spec.build(verbose=True)
 assert not torch.cuda.is_initialized()
 paths=[p for p in spec.build_dir.rglob('*') if p.is_file()]
 manifest={p.relative_to(H).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 save('toolchain_compile.json',dict(status='PASS',seconds=time.time()-started,gpu_model_calls=0,cuda_context_initialized=False,compiled_module_loaded=False,arch='10.3a (same as observed worker)',build_directory=str(spec.build_dir),manifest=manifest,versions=versions))
 print('PASS CPU compilation of actual FlashInfer sampling module',flush=True)
except BaseException:
 save('toolchain_compile.json',dict(status='FAIL',seconds=time.time()-started,error=traceback.format_exc(),gpu_model_calls=0));raise
