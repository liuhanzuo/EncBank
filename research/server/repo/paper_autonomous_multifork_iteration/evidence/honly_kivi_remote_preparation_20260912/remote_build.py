"""One CPU-only sm89 KIVI extension build in the user-authorized remote roots."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,tarfile
ROOT=Path('/srv/encbank').resolve();E=Path(__file__).resolve().parent;E.relative_to(ROOT)
CUDA=ROOT/'qcomem_runtime_20260911/cuda12.8_kivi_codex_20260912';CUDA.resolve().relative_to(ROOT)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
plan=json.loads((E/'build_plan.json').read_text());archive=E/'cuda128_compiler.tar.gz'
assert sha(archive)==plan['compiler_archive_sha256']
assert not CUDA.exists();CUDA.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():(CUDA/member.name).resolve().relative_to(CUDA.resolve())
    tar.extractall(CUDA,filter='data')
for rel,digest in plan['source_sha256'].items():assert sha(E/rel)==digest,rel
O=E/'build_attempt1';assert not O.exists();O.mkdir()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',CUDA_HOME=str(CUDA),TORCH_CUDA_ARCH_LIST='8.9',PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='2',PYTORCH_NVML_BASED_CUDA_CHECK='1')
env['PATH']=str(CUDA/'bin')+os.pathsep+env.get('PATH','')
for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('CUDA_CACHE_PATH','cuda_cache'),('TRITON_CACHE_DIR','triton_cache')]:
    p=O/name;p.resolve().relative_to(ROOT);p.mkdir();env[key]=str(p)
argv=[plan['python'],'-B','setup.py','build_ext','--build-temp',str(O/'temp'),'--build-lib',str(O/'lib')]
with (O/'stdout.log').open('xb') as out,(O/'stderr.log').open('xb') as err:
    child=subprocess.Popen(argv,cwd=E/'source/quant',env=env,stdout=out,stderr=err);code=child.wait()
receipt={'status':'extension_built_pending_GPU_qualification' if code==0 else 'CPU_build_failed_no_scientific_result','at':datetime.datetime.now().astimezone().isoformat(),'actual_held_child_pid':child.pid,'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True,'argv':argv,'CUDA_HOME':str(CUDA),'architecture':'8.9','CUDA_VISIBLE_DEVICES':'-1','GPU_model_or_qualification_executed':False,'stdout_sha256':sha(O/'stdout.log'),'stderr_sha256':sha(O/'stderr.log'),'automatic_retry':False}
if code==0:
    libs=list((O/'lib').glob('*.so'));assert len(libs)==1;receipt['binary']={'path':str(libs[0]),'sha256':sha(libs[0]),'bytes':libs[0].stat().st_size}
save(E/'build_execution_receipt.json',receipt);print(json.dumps(receipt));raise SystemExit(code)
