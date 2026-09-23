"""One CPU-only original-source sm100 + compute100 PTX build; isolated compiler reuse."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback
ROOT=Path('/srv/encbank').resolve();E=Path(__file__).resolve().parent;E.relative_to(ROOT)
CUDA=ROOT/'qcomem_runtime_20260911/cuda12.8_kivi_codex_20260912';CUDA.resolve().relative_to(ROOT)
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
assert len(sys.argv)==2 and sha(E/'build_plan.json')==sys.argv[1]
plan=json.loads((E/'build_plan.json').read_text())
assert plan['architecture']=='10.0+PTX' and plan['compiler_reused_without_extraction'] is True
assert sha(Path(__file__))==plan['remote_build_sha256'] and CUDA.is_dir()
for rel,digest in plan['source_sha256'].items():assert sha(E/rel)==digest,rel
O=E/'build_attempt1';assert not O.exists();O.mkdir()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',CUDA_HOME=str(CUDA),TORCH_CUDA_ARCH_LIST='10.0+PTX',PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='2',PYTORCH_NVML_BASED_CUDA_CHECK='1')
env['PATH']=str(CUDA/'bin')+os.pathsep+env.get('PATH','')
for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('CUDA_CACHE_PATH','cuda_cache'),('TRITON_CACHE_DIR','triton_cache'),('HF_HOME','hf'),('TORCH_HOME','torch')]:
 p=O/name;p.resolve().relative_to(ROOT);p.mkdir();env[key]=str(p)
compiler_names=['bin/nvcc','bin/nvcc.profile','bin/ptxas','bin/fatbinary','bin/nvlink','bin/bin2c','bin/cudafe++','bin/cuobjdump','nvvm/bin/cicc','targets/x86_64-linux/lib/libcudart.so.12.8.90']
compiler_hashes={name:sha(CUDA/name) for name in compiler_names}
receipt={'status':'CPU_build_starting_no_GPU','started_at':now(),'plan_sha256':sys.argv[1],'remote_build_sha256':sha(Path(__file__)),'architecture':'10.0+PTX','CUDA_VISIBLE_DEVICES':'-1','MAX_JOBS':'2','CUDA_HOME':str(CUDA),'compiler_reused_without_extraction':True,'compiler_component_sha256':compiler_hashes,'source_sha256':plan['source_sha256'],'GPU_model_or_qualification_executed':False,'automatic_retry':False,'commands':[]}
save(E/'build_execution_receipt.json',receipt)
def capture(argv,name):
 with (O/(name+'.stdout')).open('xb') as out,(O/(name+'.stderr')).open('xb') as err:
  child=subprocess.Popen(argv,env=env,stdout=out,stderr=err);code=child.wait()
 result={'argv':argv,'pid':child.pid,'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True,'stdout_sha256':sha(O/(name+'.stdout')),'stderr_sha256':sha(O/(name+'.stderr'))}
 receipt['commands'].append(result);save(E/'build_execution_receipt.json',receipt);assert code==0,result
 return (O/(name+'.stdout')).read_text()
try:
 version=capture([str(CUDA/'bin/nvcc'),'--version'],'nvcc_version');assert 'V12.8.93' in version
 arch=capture([str(CUDA/'bin/nvcc'),'--list-gpu-arch'],'nvcc_arch');assert 'compute_100' in arch.splitlines()
 codes=capture([str(CUDA/'bin/nvcc'),'--list-gpu-code'],'nvcc_code');assert 'sm_100' in codes.splitlines()
 receipt['compiler_version']=version
 argv=[plan['python'],'-B','setup.py','build_ext','--build-temp',str(O/'temp'),'--build-lib',str(O/'lib')]
 with (O/'stdout.log').open('xb') as out,(O/'stderr.log').open('xb') as err:
  child=subprocess.Popen(argv,cwd=E/'source/quant',env=env,stdout=out,stderr=err)
  receipt.update(status='CPU_compile_running',actual_held_child_pid=child.pid,argv=argv);save(E/'build_execution_receipt.json',receipt);code=child.wait()
 receipt.update(actual_exit_code=code,actual_parent_wait=True,process_exit_observed=True,stdout_sha256=sha(O/'stdout.log'),stderr_sha256=sha(O/'stderr.log'))
 assert code==0,('actual CPU compile failed',code)
 for rel,digest in plan['source_sha256'].items():assert sha(E/rel)==digest,rel
 assert compiler_hashes=={name:sha(CUDA/name) for name in compiler_names}
 libs=list((O/'lib').glob('*.so'));assert len(libs)==1
 receipt['binary']={'path':str(libs[0]),'sha256':sha(libs[0]),'bytes':libs[0].stat().st_size}
 compile_log=(O/'stdout.log').read_text()
 assert '-gencode=arch=compute_100,code=compute_100' in compile_log and '-gencode=arch=compute_100,code=sm_100' in compile_log
 assert 'compute_100a' not in compile_log and 'sm_100a' not in compile_log
 receipt['architecture_flags_observed']=['-gencode=arch=compute_100,code=compute_100','-gencode=arch=compute_100,code=sm_100']
 receipt['status']='extension_built_pending_actual_target_GPU_qualification'
except BaseException as error:
 receipt.update(status='CPU_build_or_validation_failure_not_scientific_result',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
finally:
 receipt['finished_at']=now();save(E/'build_execution_receipt.json',receipt)
print(json.dumps(receipt))
