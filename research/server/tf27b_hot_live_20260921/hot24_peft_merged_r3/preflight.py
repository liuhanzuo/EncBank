"""Pre-submission imports, exact task/source identity, and PEFT input proofs."""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
started=time.time()
p=json.loads((ROOT/'plan.json').read_text())
assert p['lora_execution']=='peft_merged_bf16'
assert p['decode_batch_size']==p['task_concurrency']==24 and p['hot_chunks']==24 and len(p['tasks'])==32
for path,expected in json.loads((ROOT/'task_manifest.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
for f in ROOT.glob('*.py'):ast.parse(f.read_text())
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',LITELLM_LOCAL_MODEL_COST_MAP='True',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
c=subprocess.run([p['harbor_python'],'-B','-c','import controller,tb_agent,apptainer_environment'],env=env,cwd=ROOT,capture_output=True,text=True)
assert c.returncode==0,c.stderr
from native_common import MODELS
from runtime_identity import resolve_configs
from encbank_peft import file_sha
from merged_qualification import qualify
from peft import __version__ as peft_version
resolve_configs(p,MODELS[1])
assert file_sha(p['adapter_path'])==p['adapter_sha256']
directory=Path(p['peft_adapter']);receipt=json.loads((directory/'conversion_receipt.json').read_text())
assert receipt['module_count']==333 and receipt['tensor_count']==666 and receipt['roundtrip_exact']
for name,d in receipt['files'].items():assert file_sha(directory/name)==d,name
q=Path(p['qualification_root'])
assert json.loads((q/'qualification_complete.json').read_text())['passed']
assert json.loads((q/'parent_exit.json').read_text())['exit_code']==0
for n in ['hybrid_hot.py','hybrid_reader.py','batch_cache.py']:assert file_sha(ROOT/n)==file_sha(q/n),n
result=dict(passed=True,epoch=time.time(),seconds=time.time()-started,model_calls=0,
    peft_version=peft_version,peft_adapter_files=receipt['files'],tasks=32,task_concurrency=24,decode_batch_size=24,hot_chunks=24,
    task_files_sha_verified=True,harbor_import_exit=c.returncode,unchanged_core_qualification=str(q),
    original_model_setup_equivalence_claimed=False,user_authorized_new_weight_execution=True,
    required_gpu_startup_qualification='merged_qualification.qualify before worker ready and any benchmark task',
    production_job_changes=0)
with (ROOT/'peft_preflight.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
