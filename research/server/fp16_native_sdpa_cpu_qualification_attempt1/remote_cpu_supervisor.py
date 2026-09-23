"""Exact new-source CPU-only test, held child and bounded task-owned caches."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED=Path('/srv/encbank/qencbank_align_codex_20260911/fp16_native_sdpa_cpu_qualification_attempt1')
assert ROOT==EXPECTED and ROOT.is_relative_to('/srv/encbank')
assert sys.executable=='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
OUT=ROOT/'run';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for n,h in manifest['source_sha256'].items():
    p=(ROOT/n).resolve();p.relative_to(ROOT);assert sha(p)==h,n
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='20260912',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for key,sub in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_HOME':'cache/torch','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda'}.items():
    p=ROOT/sub;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
argv=[sys.executable,'-X','faulthandler','-B',str(ROOT/'check_native_sdpa_cpu.py'),'--output',str(OUT/'report.json')]
started=datetime.datetime.now().astimezone().isoformat()
with (OUT/'stdout.txt').open('wb') as stdout,(OUT/'stderr.txt').open('wb') as stderr:
    child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    timeout=False
    try:code=child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timeout=True;child.terminate()
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();code=child.wait()
record={'argv':argv,'pid':child.pid,'started_at':started,'finished_at':datetime.datetime.now().astimezone().isoformat(),'actual_parent_wait':True,'process_exit_observed':True,'actual_exit_code':code,'timed_out':timeout,'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'sources_unchanged':all(sha(ROOT/n)==h for n,h in manifest['source_sha256'].items()),'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'task_cache_environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH')},'scope':'CPU tiny random only, no real checkpoint/GPU/Slurm/training/installation'}
(OUT/'execution_receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record),flush=True)
sys.exit(0 if code==0 else 1)
