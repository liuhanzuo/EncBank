"""Remote cc10.3 tiny Half-kernel qualification only; no model imports."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
REMOTE_ROOT='/srv/encbank/qencbank_align_codex_20260911/repo'
PYTHON='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
ARMS=('kivi_half_synthetic',)
RESOURCE={'allocator_cap_bytes':200*2**30,'minimum_free_bytes':220*2**30,'stable_idle_seconds':45}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)
def local(path):
    p=(ROOT/path).resolve();p.relative_to(ROOT);return p
def module(path,name):
    path=Path(path).resolve();spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);assert Path(m.__file__).resolve()==path;return m
def parser():
    assert __debug__
    p=argparse.ArgumentParser()
    for key in ('plan','expected-plan-sha256','arm','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true');return p
def preflight(args,envelope=False):
    assert args.arm in ARMS and sha(args.plan)==args.expected_plan_sha256
    p=read(args.plan);assert p['schema']=='kivi_half_cc103_sm100ptx_remote_tiny_qualification_v1' and p['resource_policy']==RESOURCE
    assert p['model_load_allowed'] is p['training_allowed'] is p['runtime_offload_allowed'] is p['automatic_retry'] is False
    for name,digest in p['source_sha256'].items():assert sha(local(name))==digest,name
    for key in ('build_receipt','build_plan','reused_local_qualification_plan','reused_local_completion'):
        assert sha(local(p[key]['path']))==p[key]['sha256'],key
    built=read(local(p['build_receipt']['path']))
    assert built['status']=='extension_built_pending_actual_target_GPU_qualification' and built['actual_exit_code']==0 and built['actual_parent_wait'] and built['process_exit_observed']
    assert built['architecture']=='10.0+PTX' and built['GPU_model_or_qualification_executed'] is False
    assert built['binary']['sha256']==p['binary']['sha256']
    assert built['binary']['path']==REMOTE_ROOT+'/'+p['binary']['path']
    assert p['required_compute_capability']==[10,3] and p['compiled_architecture']=='10.0+PTX'
    assert sha(local(p['architecture_metadata']['path']))==p['architecture_metadata']['sha256']
    prior=read(local(p['reused_local_qualification_plan']['path']))
    for field in ('seed','cases','tail_cases','max_abs_tolerance','rms_tolerance','expected_closed_phases','versions'):
        assert p[field]==prior[field],field
    assert p['expected_closed_phases']==7 and len(p['cases'])==4 and len(p['tail_cases'])==2
    if ROOT==Path(REMOTE_ROOT):assert sha(local(p['binary']['path']))==p['binary']['sha256']
    else:assert args.check_only,'Local preparation may only inspect bindings; remote binary not downloaded or executed'
    out=Path(args.output).resolve();assert out==local(p['outputs'][args.arm])
    if envelope:
        rec=read(out/'execution.json');assert rec['plan_sha256']==args.expected_plan_sha256 and rec['arm']==args.arm
        assert not (out/'worker.json').exists() and not (out/'result.json').exists()
    else:assert not out.exists(),'No duplicate or resume'
    return p,None
