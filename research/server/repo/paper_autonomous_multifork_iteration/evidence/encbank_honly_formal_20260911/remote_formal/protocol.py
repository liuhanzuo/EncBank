"""Native Slurm envelope for the same frozen scientific runner; no model imports."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ARMS=('dense','h16','h8','h4')
PYTHON='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
REMOTE_ROOT='/srv/encbank/qencbank_align_codex_20260911/repo'
MODEL_ROOT='/srv/encbank/qencbank_align_codex_20260911/exact_local_reader'
RESOURCE={'allocator_cap_bytes':200*2**30,'minimum_free_bytes':220*2**30,'stable_idle_seconds':45}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)
def local(path):
 p=(ROOT/path).resolve();p.relative_to(ROOT);return p
def model_path(plan,part):
 assert part in ('model','adapter')
 p=Path(plan[part+'_root']);assert p==Path(MODEL_ROOT)/part
 return p
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def parser():
 assert __debug__,'Python -O prohibited'
 p=argparse.ArgumentParser()
 for key in ('plan','expected-plan-sha256','arm','output'):p.add_argument('--'+key,required=True)
 p.add_argument('--check-only',action='store_true');return p
def preflight(args,envelope=False):
 assert args.arm in ARMS and sha(args.plan)==args.expected_plan_sha256
 p=read(args.plan);assert p['schema']=='honly_ruler_single_8k_100_remote_v1' and p['resource_policy']==RESOURCE
 assert p['configuration']=={'resume_j':12,'chunk_size':512,'selector':'iter_bm25','topk':12,'iter_hop_topk':4,'iter_rounds':0,'group_size':64,'bos_token_id':151643,'eos_token_id':151645,'max_new_tokens':48}
 for name,digest in p['source_sha256'].items():assert sha(local(name))==digest,name
 for key in ('fixture','activation','cpu_gate','cpu_gate_receipt','model_expected_manifest'):
  assert sha(local(p[key]['path']))==p[key]['sha256'],key
 gate=read(local(p['cpu_gate_receipt']['path']))
 assert gate['actual_exit_code']==0 and gate['actual_parent_wait'] and gate['process_exit_observed']
 fixture=read(local(p['fixture']['path']));assert len(fixture['items'])==100
 assert len({x['id'] for x in fixture['items']})==100
 for x in fixture['items']:
  assert not ({'answer','answers','outputs','gold','reference','references'} & set(x))
  assert x['document_token_ids'] and x['query_token_ids'] and x['bare_question_token_ids']
  assert 1+len(x['document_token_ids'])+len(x['query_token_ids'])+48<=8192
 activation=read(local(p['activation']['path']))
 assert activation['status']=='completed_external_adapter_activated_for_inference_only'
 assert activation['external_status']['complete'] and activation['external_status']['step']==4000
 out=Path(args.output).resolve();assert out==local(p['outputs'][args.arm])
 if envelope:
  rec=read(out/'execution.json');assert rec['plan_sha256']==args.expected_plan_sha256 and rec['arm']==args.arm
  assert not (out/'worker.json').exists() and not (out/'result.json').exists()
 else:assert not out.exists(),'Refuse duplicate output or resume'
 return p,activation
