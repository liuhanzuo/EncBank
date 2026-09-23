"""Temporary CPU receipts only; never modifies real results or loads a model."""
import csv,copy,json,tempfile,sys,py_compile
from pathlib import Path
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';sys.path.insert(0,str(B))
from native_priority_receipt import completed
from babi32_priority_bootstrap import verify_scope,launch_allowed
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x))
p=read(B/'babi32_priority_full_plan.json');main=read(B/'full_plan.json');state=read(R/'outputs/queue_v2/full/state.json');verify_scope(p,main)
lookup={j['id']:j for j in main['jobs']};original=lookup['babilong__pub__qa2_0k'];assert completed(original)
assert all(completed(lookup[dep]) for dep in p['jobs'][0]['depends_on']);assert len(p['jobs'][0]['depends_on'])==56
base=Path(original['output']);marker=read(base/'COMPLETED.json');config=read(base/'run_config.json')
with Path(marker['files'][0]['file']).open(newline='') as f:source_rows=list(csv.DictReader(f))
cases=0
with tempfile.TemporaryDirectory(prefix='babi_receipt_0429_',dir=R/'outputs/diagnostics') as temp:
  root=Path(temp)
  def fixture(name):
    out=root/name;native=out/'attempts/0001/native';native.mkdir(parents=True);job=copy.deepcopy(original);job['output']=str(out)
    job['argv'][job['argv'].index('--out')+1]=str(out)
    m=copy.deepcopy(marker);m['output_dir']=str(native);m['driver_argv'][-1]=str(native);m['files'][0]['file']=str(native/'qa2_0k_official_explicit.csv')
    write(out/'COMPLETED.json',m);write(out/'run_config.json',config)
    with Path(m['files'][0]['file']).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=source_rows[0]);w.writeheader();w.writerows(source_rows)
    return job,out,m
  good,out,m=fixture('valid');assert completed(good);cases+=1
  j,out,m=fixture('no_marker');(out/'COMPLETED.json').unlink();assert not completed(j);cases+=1
  def rejected(name,change):
    global cases
    j,out,m=fixture(name);change(j,out,m)
    try:completed(j)
    except (AssertionError,ValueError,FileNotFoundError,KeyError):cases+=1
    else:raise AssertionError('invalid receipt accepted: '+name)
  rejected('receipt_count',lambda j,o,m:(m['files'][0].update(records=99),write(o/'COMPLETED.json',m)))
  rejected('generation_count',lambda j,o,m:(m.update(new_generations=99),write(o/'COMPLETED.json',m)))
  rejected('wrong_topk',lambda j,o,m:write(o/'run_config.json',{**config,'driver_options':{**config['driver_options'],'topk':12}}))
  rejected('wrong_argv',lambda j,o,m:(m['driver_argv'].__setitem__(m['driver_argv'].index('--limit')+1,'99'),write(o/'COMPLETED.json',m)))
  rejected('outside_file',lambda j,o,m:(m['files'][0].update(file=str(root/'outside.csv')),write(o/'COMPLETED.json',m)))
  rejected('wrong_reader',lambda j,o,m:(m['reader'].update(effective_j=36),write(o/'COMPLETED.json',m)))
  def change_rows(j,o,m,kind):
    rr=copy.deepcopy(source_rows)
    if kind=='missing':rr=rr[:-1]
    elif kind=='duplicate':rr[1]['index']='0'
    elif kind=='id':rr[1]['id']='qa3/0k/1'
    elif kind=='oom':rr[1]['output']='[OOM]'
    with Path(m['files'][0]['file']).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
  for kind in ('missing','duplicate','id','oom'):rejected(kind,lambda j,o,m,k=kind:change_rows(j,o,m,k))
  j,out,m=fixture('resumed');m.update(new_generations=60,reused_generations=40);write(out/'COMPLETED.json',m);assert completed(j);cases+=1
job=p['jobs'][0];gate=launch_allowed(job,main,state,completed);assert gate[0] and gate[2]>=40
ss=copy.deepcopy(state);ss['jobs'][job['id']]['status']='running';assert not launch_allowed(job,main,ss,completed)[0]
assert launch_allowed(job,main,state,lambda j:True)[:2]==(True,'canonical_complete_reuse_before_model_load')
for j in main['jobs']:
  if j['id']!=job['id']:ss['jobs'][j['id']]['status']='completed'
ss['jobs'][job['id']]['status']='pending';assert not launch_allowed(job,main,ss,lambda j:False)[0]
wrong=copy.deepcopy(p);wrong['jobs'][0]['argv']+=['--limit','99']
try:verify_scope(wrong,main)
except RuntimeError:pass
else:raise AssertionError('changed canonical job accepted')
wrong=copy.deepcopy(p);wrong['jobs']=wrong['jobs'][:-1]
try:verify_scope(wrong,main)
except RuntimeError:pass
else:raise AssertionError('partial scope accepted')
for path in (B/'native_priority_receipt.py',B/'babi32_priority_bootstrap.py'):py_compile.compile(str(path),doraise=True)
result={'passed':True,'receipt_cases':cases,'frontier_scope_cases':6,'real_complete_dependency_receipts':56,'current_first_job_gate':gate,'gpu_used':False}
write(R/'outputs/diagnostics/heartbeat_native_receipt_20260909_0429_tests.json',result);print(json.dumps(result))
