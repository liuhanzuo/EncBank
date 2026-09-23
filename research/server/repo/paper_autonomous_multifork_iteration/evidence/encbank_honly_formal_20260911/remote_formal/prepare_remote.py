"""Freeze a sparse Slurm package; no SSH, GPU allocation or sbatch submission."""
import argparse,ast,datetime,difflib,hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
from protocol import HERE,ROOT,REMOTE_ROOT,MODEL_ROOT,PYTHON,RESOURCE,ARMS,read,save,sha
def bind(path):return {'path':Path(path).relative_to(ROOT).as_posix(),'sha256':sha(path)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--local-plan',required=True);p.add_argument('--expected-local-plan-sha256',required=True);a=p.parse_args()
 assert not (HERE/'plan.json').exists(),'Already frozen; no overwrite'
 original=Path(a.local_plan).resolve();assert sha(original)==a.expected_local_plan_sha256
 plan=read(original);assert plan['schema']=='honly_ruler_single_8k_100_v1'
 assert 'only' in plan['dispatch_scope'] or 'intermediate' in plan['dispatch_scope']
 assert plan['configuration']['max_new_tokens']==48
 for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
 gate=read(ROOT/plan['cpu_gate']['path']);receipt=read(ROOT/plan['cpu_gate_actual_exit']['path'])
 assert gate['status']=='PASS_focused_tiny_CPU_runtime_qualification' and not gate['blockers']
 assert receipt['actual_exit_code']==0 and receipt['actual_parent_wait'] and receipt['process_exit_observed']
 for key in ('cpu_gate','cpu_gate_actual_exit','fixture','activation','labels','original_fixture','scorer','postprocess'):assert sha(ROOT/plan[key]['path'])==plan[key]['sha256']
 # Byte-identical scientific runner; analyzer only substitutes truthful native Linux exit evidence.
 common=original.parent;runner=HERE/'run_quality.py';assert not runner.exists();runner.write_bytes((common/'run_quality.py').read_bytes())
 old=(common/'analyze.py').read_text(encoding='utf-8');needle="execution['wsl_exit_code'] == 0"
 assert old.count(needle)==1
 new=old.replace(needle,"execution['worker_exit_code'] == 0 and execution['actual_parent_wait']")
 target=HERE/'analyze.py';assert not target.exists();target.write_text(new,encoding='utf-8',newline='\n')
 (HERE/'analyzer_native_exit_delta.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='formal/analyze.py',tofile='remote_formal/analyze.py')),encoding='utf-8')
 activation=read(ROOT/plan['activation']['path'])
 expected={'model':activation['model_file_sha256'],'adapter':{k:v['sha256'] for k,v in activation['files'].items()},'activation':plan['activation'],'remote_destination':MODEL_ROOT,'staging_status_is_dynamic_not_frozen':True}
 save(HERE/'expected_model_manifest.json',expected)
 scientific=[ROOT/name for name in plan['source_sha256'] if '/runtime/' in name]
 scientific += [ROOT/'tmp_external_baselines/encbank_official/eval/_common.py',ROOT/'gpu/kvquant_quality_balanced/phase_profile.py',ROOT/plan['scorer']['path'],ROOT/plan['postprocess']['path']]
 scripts=[HERE/n for n in ('protocol.py','resource_guard.py','resource_worker.py','run_batch.py','record_shell_exit.py','prepare_remote.py','run_quality.py','analyze.py')]
 for path in scientific+scripts:
  if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
 batch='results/encbank-honly-ruler-single2-8k-100-gpu-node1-codex-20260911/attempt1'
 plan.update(schema='honly_ruler_single_8k_100_remote_v1',frozen_at=datetime.datetime.now().astimezone().isoformat(),resource_policy=RESOURCE,model_root=MODEL_ROOT+'/model',adapter_root=MODEL_ROOT+'/adapter',cpu_gate_receipt=plan['cpu_gate_actual_exit'],model_expected_manifest=bind(HERE/'expected_model_manifest.json'),local_intermediate_plan=bind(original),batch_output=batch,outputs={arm:batch+'/'+arm for arm in ARMS},dispatch_scope='Remote Slurm single GPU only, all four arms in one allocation. Submit only after source/input staging and exact model/adapter remote SHA verification complete. Never invoke local launcher.',remote_python=PYTHON,remote_repo_root=REMOTE_ROOT,artifact_staging_at_preparation='pending; not scientific readiness or GPU allocation',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in scientific+scripts},remote_exit_contract='native worker actual wait for each arm; shell trap is not final Slurm controller accounting; final independent completion must check sacct job exit0',device_block='Observed driver-reported identity only; no confirmed B300 claim and no insertion into RTX5090 timing rows')
 save(HERE/'plan.json',plan)
 rel=HERE.relative_to(ROOT).as_posix();remotehere=REMOTE_ROOT+'/'+rel
 script='''#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=qencbank-align-codex
#SBATCH --dependency=singleton
#SBATCH --output=REMOTEHERE/slurm-%j.out
#SBATCH --error=REMOTEHERE/slurm-%j.err
set -euo pipefail
export QENCBANK_REPO_ROOT=REMOTEROOT
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QENCBANK_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  PYTHON -B REMOTEHERE/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
PYTHON -u -B REMOTEHERE/run_batch.py --plan REMOTEHERE/plan.json --expected-plan-sha256 PLANHASH
'''.replace('REMOTEHERE',remotehere).replace('REMOTEROOT',REMOTE_ROOT).replace('PYTHON -',PYTHON+' -').replace('PLANHASH',sha(HERE/'plan.json'))
 (HERE/'batch.sbatch').write_text(script,encoding='utf-8',newline='\n')
 upload=set(scientific+scripts+[HERE/'plan.json',HERE/'batch.sbatch',HERE/'expected_model_manifest.json',HERE/'analyzer_native_exit_delta.patch',original])
 upload.update(ROOT/plan[k]['path'] for k in ('fixture','original_fixture','labels','activation','cpu_gate','cpu_gate_receipt','scorer','postprocess'))
 files=[{'local_path':str(x),'repo_relative_path':x.relative_to(ROOT).as_posix(),'remote_path':REMOTE_ROOT+'/'+x.relative_to(ROOT).as_posix(),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(upload)]
 save(HERE/'upload_manifest.json',{'remote_root':REMOTE_ROOT,'files':files,'model_files_excluded':'already owned separate exact-reader staging; validate against expected_model_manifest before sbatch','labels_scope':'only CPU analyze.py reads labels; resource worker and runner do not load labels'})
 argv=['sbatch','--parsable',remotehere+'/batch.sbatch']
 save(HERE/'launch_commands.json',{'submit_argv':argv,'remote_cwd':REMOTE_ROOT,'plan_sha256':sha(HERE/'plan.json'),'script_sha256':sha(HERE/'batch.sbatch'),'automatic_retry':False,'prerequisites':['remote CPU actual exit0 already bound','separate model staging actual complete and each remote model/adapter SHA equals expected_model_manifest','every upload_manifest file remote SHA exact','fresh output directories absent and no duplicate same-namespace job submitted'],'not_submitted':True})
 checks=[]
 for label,okay in [('same_scientific_runner',sha(runner)==sha(common/'run_quality.py')),('same_configuration',plan['configuration']==read(original)['configuration']),('one_allocated_GPU','#SBATCH --gres=gpu:1\n' in script),('singleton','#SBATCH --dependency=singleton\n' in script),('no_local25G_resource',plan['resource_policy']==RESOURCE),('no_gold_worker','labels' not in (HERE/'resource_worker.py').read_text()),('all_four_outputs_fresh',all(not (ROOT/v).exists() for v in plan['outputs'].values()))]:
  assert okay,label;checks.append({'name':label,'passed':True})
 save(HERE/'preparation_receipt.json',{'status':'PREPARED_NOT_SUBMITTED_ARTIFACT_STAGING_PENDING','plan_sha256':sha(HERE/'plan.json'),'upload_manifest_sha256':sha(HERE/'upload_manifest.json'),'launch_commands_sha256':sha(HERE/'launch_commands.json'),'checks':checks,'python_AST_checks_passed':len(scientific+scripts),'shell_syntax_check':'pending native bash -n only, no execution','model_staging_not_assumed_complete':True,'GPU_or_scheduler_actions':0,'known_limit':'remote CUDA qualification is formal first execution; tiny CPU only passed. Final sacct identity/exit must be inspected after job stops.'})
 print(json.dumps({'status':'PREPARED_NOT_SUBMITTED','plan_sha256':sha(HERE/'plan.json'),'upload_files':len(files),'launch_commands':str(HERE/'launch_commands.json'),'upload_manifest':str(HERE/'upload_manifest.json')}))
if __name__=='__main__':main()
