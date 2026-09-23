"""Specialize working cache-fixed packages only by official cell and fresh namespaces."""
import copy, difflib, shlex
from pathlib import PurePosixPath
from prepare_remaining import H,ROOT,B,BASE,CELLS,read,sha,bind,rel,save,write,now
REMOTE='/srv/encbank/qencbank_align_codex_20260911/repo'
MODULE='273eb77a08c2f6b9b5dad9436e4b4bcc884514b60983a8d065e6ac5327447715'
def main():
 assert read(H/'input_generation_complete.json')['status']=='ALL_17_ACTUAL_HELD_CPU_INPUT_GENERATION_EXITS_ZERO'
 old=read(BASE/'package/plan.json');old_manifest=read(BASE/'package/upload_manifest.json')
 assert sha(BASE/'package/plan.json')=='c040c344bd6c1784b04e536c2742c8c826f6a33d87559aaef0148147545b18fd'
 for r,d in old['source_sha256'].items():assert sha(ROOT/r)==d,r
 assert sha(BASE/'package/qencbank_triton_cache.py')==MODULE
 for task,length in CELLS:
  c=H/'cells'/f'{task}_{length}';p=c/'package';p.mkdir(exist_ok=False)
  inp=read(c/'inputs/manifest.json');prep=read(c/'preparation_plan.json');N=inp['items'];D=inp['unique_documents'];phases=4*D+2*N+1
  assert N==100 and inp['expected_phases_per_arm']==phases
  generation=read(c/'generation_attempt1/execution_receipt.json')
  assert generation['actual_wsl_exit_code']==generation['linux_actual_exit_code']==0 and generation['actual_parent_wait']
  newjob=f'qencbank-align-codex-babi-{task}-{length}-r17'
  cache=f'/srv/encbank/qencbank_align_codex_20260911/task_cache/babilong_{task}_{length}_remaining17_attempt1'
  control=f'/srv/encbank/qencbank_align_codex_20260911/babilong_{task}_{length}_remaining17_dispatch_attempt1'
  def namespace(text):
   return text.replace(rel(BASE/'package'),rel(p)).replace(old['scheduler']['job_name'],newjob).replace(old['task_cache_root'],cache).replace('/srv/encbank/qencbank_align_codex_20260911/babilong_qa1_4k_dispatch_cache_fixed_attempt1',control).replace('qa1_4k',f'{task}_{length}').replace('qa1/4k',f'{task}/{length}')
  sources={r:d for r,d in old['source_sha256'].items() if not r.startswith(rel(BASE/'package')+'/')};deltas=[]
  for src in sorted((BASE/'package').glob('*.py')):
   before=src.read_text('utf-8');after=namespace(before)
   write(p/src.name,after);sources[rel(p/src.name)]=sha(p/src.name)
   deltas.append({'base':bind(src),'new':bind(p/src.name),'byte_identical':src.read_bytes()==(p/src.name).read_bytes()})
   if before!=after:write(c/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(p/src.name))))
  plan=copy.deepcopy(old)
  plan.update(schema=f'BABILong_{task}_{length}_100_FP16_six_method_remote_quality_v1',frozen_at=now(),fixture=bind(c/'inputs/inference_fixture.json'),labels=bind(c/'inputs/scoring_only/labels.json'),scorer=bind(c/'public_source/metrics.py'),source_sha256=sources,batch_output=rel(p/'run/attempt1'),items_per_arm=N,documents_per_arm=D,expected_complete_phases_per_arm=phases,expected_total_answers=6*N,expected_total_Writes=6*D,expected_total_phases=6*phases,source_preparation_manifest=bind(c/'inputs/manifest.json'),task_cache_root=cache)
  plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
  ranges=inp['ranges'];plan['actual_frozen_input_lengths']={out:ranges[src] for out,src in [('document_tokens','document_tokens'),('query_tokens','query_tokens'),('logical_prompt_tokens','full_prompt_with_BOS_tokens'),('logical_prompt_plus_cap','full_prompt_with_BOS_plus_generation_reserve')]}
  plan['scheduler']['job_name']=newjob
  plan['input_scope']=f'Public official pinned BABILong {task}/{length}, original 100 source ordinals 0..99; exact default instruction/examples/post_prompt/template, complete context and question. No crop, resampling, filtering or oracle.'
  plan['analysis']['cluster_unit']=f'{D} exact-context document groups; item-weighted shared-entry queries. No pooling across length cells.'
  plan['preparation_scope']='New offline CPU remaining17 package; working cache-fixed runtime and exact official task prompt/scorer. Future root adoption and serialized live duplicate/pending+running GPU-count checks required; no submission in preparation.'
  plan['benchmark_cell']={'task':task,'length':length,'source':bind(c/'original_data.json'),'prompts':bind(c/'public_source/prompts.py'),'source_revision':prep['source_revision'],'document_group_sizes':inp['document_group_sizes']}
  plan['nested_cell_root_binding']['source_public_cell_config']=prep['configuration_source']
  # Keep historical bug-fix provenance explicit; the new source is an unchanged fixed worker/guard.
  plan['triton_cache_cleanup_fix']['module']=bind(p/'qencbank_triton_cache.py')
  plan['remaining17_derivation']={'base_working_plan':bind(BASE/'package/plan.json'),'scope':bind(H/'candidate_scope.json'),'assembler':bind(__file__),'original_public_source':prep['original_public_source'],'original_prompts_source':prep['pinned_prompts_source'],'original_metric_source':prep['pinned_metric_source'],'runtime_seed':42,'bootstrap_seed':20260912,'data_selection':'Original 100 in order; no sampling or generation','derived_N':N,'derived_D':D,'derived_phases_per_arm':phases,'same_directory_depth':True}
  save(p/'plan.json',plan)
  write(p/'batch.sbatch',namespace((BASE/'package/batch.sbatch').read_text('utf-8')).replace(sha(BASE/'package/plan.json'),sha(p/'plan.json')))
  for name in ('dispatch_once.py','remote_ops.py'):
   before=(BASE/name).read_text('utf-8');after=namespace(before);write(c/name,after)
   if before!=after:write(c/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(BASE/name),tofile=rel(c/name))))
  # Preserve original runtime/backend qualification proof closure, plus all new input/source bindings.
  paths={ROOT/x['relative_path'] for x in old_manifest['files'] if not x['relative_path'].startswith(rel(BASE/'package')+'/')}
  paths.update(p.glob('*'));paths.update([BASE/'package/plan.json',H/'candidate_scope.json',Path(__file__),c/'prepare_inputs.py',c/'preparation_plan.json',c/'original_data.json',c/'public_source/prompts.py',c/'public_source/metrics.py',ROOT/prep['configuration_source']['path']])
  paths.update(x for x in (c/'inputs').rglob('*') if x.is_file())
  paths.update([c/'generation_attempt1/execution_receipt.json',c/'generation_controller_attempt1/execution_receipt.json'])
  for path in paths:assert path.is_file(),path
  manifest={k:v for k,v in old_manifest.items() if k!='files'}
  manifest.update(plan_sha256=sha(p/'plan.json'),task_cache_root=cache,scope='Working frozen runtime/backend proof closure plus complete new official input; CPU prepared future formal candidate.')
  manifest['files']=[{'relative_path':rel(x),'local_path':str(x),'remote_path':REMOTE+'/'+rel(x),'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(paths)]
  save(p/'upload_manifest.json',manifest)
  cmd=read(BASE/'remote_commands.json');rp=REMOTE+'/'+rel(p)
  cmd.update(remote_package=rp,plan_sha256=sha(p/'plan.json'),upload_manifest={'path':str(p/'upload_manifest.json'),'sha256':sha(p/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'],scope=plan['preparation_scope'])
  cmd['sbatch_command']=shlex.join(cmd['sbatch_argv']);cmd['roots_and_staged_check_command']='QENCBANK_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(p/'upload_manifest.json')])
  save(c/'remote_commands.json',cmd)
  for value in (cache,control,rp,plan['model_root'],plan['adapter_root'],plan['python']):assert PurePosixPath(value).is_relative_to(PurePosixPath('/srv/encbank')) and '..' not in PurePosixPath(value).parts
  save(c/'source_delta.json',{'status':'ASSEMBLED_CPU_CHECKS_PENDING_NOT_SUBMITTED','base_plan':bind(BASE/'package/plan.json'),'plan':bind(p/'plan.json'),'manifest':bind(p/'upload_manifest.json'),'source_changes':deltas,'change_scope':'Only new cell identifiers/dataset/schema and task-specific official input/scorer bindings plus fresh job/output/cache/control namespaces; same numerical runtime and native admission/cache fix. Derived counts match frozen protocol constants.','method_parameters_unchanged':True,'old_sources_not_modified':True,'no_GPU_or_network_or_submission':True})
  print(task+'/'+length,sha(p/'plan.json'),flush=True)
if __name__=='__main__':
 from pathlib import Path
 main()
