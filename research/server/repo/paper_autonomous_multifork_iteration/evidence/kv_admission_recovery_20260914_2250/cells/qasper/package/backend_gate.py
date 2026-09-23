"""Hash-bound cluster-native SDPA admission; no KIVI binary or GPU smoke requirement."""
from pathlib import PurePosixPath
from protocol import local,read,sha
RUNTIME={'torch':'2.10.0+cu128','transformers':'5.5.4','peft':'0.20.0','triton':'3.6.0'}
ALLOWED=['gpu-node3','gpu-node4','gpu-node5','gpu-node6','gpu-node7']
def checked_file(spec):
    path=PurePosixPath(spec['path']);assert not path.is_absolute() and '..' not in path.parts
    assert len(spec['sha256'])==64 and sha(local(spec['path']))==spec['sha256'];return local(spec['path'])
def validate_backend_binding(plan,*,allow_unbound=False):
    assert plan['launch_permitted'] is True and plan['attention_backend']=='sdpa' and plan['backbone_dtype']=='float16'
    assert plan['arm_order'] in (['comem_frozen_j12'],['public_kvdirect_j0'])
    assert plan['method_configuration']['adapter_active'] is False and not plan['training_or_runtime_offload'] and not plan['automatic_retry']
    q=plan['backend_qualification'];policy=read(checked_file(q['registration']))
    assert policy['schema']=='native_SDPA_cluster_admission_v1' and policy['status']=='CPU_DERIVED_NATIVE_SDPA_ADMISSION_NOT_NEW_GPU_QUALIFICATION'
    assert q['qualified_nodes']==policy['allowed_nodes']==ALLOWED
    assert q['node_hostnames']==policy['node_hostnames'] and set(q['node_hostnames'])==set(ALLOWED)
    assert q['runtime_versions']==policy['runtime_versions']==RUNTIME
    assert q['compute_capability']==policy['required_compute_capability']==[10,3]
    assert policy['required_actual_GPU_name']=='NVIDIA L20D'
    assert policy['scheduler']=={'partition':'gpu','constraint':'l20d','exclude':['gpu-node1','gpu-node2','gpu-node8'],'nodes':1,'gres':'gpu:1'}
    for key in ('CPU_Slurm_inventory','CPU_reservation_and_account_snapshot','prior_native_SDPA_completion','prior_native_SDPA_plan','prior_native_SDPA_independent_review','prior_native_SDPA_independent_exit'):checked_file(policy[key])
    prior=read(checked_file(policy['prior_native_SDPA_completion']))
    assert prior['job_id']=='24062' and prior['actual_job_and_batch_exit_zero'] and prior['coverage']['KIVI_included'] is False
    assert prior['physical_gpu_identity']['name']=='NVIDIA L20D'
    old=read(checked_file(policy['prior_native_SDPA_plan']))
    assert old['model_root']==plan['model_root'] and old['python']==plan['python'] and old['native_attention_runtime_files']==plan['native_attention_runtime_files']
    assert read(checked_file(old['activation']))['model_file_sha256']==read(checked_file(plan['activation']))['model_file_sha256']
    return True
def qualified_node_identity(plan,environ,hostname):
    q=plan['backend_qualification'];node=environ.get('SLURMD_NODENAME') or environ.get('SLURM_JOB_NODELIST')
    assert node in q['qualified_nodes'],'Allocated Slurm node outside frozen CPU inventory'
    assert hostname in q['node_hostnames'][node],'Allocated hostname differs from frozen Slurm mapping'
    return {'slurm_node':node,'hostname':hostname,'policy':'native_SDPA_cluster_admission_v1','per_node_GPU_smoke_claimed':False}
