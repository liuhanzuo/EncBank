"""One remote allocated GPU, exact Blackwell-compatible binary, unchanged tiny kernel tests."""
import datetime,gc,os,sys,traceback
from pathlib import Path
from protocol import HERE,ROOT,RESOURCE,local,module,parser,preflight,read,save,sha
from resource_guard import query,require_identity
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
    args=parser().parse_args();plan,_=preflight(args,envelope=not args.check_only)
    if args.check_only:print('{"status":"CPU_source_and_record_bindings_checked_binary_remote_GPU_pending"}');return
    assert Path(os.environ['QCOMEM_REPO_ROOT']).resolve()==ROOT
    out=Path(args.output);parent=read(out/'execution.json');identity=parent['physical_gpu_identity']
    assert parent['status']=='worker_running' and parent['gpu_worker_started'] and parent['stable_interval_seconds']>=45
    record={'status':'preparing','pid':os.getpid(),'arm':args.arm,'started_at':now(),'plan_sha256':args.expected_plan_sha256,'model_loaded':False,'training_allowed':False,'runtime_offload_allowed':False,'quality_evidence':False,'profile_evidence':False,'physical_gpu_identity':identity}
    save(out/'worker.json',record)
    try:
        assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='backend:native'
        for key in ('TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
            path=out/key.lower();path.resolve().relative_to(Path('/srv/encbank'));path.mkdir(exist_ok=False);os.environ[key]=str(path)
        import torch,importlib.metadata
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','transformers','peft')};assert versions==plan['versions'],versions
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        def forbidden(*a,**kw):raise RuntimeError('No backward or training in kernel qualification')
        torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
        assert torch.cuda.device_count()==1 and torch.cuda.get_allocator_backend()=='native'
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        record['actual_torch_properties']={'major':int(props.major),'minor':int(props.minor),'name':props.name,'uuid':str(props.uuid),'total_memory':int(props.total_memory)}
        save(out/'worker.json',record)
        assert [props.major,props.minor]==plan['required_compute_capability'],'Allocated architecture differs from frozen cc10.3 candidate; preserve properties and stop'
        assert str(props.uuid).lower().removeprefix('gpu-')==identity['uuid'].lower().removeprefix('gpu-')
        total=int(props.total_memory);assert total>=RESOURCE['minimum_free_bytes']
        torch.cuda.set_per_process_memory_fraction(RESOURCE['allocator_cap_bytes']/total,0)
        assert abs(torch.cuda.get_per_process_memory_fraction(0)*total-RESOURCE['allocator_cap_bytes'])<1
        assert torch.cuda.current_stream()==torch.cuda.default_stream(),'Original Half tests are default-stream qualified only'
        snapshot=query();require_identity(snapshot,identity);free,actual_total=torch.cuda.mem_get_info(0)
        assert free>=RESOURCE['minimum_free_bytes']
        record['pre_allocation_checks']=[dict(snapshot,cuda_free_bytes=int(free),cuda_total_bytes=int(actual_total))]
        record['cuda_policy']={'allocator_backend':'native','allocator_cap_bytes':RESOURCE['allocator_cap_bytes'],'device_total_bytes':total,'device_name':props.name,'device_uuid':str(props.uuid),'compute_capability':[props.major,props.minor],'default_stream_only':True}
        save(out/'worker.json',record)
        assert sha(local(plan['binary']['path']))==plan['binary']['sha256']
        # Exact path imports before the test; no shadowable late campaign imports.
        kernels_module=module(HERE/'bridge.py','bridge')
        profile=module(HERE/'phase_profile.py','phase_profile')
        checks=module(HERE/'kernel_checks.py','_qcomem_remote_kivi_kernel_checks')
        profiler=profile.Profiler(torch)
        with torch.inference_mode(),profiler.phase('outer_kernel_import_synthetic_checks_and_release'):
            binary=module(local(plan['binary']['path']),'kivi_gemv')
            assert sha(Path(binary.__file__))==plan['binary']['sha256']
            kernels=kernels_module.OfficialHalfKernels(local(plan['kivi_root']),plan['kivi_source_sha256'])
            result=checks.run(torch,kernels,plan,out,profiler)
            del kernels;gc.collect();torch.cuda.synchronize()
            record['after_tensor_release']={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        assert not profiler.stack and all(p['completed'] for p in profiler.records)
        assert len(result['cases'])==4 and len(result['tail_cases'])==2 and len(profiler.records)==7
        for path,digest in plan['source_sha256'].items():assert sha(local(path))==digest,path
        assert sha(local(plan['binary']['path']))==plan['binary']['sha256']
        result.update(status='worker_complete_parent_exit_pending',outer_closed=True,provenance={'plan_sha256':args.expected_plan_sha256,'source_sha256':plan['source_sha256'],'binary':plan['binary'],'versions':versions,'cuda_policy':record['cuda_policy'],'physical_gpu_identity':identity,'slurm_job_id':os.environ['SLURM_JOB_ID'],'seed':plan['seed'],'checkpoint_loaded':False,'cpu_dequantization':'oracle only after actual CUDA outputs, never implementation fallback'})
        save(out/'result.json',result);record.update(status='completed',scientific_result_sha256=sha(out/'result.json'))
    except BaseException as error:
        record.update(status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},oom=isinstance(error,torch.cuda.OutOfMemoryError) if 'torch' in locals() else False);raise
    finally:
        if 'profiler' in locals():record['phases']=profiler.records
        record['finished_at']=now();save(out/'worker.json',record)
if __name__=='__main__':main()
