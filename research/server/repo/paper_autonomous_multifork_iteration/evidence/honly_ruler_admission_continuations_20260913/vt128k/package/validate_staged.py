"""Read-only roots, staged hashes and fresh-output check; no weights/GPU/scoring."""
import argparse, json, os
from pathlib import Path,PurePosixPath
from protocol import HERE,ROOT,REMOTE_ROOT,PYTHON,local,read,sha,ARMS,preflight

def readonly_interpreter_identity():
    # The approved venv entry point is read-only; its system symlink target is not a task write.
    lexical=PurePosixPath(PYTHON);lexical.relative_to(PurePosixPath('/srv/encbank'))
    assert '..' not in lexical.parts and lexical.is_absolute()
    interpreter=Path(PYTHON)
    assert interpreter.is_file() and os.access(interpreter,os.X_OK)
    return {'requested':PYTHON,'resolved':str(interpreter.resolve()),'executable_sha256':sha(interpreter),'read_only_existing_executable':True,'target_is_not_task_managed_storage':True}

def managed_storage_paths(manifest,home):
    for value in (ROOT,Path(manifest['model_root']),Path(manifest['adapter_root']),Path(manifest['task_cache_root'])):
        value.resolve().relative_to(home)

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-manifest-sha256',required=True)
    a=p.parse_args();manifest=HERE/'upload_manifest.json'
    assert sha(manifest)==a.expected_manifest_sha256
    m=read(manifest);home=Path('/srv/encbank').resolve()
    assert ROOT==Path(REMOTE_ROOT).resolve() and Path(os.environ['QENCBANK_REPO_ROOT']).resolve()==ROOT
    managed_storage_paths(m,home)
    interpreter_identity=readonly_interpreter_identity()
    assert Path(m['model_root']).is_dir() and Path(m['adapter_root']).is_dir()
    for x in m['files']:
        value=local(x['relative_path']);value.relative_to(home)
        assert value.is_file() and sha(value)==x['sha256'],x['relative_path']
    plan=read(HERE/'plan.json');assert sha(HERE/'plan.json')==m['plan_sha256']
    from backend_gate import validate_backend_binding
    validate_backend_binding(plan)  # Staging success never admits an unqualified candidate.
    assert not local(plan['batch_output']).exists(),'Existing batch output; never resume/duplicate'
    for arm in ARMS:
        preflight(argparse.Namespace(plan=str(HERE/'plan.json'),expected_plan_sha256=m['plan_sha256'],arm=arm,output=str(local(plan['outputs'][arm])),check_only=True))
    print(json.dumps({'status':'staged_hashes_root_paths_and_6_continuation_native_check_only_PASS','plan_sha256':m['plan_sha256'],'model_weights_rehashed':False,'GPU_used':False,'outputs_created':False,'read_only_interpreter':interpreter_identity}))
if __name__=='__main__':main()
