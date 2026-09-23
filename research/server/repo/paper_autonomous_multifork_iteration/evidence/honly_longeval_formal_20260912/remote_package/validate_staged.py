"""Read-only roots, staged hashes and fresh-output check; no weights/GPU/scoring."""
import argparse, json, os
from pathlib import Path
from protocol import HERE,ROOT,REMOTE_ROOT,PYTHON,local,read,sha,ARMS,preflight

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-manifest-sha256',required=True)
    a=p.parse_args();manifest=HERE/'upload_manifest.json'
    assert sha(manifest)==a.expected_manifest_sha256
    m=read(manifest);home=Path('/srv/encbank').resolve()
    assert ROOT==Path(REMOTE_ROOT).resolve() and Path(os.environ['QENCBANK_REPO_ROOT']).resolve()==ROOT
    for value in (ROOT,Path(PYTHON),Path(m['model_root']),Path(m['adapter_root']),Path(m['task_cache_root'])):
        value.resolve().relative_to(home)
    assert Path(PYTHON).is_file() and Path(m['model_root']).is_dir() and Path(m['adapter_root']).is_dir()
    for x in m['files']:
        value=local(x['relative_path']);value.relative_to(home)
        assert value.is_file() and sha(value)==x['sha256'],x['relative_path']
    plan=read(HERE/'plan.json');assert sha(HERE/'plan.json')==m['plan_sha256']
    assert not local(plan['batch_output']).exists(),'Existing batch output; never resume/duplicate'
    for arm in ARMS:
        preflight(argparse.Namespace(plan=str(HERE/'plan.json'),expected_plan_sha256=m['plan_sha256'],arm=arm,output=str(local(plan['outputs'][arm])),check_only=True))
    print(json.dumps({'status':'staged_hashes_root_paths_and_four_native_check_only_PASS','plan_sha256':m['plan_sha256'],'model_weights_rehashed':False,'GPU_used':False,'outputs_created':False}))
if __name__=='__main__':main()
