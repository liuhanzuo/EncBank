"""Isolate old model APIs while reusing the validated CUDA12.8 torch binaries read-only."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/srv/encbank')
WORK = ROOT / 'qcomem_align_codex_20260911/locomo_paper_astra_expansion_20260916/memoryllm_preparation'
ENV = ROOT / 'qcomem_runtime_20260911/memoryllm_torch210_20260916'
BASE = ROOT / 'qcomem_runtime_20260911/python312/bin/python'


def main():
    for path in [WORK, ENV]:
        assert str(path.resolve()).startswith(str(ROOT) + '/')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PIP_CACHE_DIR=str(WORK / 'pip_cache'), TMPDIR=str(WORK / 'tmp'))
    # BASE is the already registered interpreter, whose symlink may resolve to /usr/bin;
    # only the new venv and its own cache directories are written.
    subprocess.run([str(BASE), '-m', 'venv', str(ENV)], check=True, env=env)
    python = str(ENV / 'bin/python')
    base_site = subprocess.check_output([str(BASE), '-c',
                'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True, env=env).strip()
    assert str(Path(base_site).resolve()).startswith(str(ROOT) + '/')
    local_site = subprocess.check_output([python, '-c',
                'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True, env=env).strip()
    assert str(Path(local_site).resolve()).startswith(str(ENV) + '/')
    (Path(local_site) / 'validated_torch210_readonly.pth').write_text(base_site + '\n')
    with (WORK / 'runtime_torch210_install.log').open('w') as log:
        subprocess.run([python, '-m', 'pip', 'install', 'transformers==4.48.2', 'peft==0.10.0',
                        'accelerate==1.2.0', 'numpy==1.26.4', 'einops==0.8.0',
                        'packaging==25.0', 'sentencepiece'], check=True, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], env=env, text=True)
    (WORK / 'runtime_torch210_freeze.txt').write_text(freeze)
    checks = {}
    for name, path in [('torch251', str(ROOT / 'qcomem_runtime_20260911/memoryllm_20260916/bin/python')),
                       ('torch210', python)]:
        code = ('import torch,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,'
                'architecture_flags=torch._C._cuda_getArchFlags(),cuda_initialized=torch.cuda.is_initialized(),'
                'torch_file=torch.__file__)))')
        checks[name] = json.loads(subprocess.check_output([path, '-c', code], env=env, text=True))
    assert checks['torch210']['torch'] == '2.10.0+cu128', checks
    assert not checks['torch210']['cuda_initialized']
    subprocess.run([python, str(WORK / 'check_official_cpu.py'), 'official_cpu_check_torch210.json'],
                   check=True, env=env)
    result = {'status': 'COMPLETE_CPU_QUALIFIED', 'python': python, 'architecture_checks': checks,
              'existing_torch_site_packages': base_site, 'reuse_policy':
              'Existing validated Torch2.10 CUDA12.8 binary read-only; local transformers4.48.2/peft0.10.0 APIs shadow base packages. Existing runtime unchanged.',
              'gpu_executions': 0}
    (WORK / 'runtime_torch210_status.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
