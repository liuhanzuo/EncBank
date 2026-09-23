"""CPU fault injection against the exact captured official FileCacheManager."""
import ast
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAVED = ROOT / 'paper_autonomous_multifork_iteration/evidence/honly_babilong_qa1_0k_admission_fixed_20260912/failure_25084/saved/installed_triton_cache.py'
OUT = HERE / 'focused_checks_attempt1'
WORK = OUT / 'test_artifacts'
WORK.mkdir(exist_ok=False)
checks = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Only Triton's version/knob containers are stubbed. Its captured complete
# cache.py and inherited get/get_group/put_group/__init__ execute unchanged.
cache_knobs = types.SimpleNamespace(dir=str(WORK), dump_dir=str(WORK/'dump'), override_dir=str(WORK/'override'), manager_class=None)
triton = types.ModuleType('triton')
triton.__version__ = '3.6.0'
triton.knobs = types.SimpleNamespace(cache=cache_knobs)
triton.__path__ = []
runtime = types.ModuleType('triton.runtime')
runtime.__path__ = []
sys.modules.update({'triton': triton, 'triton.runtime': runtime})
official = load('triton.runtime.cache', SAVED)
runtime.cache = official
fix = load('qencbank_triton_cache', HERE/'qencbank_triton_cache.py')
fix.REMOTE_HOME = WORK  # Test-only local boundary; production constant unchanged.
assert hashlib.sha256(SAVED.read_bytes()).hexdigest() == fix.EXPECTED_OFFICIAL_SOURCE_SHA256
assert fix.CleanupTolerantFileCacheManager.__bases__ == (official.FileCacheManager,)
checks.append('exact_captured_official_source_and_inherited_implementation')


def manager(key):
    return fix.CleanupTolerantFileCacheManager(key)


def raises(kind, function, errno_expected=None):
    try:
        function()
    except kind as error:
        if errno_expected is not None:
            assert error.errno == errno_expected
        return error
    raise AssertionError('Expected ' + kind.__name__)


# Normal successful byte/text/non-string and binary-argument coercion semantics.
for i, (data, binary) in enumerate([(b'\x00\xffkernel', False), ('text\n\u03bb\n', True), (12345, False)]):
    old = official.FileCacheManager('official'+str(i))
    new = manager('normal'+str(i))
    a = old.put(data, 'kernel', binary=binary)
    b = new.put(data, 'kernel', binary=binary)
    assert Path(a).read_bytes() == Path(b).read_bytes()
    assert new.get_file('kernel') == b and not list(Path(new.cache_dir).glob('tmp.pid_*'))
checks.append('normal_bytes_text_unicode_newlines_and_nonstring_match_official')
group = manager('group')
child = group.put(b'data', 'child')
group.put_group('group', {'child': child})
assert group.get_group('group') == {'child': child}
checks.append('inherited_group_roundtrip')


def busy(path):
    raise OSError(errno.EBUSY, 'fault injected retained leaf', path)


for i, data in enumerate([b'\x00\xffcompiled', 'text\n\u03bb\n']):
    m = manager('busy'+str(i))
    capture = io.StringIO()
    with patch.object(fix.os, 'removedirs', side_effect=busy), contextlib.redirect_stderr(capture):
        result = m.put(data, 'artifact')
    event = json.loads(capture.getvalue())
    assert event['event'] == 'triton_cache_cleanup_ebusy_retained'
    assert event['committed_bytes_verified'] and event['errno'] == errno.EBUSY
    assert event['artifact'] == result and Path(event['retained_temp_dir']).is_dir()
    assert event['artifact_sha256'] == hashlib.sha256(Path(result).read_bytes()).hexdigest()
checks.append('EBUSY_only_after_successful_replace_verified_bytes_text_and_structured_log')


for operation, number in [('makedirs', errno.EBUSY), ('replace', errno.EBUSY), ('removedirs', errno.EACCES)]:
    m = manager('propagate_'+operation)
    def fail(*args, **kwargs):
        raise OSError(number, 'fault injected '+operation, args[0])
    capture = io.StringIO()
    with patch.object(fix.os, operation, side_effect=fail), contextlib.redirect_stderr(capture):
        raises(OSError, lambda: m.put(b'data', 'artifact'), number)
    assert not capture.getvalue()
checks.append('mkdir_EBUSY_replace_EBUSY_cleanup_EACCES_propagate_without_success_log')
m = manager('write_error')
with patch.object(fix, 'open', side_effect=OSError(errno.ENOSPC, 'write unavailable'), create=True):
    raises(OSError, lambda: m.put(b'data', 'artifact'), errno.ENOSPC)
assert not Path(m._make_path('artifact')).exists()
checks.append('open_write_failure_propagates_no_final_artifact')

for mode in ['mismatch', 'missing', 'other_leaf', 'missing_filename']:
    m = manager(mode)
    def cleanup(path):
        if mode == 'mismatch':
            Path(m._make_path('artifact')).write_bytes(b'corrupt')
        elif mode == 'missing':
            Path(m._make_path('artifact')).unlink()
        if mode == 'other_leaf':
            raise OSError(errno.EBUSY, 'different path', str(Path(path).parent))
        if mode == 'missing_filename':
            raise OSError(errno.EBUSY, 'no proven leaf')
        busy(path)
    capture = io.StringIO()
    with patch.object(fix.os, 'removedirs', side_effect=cleanup), contextlib.redirect_stderr(capture):
        expected = RuntimeError if mode == 'mismatch' else FileNotFoundError if mode == 'missing' else OSError
        raises(expected, lambda: m.put(b'data', 'artifact'))
    assert not capture.getvalue()
checks.append('mismatched_missing_artifact_and_unproven_EBUSY_fail_closed')

raises(ValueError, lambda: manager('../escape'))
m = manager('path_boundary')
raises(ValueError, lambda: m.put(b'x', '../../escape'))
checks.append('resolved_cache_key_and_output_paths_stay_in_explicit_boundary')

with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(WORK)}):
    receipt = fix.install(str(WORK))
assert cache_knobs.manager_class is fix.CleanupTolerantFileCacheManager
assert isinstance(official.get_cache_manager('ab'), fix.CleanupTolerantFileCacheManager)
assert Path(cache_knobs.dump_dir).is_relative_to(WORK)
assert Path(cache_knobs.override_dir).is_relative_to(WORK)
with patch.object(triton, '__version__', 'unexpected'):
    raises(RuntimeError, lambda: fix.install(str(WORK)))
with patch.object(fix, 'EXPECTED_OFFICIAL_SOURCE_SHA256', '0'*64):
    raises(RuntimeError, lambda: fix.install(str(WORK)))
with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(WORK/'different')}):
    raises(RuntimeError, lambda: fix.install(str(WORK)))
checks.append('process_local_install_official_selection_and_version_source_path_guards')

tree = ast.parse((HERE/'qencbank_triton_cache.py').read_text('utf-8'))
assert not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and ast.unparse(t) == 'os.removedirs' for t in n.targets) for n in ast.walk(tree))
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
checks.append('no_global_removedirs_patch_no_model_framework_or_GPU')
report = {'status':'PASS_FOCUSED_CPU_CACHE_CLEANUP_FAULT_INJECTION', 'checks':checks, 'count':len(checks),
          'official_source_sha256':fix.EXPECTED_OFFICIAL_SOURCE_SHA256,
          'module_sha256':hashlib.sha256((HERE/'qencbank_triton_cache.py').read_bytes()).hexdigest(),
          'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'installed_runtime_used':False, 'test_runtime':'captured actual cache.py executed with version/knob containers stubbed; real CPU filesystem and inherited official methods',
          'production_installer_rechecks_actual_runtime_source':True,
          'no_SSH_GPU_shared_environment_or_existing_outputs_changed':True}
(OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report))
