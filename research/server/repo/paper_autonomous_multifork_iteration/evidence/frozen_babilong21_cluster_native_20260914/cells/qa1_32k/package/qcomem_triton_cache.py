"""Process-local Triton 3.6.0 cache cleanup repair; no installed files are changed.

FileCacheManager.put is copied from the captured installed source, with an
exception handler around only its final os.removedirs call. Compilation,
directory creation, writing and atomic replacement errors still propagate.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import triton
from triton import knobs
from triton.runtime import cache as official_cache
from triton.runtime.cache import FileCacheManager

REMOTE_HOME = Path('/srv/encbank')
EXPECTED_TRITON_VERSION = '3.6.0'
EXPECTED_OFFICIAL_SOURCE_SHA256 = '0355c3e3452fa4500122244b8fff8864f22bc05c417a3d6f5dada9f2e92f8040'


def _within_home(path):
    resolved = Path(path).resolve()
    resolved.relative_to(REMOTE_HOME.resolve())
    return resolved


class CleanupTolerantFileCacheManager(FileCacheManager):
    def __init__(self, key, override=False, dump=False):
        base = knobs.cache.dump_dir if dump else knobs.cache.override_dir if override else knobs.cache.dir
        if not base:
            raise RuntimeError('Explicit Triton cache directory is required')
        root = _within_home(base)
        target = _within_home(root / key)
        target.relative_to(root)
        super().__init__(key, override=override, dump=dump)

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        _within_home(filepath).relative_to(Path(self.cache_dir).resolve())
        # Random ID to avoid any collisions
        rnd_id = str(uuid.uuid4())
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
            write_encoding = None if binary else f.encoding
            write_errors = None if binary else f.errors
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        try:
            os.removedirs(temp_dir)
        except OSError as error:
            # os.removedirs can walk upwards: tolerate only this retained leaf.
            if (error.errno != errno.EBUSY or error.filename is None or
                    Path(error.filename).resolve() != Path(temp_dir).resolve() or
                    not Path(temp_dir).is_dir()):
                raise
            expected = data if binary else data.replace('\n', os.linesep).encode(write_encoding, write_errors)
            actual = Path(filepath).read_bytes()
            if actual != expected:
                raise RuntimeError('Committed Triton cache artifact does not match written bytes') from error
            print(json.dumps({
                'event': 'triton_cache_cleanup_ebusy_retained', 'pid': pid,
                'retained_temp_dir': temp_dir, 'artifact': filepath,
                'artifact_sha256': hashlib.sha256(actual).hexdigest(),
                'artifact_bytes': len(actual), 'committed_bytes_verified': True,
                'error_type': type(error).__name__, 'errno': error.errno,
                'error': str(error), 'error_filename': error.filename,
            }, sort_keys=True), file=sys.stderr, flush=True)
        return filepath


def install(expected_cache_dir):
    """Called in the new worker before torch/model/KIVI imports; CPU-only setup."""
    if triton.__version__ != EXPECTED_TRITON_VERSION:
        raise RuntimeError('Triton version differs from the captured failed runtime')
    source = Path(official_cache.__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_OFFICIAL_SOURCE_SHA256:
        raise RuntimeError('Installed Triton cache.py differs from the frozen source')
    expected = _within_home(expected_cache_dir)
    if _within_home(os.environ['TRITON_CACHE_DIR']) != expected or _within_home(knobs.cache.dir) != expected:
        raise RuntimeError('Triton cache directory differs from the frozen task path')
    if knobs.cache.manager_class not in (None, CleanupTolerantFileCacheManager):
        raise RuntimeError('Another Triton cache manager is already configured')
    knobs.cache.dump_dir = str(expected / 'dump')
    knobs.cache.override_dir = str(expected / 'override')
    knobs.cache.manager_class = CleanupTolerantFileCacheManager
    if knobs.cache.manager_class is not CleanupTolerantFileCacheManager:
        raise RuntimeError('Process-local Triton cache manager selection failed')
    return {
        'manager': __name__ + '.CleanupTolerantFileCacheManager',
        'module_file': str(Path(__file__).resolve()),
        'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'triton_version': triton.__version__, 'official_cache_file': str(source),
        'official_cache_sha256': digest, 'cache_dir': str(expected),
        'scope': 'process-local; only verified post-replace leaf cleanup EBUSY tolerated',
        'cleanup_events': 'JSON lines in this worker stderr.log',
    }
