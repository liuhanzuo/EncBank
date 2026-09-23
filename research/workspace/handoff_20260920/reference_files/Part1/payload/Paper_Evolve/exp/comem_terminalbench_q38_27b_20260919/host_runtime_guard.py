"""Load scantree's installed code without scanning an unrelated parent Git repo."""
import importlib.util
import os
from pathlib import Path


def preload_scantree():
    spec = importlib.util.find_spec('scantree')
    if spec is None or spec.origin is None:
        raise RuntimeError('The existing Harbor runtime must provide scantree')
    # Versioneer walks upward from site-packages. Bound Git discovery to this
    # installed runtime, so its version lookup cannot traverse the paper repo.
    runtime = Path(spec.origin).resolve().parents[4]
    old = os.environ.get('GIT_CEILING_DIRECTORIES')
    os.environ['GIT_CEILING_DIRECTORIES'] = str(runtime)
    try:
        import scantree
        return scantree.__version__
    finally:
        if old is None:
            os.environ.pop('GIT_CEILING_DIRECTORIES', None)
        else:
            os.environ['GIT_CEILING_DIRECTORIES'] = old
