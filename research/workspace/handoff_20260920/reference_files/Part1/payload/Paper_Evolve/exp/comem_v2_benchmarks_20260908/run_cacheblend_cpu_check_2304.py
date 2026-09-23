"""Remote CPU check with a copied current local harness, without replacing it."""
from pathlib import Path
import importlib.util
import runpy
import sys

here = Path(__file__).resolve().parent
snapshot = here / 'serving_reuse_snapshot_2304.py'
if not snapshot.exists():
    snapshot = here / 'serving_reuse.py'
spec = importlib.util.spec_from_file_location('serving_reuse', snapshot)
module = importlib.util.module_from_spec(spec)
sys.modules['serving_reuse'] = module
spec.loader.exec_module(module)
runpy.run_path(str(here / 'test_cacheblend_serving_reuse_cpu.py'), run_name='__main__')
