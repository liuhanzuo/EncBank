"""I/O-only wrapper around frozen scientific entry points."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    target = (ROOT / sys.argv[1]).resolve()
    assert target.parent == ROOT
    assert target.name in {'train_large.py', 'evaluate_quant.py'}
    import common
    from io_resilience import install
    install(common)
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
