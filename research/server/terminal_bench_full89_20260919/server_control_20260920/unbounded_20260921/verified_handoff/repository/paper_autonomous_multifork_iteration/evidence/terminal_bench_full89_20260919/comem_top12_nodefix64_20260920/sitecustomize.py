"""Only this campaign's future WSL Harbor children use the stable-cwd shim."""
import os,sys
from pathlib import Path
if sys.platform=='linux' and str(Path(__file__).resolve()).startswith('/srv/encbank/legacy_workspace/'):
    os.environ['PATH']=str(Path(__file__).resolve().parent/'bin')+os.pathsep+os.environ['PATH']
