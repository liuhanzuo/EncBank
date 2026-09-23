"""Pre-allocation import check using exactly the worker's dependency path."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'vendor'),'/srv/encbank/comem_infra_recheck_20260912/deps']
import peft,torch,transformers
from band_reader import JointBand
from common import save
assert not torch.cuda.is_initialized()
save(R/'dependency_check.json',dict(passed=True,peft=peft.__version__,peft_path=peft.__file__,torch=torch.__version__,transformers=transformers.__version__,cuda_initialized=False,model_calls=0))
print('Dependency imports passed; no CUDA context or model inference.')
