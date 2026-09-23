"""Source, adapter identity, runtime import and input checks; zero model calls."""
import ast,hashlib,inspect,json,time
from pathlib import Path
H=Path(__file__).resolve().parent
started=time.time()
manifest=json.loads((H/'source_manifest.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,name
    if name.endswith('.py'):ast.parse((H/name).read_text())
from native_common import MODELS,tokenizer
from runtime_identity import resolve_configs
from model_setup import tokens
from comem_peft import file_sha
from hybrid_hot import Request,quantum
plan=json.loads((H/'plan.json').read_text());protocol=json.loads((H/'protocol.json').read_text())
identity,cfg=resolve_configs(plan,MODELS[1])
assert file_sha(plan['adapter_path'])==plan['adapter_sha256']
adapter=Path(protocol['peft_adapter']);receipt=json.loads((adapter/'conversion_receipt.json').read_text())
assert receipt['module_count']==333 and receipt['roundtrip_exact']
for name,digest in receipt['files'].items():assert file_sha(adapter/name)==digest,name
assert inspect.signature(quantum).parameters['steps'].default==32
assert file_sha(protocol['recorded_request'])==protocol['recorded_request_sha256']
tok=tokenizer(cfg);ids=tokens(tok,json.loads(Path(protocol['recorded_request']).read_text())['messages'])
assert len(ids)>max(x['history_tokens'] for x in protocol['runs'])+600
assert len({x['id'] for x in protocol['runs']})==12
result=dict(status='PASS',epoch=time.time(),seconds=time.time()-started,model_calls=0,gpu_allocations=0,
    corpus_tokens=len(ids),adapter_files=receipt['files'],source_manifest_sha256=file_sha(H/'source_manifest.json'),
    tests=['runtime imports','all immutable source SHA','training/runtime identity','adapter SHA','existing PEFT conversion hashes','recorded tokenized input capacity','unique cases','quantum interface'])
with (H/'cpu_preflight.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
