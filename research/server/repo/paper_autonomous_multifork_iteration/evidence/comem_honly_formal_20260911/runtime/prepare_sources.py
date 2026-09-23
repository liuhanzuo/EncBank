"""Freeze exact official primitives and a minimal exact-source codec extraction."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
UPSTREAM=ROOT/'tmp_external_baselines/comem_official/comem'
CODEC=ROOT/'gpu/qcomem_torch.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(UPSTREAM/'model.py')=='939b8d5bd7c9e3ca8f78e878683f4014149b3c156b9bdd5b887d2ac459a0c7df'
assert sha(UPSTREAM/'selectors.py')=='4421e620c2ec47cf20330b35337c924026bfa3034191555e9b09ec2f39181d78'
assert sha(CODEC)=='40e4ebcecd6d05dbe17c836db2d3717b0bcea82453612308c3fd535be82b628f'
vendor=HERE/'vendor_comem'; vendor.mkdir(parents=True,exist_ok=True)
for name in ['model.py','selectors.py']:
    p=vendor/name
    assert not p.exists()
    p.write_bytes((UPSTREAM/name).read_bytes())
(vendor/'__init__.py').write_text('"""Immutable official CoMem primitives; no modified model mathematics."""\n',encoding='utf-8')
source=CODEC.read_text(encoding='utf-8'); lines=source.splitlines(keepends=True)
needed=['tensor_nbytes','_pack_unsigned','_unpack_unsigned','PackedTensor','quantize_tensor']
nodes={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
parts=['from __future__ import annotations\nimport math\nfrom dataclasses import dataclass\nimport torch\nSUPPORTED_BITS = (2, 4, 8, 16)\n']
bindings=[]
for name in needed:
    n=nodes[name]
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])
    text=''.join(lines[start-1:n.end_lineno])
    parts.append(text)
    bindings.append({'name':name,'source_lines':[start,n.end_lineno],
                     'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
out=HERE/'extracted_minmax.py'; assert not out.exists()
out.write_text('\n\n'.join(parts)+'\n',encoding='utf-8',newline='\n')
record={'official_commit':'2c407a6bf9d1503ebb004847f664b01d2ad898cc',
        'official_files':{n:{'source':str(UPSTREAM/n),'copied_sha256':sha(vendor/n)} for n in ['model.py','selectors.py']},
        'codec_source':str(CODEC),'codec_source_sha256':sha(CODEC),
        'codec_extraction':bindings,'extracted_codec_sha256':sha(out),
        'extraction_changes':'imports/constants only; named class/functions copied without arithmetic edits',
        'model_executions':0}
(HERE/'upstream_source_bindings.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
