"""Freeze tokenizer-derived synthetic inputs without initializing CUDA/model."""
from pathlib import Path
import hashlib,json,importlib.metadata,os
from transformers import AutoTokenizer
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
H.resolve().relative_to(Path('/srv/encbank').resolve())
tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
def repeat(text,n):
    ids=tok.encode(text,add_special_tokens=False)
    assert ids and not set(ids).intersection(tok.all_special_ids)
    return (ids*((n+len(ids)-1)//len(ids)))[:n]
rows=[]
for s in json.loads((H/'input_sources.json').read_text()):
    doc=repeat(s['document'],P['chunk_tokens']*P['selected_chunks'])
    # Distinct independent chunks, even if repeated source text aligns by chance.
    chunks=[doc[i:i+P['chunk_tokens']] for i in range(0,len(doc),P['chunk_tokens'])]
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    rows.append(dict(session=s['session'],chunks=[[sink]]+chunks,
        query=repeat(s['query'],P['query_tokens']),replay=repeat(s['replay'],P['output_positions']-1)))
assert len(rows)==3 and all(len(r['chunks'])==13 for r in rows)
assert rows[0]['query']!=rows[1]['query']!=rows[2]['query']
assert not (H/'inputs.json').exists()
(H/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
cp=Path(P['adapter_path']).resolve();cp.relative_to(Path('/srv/encbank').resolve())
assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']
record=dict(status='PASS',model_calls=0,cuda_context=False,input_sha256=hashlib.sha256((H/'inputs.json').read_bytes()).hexdigest(),
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
    lengths=[dict(session=r['session'],selected_tokens=sum(map(len,r['chunks'])),query_tokens=len(r['query']),replay_tokens=len(r['replay'])) for r in rows])
(H/'cpu_preflight_remote.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
