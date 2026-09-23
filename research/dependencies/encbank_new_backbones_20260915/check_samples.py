import json
from pathlib import Path
from transformers import AutoTokenizer
from run_pilot import make_samples

root=Path(__file__).resolve().parent
for name in ('Qwen3.5-9B','Qwen3.8-27B'):
    tok=AutoTokenizer.from_pretrained(root/'models'/name,local_files_only=True)
    tok.model_max_length=10**9
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    rows=make_samples(tok,sink,1)
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    for row in rows:
        assert json.loads(json.dumps(row))==row
        assert sum(map(len,row['segments']))<=6657
        assert len(row['segments'])<=14
        assert sum(map(len,row['segments']))<=len(row['input_ids'])+1
    print(json.dumps({'model':name,'cpu_sample_checks':True,'n':len(rows),
                      'selected':[len(r['selected']) for r in rows],
                      'source_lengths':[r['source_tokens'] for r in rows]}),flush=True)
