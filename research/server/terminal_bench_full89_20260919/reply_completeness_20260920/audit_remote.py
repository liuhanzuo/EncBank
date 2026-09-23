"""Tokenizer-only decode/termination audit against all stored remote responses."""
from pathlib import Path
from collections import Counter
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from transformers import AutoTokenizer
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
S=H.parent/'encbank_refill_retest_20260919';P=json.loads((S/'plan.json').read_text());M=Path(P['model']).resolve();M.relative_to(Path('/srv/encbank').resolve())
g=json.loads((M/'generation_config.json').read_text());stop=g['eos_token_id'];stop=set(stop if isinstance(stop,list) else [stop]);tok=AutoTokenizer.from_pretrained(M,local_files_only=True)
rows=[]
for path in sorted((S/'run_encbank/mailbox').glob('*.response.json')):
    r=json.loads(path.read_text());ids=r.get('generated_ids',[])
    assert r['generated_tokens']==len(ids)
    assert tok.decode(ids,skip_special_tokens=True)==r['text'],'Decoded text mismatch'
    terminal=bool(ids) and ids[-1] in stop
    if r['status']=='ok':assert terminal or r['hit_generation_cap']
    limit=min(P['max_new_tokens'],P['context_tokens']-r['full_history_tokens'])
    assert len(ids)<=limit
    rows.append(dict(request_id=r['request_id'],status=r['status'],generated_tokens=len(ids),terminal_eos=terminal,cap=r['hit_generation_cap'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),limit=limit))
out=dict(status='PASS',model_calls=0,GPU_calls=0,stop_token_ids=sorted(stop),stop_token_names={str(x):tok.convert_ids_to_tokens(x) for x in stop},
    generation_config_sha256=hashlib.sha256((M/'generation_config.json').read_bytes()).hexdigest(),tokenizer_config_sha256=hashlib.sha256((M/'tokenizer_config.json').read_bytes()).hexdigest(),
    formal_worker_sha256=hashlib.sha256((S/'agent_worker.py').read_bytes()).hexdigest(),formal_service_sha256=hashlib.sha256((S/'service_loop.py').read_bytes()).hexdigest(),
    remote_responses=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),all_decodes_match=True,all_lengths_match=True,
    ok_natural_eos=sum(r['status']=='ok' and r['terminal_eos'] and not r['cap'] for r in rows),ok_capped=sum(r['status']=='ok' and r['cap'] for r in rows),rows=rows)
(H/'remote_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
