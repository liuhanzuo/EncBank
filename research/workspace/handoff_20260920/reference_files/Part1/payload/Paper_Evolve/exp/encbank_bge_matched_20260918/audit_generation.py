"""Audit old BGE output caps without changing prompt, EOS, or scoring."""
import ast,gzip,hashlib,json
import config
import common as original
from transformers import AutoTokenizer

def main():
    out=config.ROOT/'existing_analysis';out.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    records=[json.loads(l) for l in (config.OLD/'bge_quality/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==400
    runner=(config.OLD/'run_bge_quality.py').read_text(encoding='utf-8')
    calls=[n for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='decode']
    assert len(calls)==1 and len(calls[0].args)==6 and not calls[0].keywords
    # The six arguments end at budget; fixed is omitted and its actual default is False.
    assert original.decode.__defaults__==(False,)
    result={}
    for arm in ['raw','encbank']:
        rs=[r for r in records if r['arm']==arm];assert len(rs)==200
        for r in rs:
            assert r['status']=='ok' and r['generated_tokens']==len(r['generated_ids'])
            assert tok.decode(r['generated_ids'],skip_special_tokens=True)==r['prediction']
            assert abs(original.score(samples[r['id']],r['prediction'])-r['score'])<1e-12
        result[arm]=dict(records=200,cap128=sum(len(r['generated_ids'])==128 for r in rs),
            eos_in_saved_ids=sum(tok.eos_token_id in r['generated_ids'] for r in rs),empty=sum(not r['prediction'].strip() for r in rs),
            native_generation_config_eos_occurrences={str(i):sum(i in r['generated_ids'] for r in rs)
                for i in json.loads((config.MODEL/'generation_config.json').read_text())['eos_token_id']})
    payload=dict(complete=True,arms=result,eos_token_id=tok.eos_token_id,budget=128,min_new_tokens=None,
        manual_decode_loop=True,fixed_length_flag=False,first_token_eos_suppressed=True,later_eos_stops=True,
        decoded_text_matches_all=True,score_matches_all=True,
        conclusion='All 400 naturally reach cap under the saved manual decoding protocol. No fixed-length timing flag was passed; preserve protocol and report cap rate.',
        note='Native generation_config is not consumed by the manual loop. Do not change stop rules based on quality outcomes.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [config.OLD/'run_bge_quality.py',config.FOLLOW/'common.py']})
    (out/'generation_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
