"""Keep the existing 300 LongEval examples and add 300 independently seeded examples."""
import collections, gzip, json, random, zlib
import config
import common as original
from comem.selectors import iter_bm25_indices
from eval import longeval
from transformers import AutoTokenizer
from train_support import atomic_json, digest


def main():
    if config.SAMPLES.exists():
        print('Using existing predefined evaluation set', flush=True)
        return
    tok = AutoTokenizer.from_pretrained(config.MODEL, local_files_only=True)
    tok.model_max_length = 10**9
    with gzip.open(config.FOLLOW/'data/kv_samples.jsonl.gz','rt',encoding='utf-8') as f:
        old = [json.loads(line) for line in f]
    rows = []
    for row in old:
        if row['benchmark'] == 'longeval':
            row['split'] = 'existing'; rows.append(row)
    assert len(rows) == 300
    for length in ('8k','16k','32k'):
        seed = 20260917 + zlib.crc32(length.encode())%100000
        for i in range(100):
            prompt, answer, label, nlines = longeval.build_lines_prompt(longeval._LENGTH_TOKENS[length], tok, random.Random(seed*1000+i))
            row = dict(id=f'confirmation_longeval_{length}_{i:03d}', benchmark='longeval', cell=f'longeval_{length}',
                split='confirmation', index=i, length=length, input_ids=tok.encode(prompt,add_special_tokens=True),
                question_ids=tok.encode(f'line {label}',add_special_tokens=False), answers=[answer], budget=16,
                seed=seed*1000+i, target_label=label, n_lines=nlines)
            chunks, query = original.parts(row)
            row['selected'] = iter_bm25_indices(chunks,row['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
            rows.append(row)
        print(length, 'prepared', flush=True)
    assert len({r['id'] for r in rows}) == 600
    assert not {r['seed'] for r in rows if r['split']=='existing'} & {r['seed'] for r in rows if r['split']=='confirmation'}
    for row in rows:
        chunks, _ = original.parts(row); original.selection(chunks,row)
    config.SAMPLES.parent.mkdir(exist_ok=True)
    tmp = config.SAMPLES.with_suffix('.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(config.SAMPLES)
    atomic_json(config.ROOT/'data/protocol.json', dict(examples=600,
        cells=dict(collections.Counter(r['split']+'/'+r['cell'] for r in rows)),
        original_source=str(config.FOLLOW/'data/kv_samples.jsonl.gz'), confirmation_seed=20260917,
        samples_sha256=digest(config.SAMPLES), j=12, overlap=0, chunk=512, topk=12, hop_topk=4, iter_rounds=0,
        source_order=True, budget=16, greedy=True, eval_bos=151643, first_token_eos_suppressed=True,
        score='original extract_prediction exact match', checkpoint='final only', train_data='PG19 only',
        note='Both splits are evaluation sets, never used for training/checkpoint selection; report all predefined arms.'))


if __name__=='__main__':
    main()
