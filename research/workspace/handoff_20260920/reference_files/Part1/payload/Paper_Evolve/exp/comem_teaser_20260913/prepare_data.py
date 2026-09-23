"""Export the existing, verified Table 2(a) measurements for editable figures."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'COMem/paper_iclr2027_rewrite_20260912/figures/teaser_data.json'
def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8'))
short=read('exp/comem_dense5090_20260913/short_summary.json')
long=read('exp/comem_dense5090_20260913/summary.json')
e2e=read('exp/comem_e2e_20260913/summary.json')
assert short['complete'] and long['complete'] and e2e['complete']
data={}
for n in (8192,16384): data[str(n)]=short['summary'][str(n)]
for n,folder in [(32768,'comem_infra_20260912'),(131072,'comem_infra_128k_20260912')]:
    cm=read(f'exp/{folder}/summary.json')['summary']['comem_k12']
    data[str(n)]={'dense':long['summary'][str(n)],'comem':{
        'prefill':{'status':'ok',**cm['read']},'ttft':{'status':'ok',**cm['ttft']},
        'e2e':{'status':'ok',**e2e['summary'][str(n)]['comem_k12']}}}
result={'source':'Existing Table 2(a) measurements; no new experiments or inferred Dense timings',
        'gpu':'RTX 5090','model':'Qwen3-8B','adapter':'shared unmerged FP32 rank-32 final4000',
        'length_units':'source token IDs, excluding 512 query tokens and BOS',
        'latency_units':'milliseconds in JSON; seconds in figures','memory_units':'decimal GB, CUDA allocated peak including weights',
        'cap_GB':28,'cap_scope':'total process CUDA allocator; driver/desktop context overhead separate',
        'retained_comem_cap_GB':26,'aggregation':'median of three process medians; memory is maximum formal allocated peak',
        'whiskers':'range of three process medians, not confidence intervals',
        'ttft':'pretokenized query through first logits; retrieval/fetch/query Write included for CoMem, document Write excluded',
        'e2e':'fresh full-source preparation (all document Write for CoMem) through 128 generated IDs; no amortization',
        'excluded':'model loading, tokenization/detokenization, external I/O, queue/network',
        'oom':'actual failures; no y-value, no interpolation/extrapolation past last completed Dense point',
        'cohorts':'three unscored PG19 documents per length; same inputs within a length, different queries across lengths',
        'repetitions':'new 8k/16k: 1 warmup + 3 repetitions per document/process; retained 32k/128k CoMem Prefill/TTFT: 5+20; retained E2E: 1+3',
        'data':data}
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(OUT)
