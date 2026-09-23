import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'hidden_reader_ruler_20260921'
assert Path('/srv/encbank') in R.parents
assert not (R/'configs.json').exists()
shutil.copytree(P/'vendor',R/'vendor');shutil.copytree(P/'scoring_only',R/'scoring_only')
shutil.copyfile(P/'official_scoring.py',R/'official_scoring.py')
shutil.copyfile(P/'input_inventory.json',R/'input_inventory.json')
source=json.loads((P/'configs.json').read_text());configs={}
for cell,c in source.items():
    assert hashlib.sha256(Path(c['input']).read_bytes()).hexdigest()==c['input_sha256']
    configs['screen_'+cell]=dict(**c,cell=cell,phase='screen',start=0,end=32,
        arms=['native']+['n'+str(n) for n in [12,14,16,18,20,24,28,32,36]])
    configs['confirm_'+cell]=dict(**c,cell=cell,phase='confirm',start=32,end=100,arms='selected')
(R/'configs.json').write_text(json.dumps(configs,indent=2)+'\n')
protocol=dict(history_write='independent512-token H12; sink as its own block',
    numbering='H12 is block12 output; n is last jointly executed block; joint blocks13..n, local blocks(n+1)..36',
    local_attention='strict same-chunk causal attention, sink excluded for other chunks; physical equal-length batches without padding',
    positions='packed positions retained for all blocks, including physically batched local phase',
    query_decode='all36 blocks for query/generated tokens; blocks1..12 see query/continuation only; blocks13..36 see every retrieved memory block',
    trainable_parameters=0,projection_heads='original model K/V projections only; no learned cross-depth heads',
    independent_variable='joint causal depth n',fixed='same original Encbank weights, retrieval, prompts, caps, scoring, query visibility',
    n_grid=[12,14,16,18,20,24,28,32,36],screen_items_per_cell=32,confirmation_items_per_cell=68,
    selection_rule='smallest n whose task-mean loss vs native is <=2pp for each of single/multikey/VT and no cell loss exceeds5pp; fallback n36 if none',
    confirm_arms='native,n12,selected_n,n36 with duplicates removed; selection uses only first32/cell',
    timing='H12-ready to all upper history KV ready; two warmups then five wall/CUDA timings, chunk counts1/4/12; excludes retrieval/H12-write/query/decode',
    caveat='full original MLP and projections still execute at every history layer; this isolates attention locality, not the preceding learned-KV-head approximation',
    cache_dependency='post-joint states depend on ordered retrieved prefix; they are not a context-free per-chunk cache',
    benchmark_scope='same3 tasks x3 lengths; NIAH official generator, variable tracking Encbank RULER-style; no full13-task RULER claim')
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(configs=len(configs),source_files=len(manifest))))
