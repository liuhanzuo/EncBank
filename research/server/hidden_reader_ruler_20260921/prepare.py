"""Freeze paired retrieval fixtures without giving answer labels to inference."""
import hashlib,json,shutil,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
E=BASE/'repo/paper_autonomous_multifork_iteration/evidence'
shutil.copytree(BASE/'hidden_reader_four_distill_20260921/vendor',ROOT/'vendor',dirs_exist_ok=True)
sys.path.insert(0,str(ROOT/'vendor'))
from comem.selectors import iter_bm25_indices
import torch
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,separators=(',',':'))+'\n')
paths=[
('single8k','honly_fp16_ruler8k_pair_preparation_20260912/single8k/inputs'),
('multikey8k','honly_ruler_multikey_preparation_20260912/inputs'),
('vt8k','honly_ruler_parallel_preparation_20260912/inputs'),
('single32k','honly_ruler_parallel_preparation_20260912/single32k/inputs'),
('multikey32k','honly_ruler_parallel_preparation_20260912/multikey32k/inputs'),
('vt32k','honly_ruler_parallel_preparation_20260912/vt32k/inputs'),
('single128k','honly_ruler_remaining64_128_preparation_20260913/single128k/inputs'),
('multikey128k','honly_ruler_remaining64_128_preparation_20260913/multikey128k/inputs'),
('vt128k','honly_ruler_remaining64_128_preparation_20260913/vt128k/inputs')]
assert not (ROOT/'protocol.json').exists()
configs={};inventory=[]
for cell,rel in paths:
    root=E/rel;fixture=root/'inference_fixture.json'
    if cell=='multikey8k':fixture=root.parent/'package/inference_fixture.json'
    labelpaths=list((root/'scoring_only').glob('*labels.json'));assert len(labelpaths)==1,labelpaths
    labelpath=labelpaths[0]
    items=json.loads(fixture.read_text())['items'];labels=json.loads(labelpath.read_text())['items']
    labels={x.get('id',x.get('item_id')):x['references'] for x in labels}
    assert len(items)==len(labels)==100
    packed=[]
    for x in items:
        assert x['id'] in labels
        ids=x['document_token_ids'];chunks=list(torch.tensor(ids).split(512))
        selected=iter_bm25_indices(chunks,x['bare_question_token_ids'],12,iter_rounds=0,iter_hop_topk=4)
        assert selected==sorted(set(selected)) and len(selected)<=12
        assert x.get('prefix_token_ids',[151643])==[151643]
        assert x.get('eos_token_id',151645)==151645
        cap=60 if cell.startswith('vt') else 48
        assert x.get('max_new_tokens',cap)==cap
        packed.append(dict(id=x['id'],document_id=x['document_id'],source_document_tokens=len(ids),
            source_document_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            selected_indices=selected,selected_chunks=[chunks[i].tolist() for i in selected],
            query_token_ids=x['query_token_ids'],bare_question_token_ids=x['bare_question_token_ids'],
            prefix_token_ids=[151643],eos_token_id=151645,max_new_tokens=cap))
    p=ROOT/'inputs'/(cell+'.json');save(p,packed)
    lp=ROOT/'scoring_only'/(cell+'.json');save(lp,labels)
    row=dict(cell=cell,source_fixture=str(fixture),source_fixture_sha256=sha(fixture),source_labels=str(labelpath),
        source_labels_sha256=sha(labelpath),input_sha256=sha(p),labels_sha256=sha(lp),items=len(items),
        min_selected=min(len(x['selected_indices']) for x in packed),max_selected=max(len(x['selected_indices']) for x in packed),
        provenance='COMem embedded RULER-style variable tracking, hop 4' if cell.startswith('vt') else 'official NVIDIA RULER NIAH generator, frozen existing COMem fixture')
    inventory.append(row);configs[cell]=dict(input=str(p),input_sha256=sha(p),items=100)
    print(json.dumps(row),flush=True)
save(ROOT/'configs.json',configs);save(ROOT/'input_inventory.json',inventory)
scorer=E/'comem_honly_formal_20260911/inputs/official_ruler/scripts/eval/synthetic/constants.py'
assert sha(scorer)=='6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f'
shutil.copyfile(scorer,ROOT/'official_scoring.py')
save(ROOT/'protocol.json',dict(at=datetime.datetime.now().astimezone().isoformat(),
    model='Qwen3-8B',adapter='original unmerged COMem LoRA',chunk_tokens=512,topk=12,
    selector='iter_bm25',iter_hop_topk=4,iter_rounds=0,skip_nonpositive_scores=True,sort_packing='source order',
    arms=['comem','kd256','kd_selected','kd2048'],selection='minimum held-out validation KL across both longer runs; RULER never used for checkpoint selection',
    long_endpoint='2048 total updates from the run selected by validation; separately reported, never substituted for best validation checkpoint',
    history_depths=[12,16,20,24],history_write='chunk independent, no hidden checkpoint after H24',
    lower_query_attention='query and generated continuation only, identical for all arms',
    upper_query_attention='packed retrieved memory and causal query/continuation, identical mask and positions',
    generation='greedy, first-step EOS suppressed, scalar EOS 151645, no prompt/template edits',
    memory_execution='only retrieved chunks written lazily; projected KV resident during decode; quality test, not full-bank write or end-to-end performance benchmark',
    score='official string_match_all reference substring recall, case insensitive, percent',official_scorer_sha256=sha(scorer),
    scope='3 tasks x 3 lengths x 100 paired items; NOT the complete 13-task RULER score; VT is separately identified as COMem RULER-style'))
