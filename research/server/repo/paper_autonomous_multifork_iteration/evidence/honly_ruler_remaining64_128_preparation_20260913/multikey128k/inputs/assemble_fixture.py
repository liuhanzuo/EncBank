"""CPU projection of the unmodified official generated cohort; no inference/scoring."""
from pathlib import Path
import ast,collections,datetime,hashlib,importlib.metadata,json,math,os,sys
O=Path(__file__).resolve().parent;ROOT=Path('/srv/encbank/legacy_workspace');N=ROOT/'paper_autonomous_multifork_iteration/evidence/comem_honly_formal_20260911/inputs'
assert os.environ['PYTHONHASHSEED']=='42' and os.environ['USE_TORCH']=='0'
sys.path.insert(0,str(N/'dependencies/site'))
from transformers import AutoTokenizer
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sth(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def th(v):return sth(json.dumps(v,separators=(',',':'),ensure_ascii=False))
def write(path,value):
 assert not path.exists(),path
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def bound(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
receipt=json.loads((O/'generation_attempt1/execution_receipt.json').read_text())
assert receipt['actual_exit_code']==0 and receipt['actual_parent_wait'] and receipt['process_exit_observed']
rawpath=O/'scoring_only/official_raw_attempt1/niah_multikey_1/validation.jsonl'
raw=[json.loads(s) for s in rawpath.read_text(encoding='utf-8').splitlines()]
boundaries=[json.loads(s) for s in (O/'source_boundaries.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(raw)==len(boundaries)==100
model=ROOT/'models/Qwen3-8B'
tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
config=json.loads((model/'config.json').read_text());genconfig=json.loads((model/'generation_config.json').read_text())
assert config['bos_token_id']==151643 and config['eos_token_id']==151645 and config['max_position_embeddings']==40960
assert genconfig['eos_token_id']==[151645,151643] and tokenizer.eos_token_id==151645
encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
selector=ROOT/'tmp_external_baselines/comem_official/comem/selectors.py'
src=selector.read_text(encoding='utf-8');tree=ast.parse(src)
names=['bm25_scores','iter_bm25_indices'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert [n.name for n in nodes]==names
ns={'math':math,'Counter':collections.Counter}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(selector),'exec'),ns)
class CPUIds(list):
 def tolist(self):return list(self)
documents=[];items=[];lengths=[]
for i,m in enumerate(boundaries):
 assert m['ordinal']==i
 # Only the original public prompt and its length enter input preparation. Gold is never passed here.
 original_full=raw[i]['input']+raw[i]['answer_prefix']
 assert sth(original_full)==m['full_text_sha256']
 document=m['document_text'];query=m['query_text'];bare=m['bare_question_text']
 assert document+query==original_full and document.endswith('\n') and query.startswith('What ')
 assert m['document_query_char_boundary']==len(document) and query.startswith(bare) and bare.endswith('?')
 docids=encode(document);qids=encode(query);bareids=encode(bare);fullids=encode(original_full)
 assert docids+qids==fullids,('BPE_BOUNDARY_NOT_EXACT',i)
 assert len(fullids)+128==raw[i]['length'] and raw[i]['length']<=131072
 assert 1+len(fullids)+48<=131072
 assert 151643 not in fullids and 151645 not in fullids
 chunks=[docids[j:j+512] for j in range(0,len(docids),512)]
 assert [v for c in chunks for v in c]==docids and all(0<len(c)<=512 for c in chunks)
 chosen=ns['iter_bm25_indices']([CPUIds(c) for c in chunks],bareids,topk=12,iter_rounds=0,iter_hop_topk=4)
 assert chosen==sorted(set(chosen)) and len(chosen)<=12 and all(0<=j<len(chunks) for j in chosen)
 item_id=f'ruler_niah_multikey_1_128k_seed42_{i:03d}';doc_id=f'{item_id}_document'
 docs_sha=th(docids);query_sha=th(qids);full_prompt=[151643]+fullids
 documents.append({'document_id':doc_id,'source_ordinal':i,'document_text':document,'document_token_ids':docids,'document_token_sha256':docs_sha,
  'prefix_token_ids':[151643],'chunks':chunks,'chunk_token_sha256':[th(c) for c in chunks], 'chunk_sizes':[len(c) for c in chunks],
  'raw_context_char_bounds':m['raw_context_char_bounds'],'document_query_char_boundary':len(document),
  'write_input_scope':'fixed task instruction + generated context + fixed newline; no question or answer prefix'})
 items.append({'sample_index':i,'item_id':item_id,'document_id':doc_id,'task':'niah_multikey_1','subset':'validation',
  'query_text':query,'bare_question_text':bare,'query_token_ids':qids,'bare_question_token_ids':bareids,
  'query_token_sha256':query_sha,'bare_question_token_sha256':th(bareids),'selected_chunk_indices':chosen,
  'selected_chunk_token_sha256':[th(chunks[j]) for j in chosen],
  'full_prompt_text':original_full,'full_prompt_token_ids':full_prompt,'full_prompt_token_sha256':th(full_prompt),
  'plain_prompt_without_bos_sha256':sth(original_full),'prefix_token_ids':[151643],
  'max_new_tokens':48,'eos_token_id':151645,'eos_token_ids':[151645],
  'official_prompt_tokens_without_bos':len(fullids),'official_max_seq_length':131072,'official_generation_reserve':128,
  'inference_prompt_tokens_including_bos':len(full_prompt),'selected_read_tokens_including_bos_query':1+sum(len(chunks[j]) for j in chosen)+len(qids),
  'input_pair_sha256':th({'document_token_sha256':docs_sha,'query_token_sha256':query_sha})})
 lengths.append({'sample_index':i,'document_tokens':len(docids),'query_tokens':len(qids),'bare_question_tokens':len(bareids),
  'official_prompt_tokens':len(fullids),'official_prompt_plus_reserve':raw[i]['length'],'inference_full_prompt_plus_cap':len(full_prompt)+48,
  'chunks':len(chunks),'selected_chunks':len(chosen),'selected_read_tokens':items[-1]['selected_read_tokens_including_bos_query']})
assert len({x['item_id'] for x in items})==len({x['full_prompt_token_sha256'] for x in items})==100
assert len({x['document_token_sha256'] for x in documents})==100
protocol={'task':'niah_multikey_1','official_ruler_commit':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a','seed':42,'PYTHONHASHSEED':42,
 'replica_scope':'new fixed official-generator cohort for local Qwen3-8B/custom4000 matching replica, not public CoMem checkpoint or same published examples',
 'official_max_seq_length':131072,'official_generation_reserve':128,'inference_generation_cap':48,'prompt_format':'plain, no chat template',
 'bos_token_id':151643,'prefix_token_ids':[151643],'eos_token_id':151645,'eos_token_ids':[151645],'do_sample':False,
 'eos_resolution':'single tokenizer/config EOS per official CoMem; apply identically to every arm; original generation_config double EOS is metadata only',
 'original_generation_config_eos_token_ids':genconfig['eos_token_id'],'original_model_max_position_embeddings':40960,
 'chunk_size':512,'chunk_overlap':0,'chunk_positions':'each document chunk starts locally at zero; explicit BOS sink outside chunk pool',
 'selector':'iter_bm25','topk':12,'iter_hop_topk':4,'iter_rounds':0,'resolved_rounds':3,
 'selection_order':'rank during each hop, then final ascending original chunk index; zero-score guard retained',
 'query_scope':'entire official question and answer prefix, never truncated; retrieval sees bare question only',
 'document_scope':'question-independent source context and fixed instruction/separator; never fused last512 split',
 'token_hash_format':'SHA256 of UTF-8 compact JSON integer array',
 'BPE_contract':'for all100: encode(document)+encode(query)==encode(original complete plain prompt), add_special_tokens=False',
 'metric':'official NVIDIA RULER string_match_all(preds:list[str], refs:list[list[str]]) -> rounded percent',
 'no_ground_truth_in_retriever':True,'no_model_inference':True,'generation_sampling':'all official100 in order, no gold or output selection'}
fixture={'schema':'qcomem_official_ruler_input_v1','status':'FROZEN_CPU_INPUTS_NO_MODEL_EXECUTION','protocol':protocol,'documents':documents,'items':items}
write(O/'fixture.json',fixture)
# This sidecar is for a separate CPU scorer. Original gold-derived positional metadata stays here only.
labels={'schema':'ruler_scoring_labels_v1','fixture_sha256':sha(O/'fixture.json'),'items':[{'sample_index':i,'item_id':items[i]['item_id'],'references':r['outputs'],'original_saved_index':r['index'],'original_token_position_answer':r['token_position_answer']} for i,r in enumerate(raw)]}
assert all(len(x['references'])==1 for x in labels['items'])
write(O/'scoring_only/labels.json',labels)

write(O/'lengths.json',lengths)
sources=[SRC for SRC in (N/'official_ruler/scripts/data/synthetic/niah.py',N/'official_ruler/scripts/synthetic.yaml',N/'official_ruler/scripts/data/synthetic/constants.py',N/'official_ruler/scripts/data/template.py',N/'official_ruler/scripts/data/tokenizer.py',N/'official_ruler/scripts/data/synthetic/json/PaulGrahamEssays.json',N/'official_ruler/scripts/eval/synthetic/constants.py',N/'official_ruler/scripts/eval/evaluate.py',selector,O/'generate_official.py',O/'assemble_fixture.py',O/'generation_plan.json',O/'generation_attempt1/execution_receipt.json',O/'source_boundaries.jsonl',rawpath)]
sources.extend(model/name for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json') if (model/name).exists())
write(O/'input_manifest.json',{'sources':[bound(path) for path in sources], 'reused_dependency_manifest':bound(N/'input_manifest.json'), 'new_task':'niah_multikey_1','num_needle_k':4,'num_needle_v':1,'num_needle_q':1,'generation_no_model':True,'official_generator_source_unmodified':True,'source_corpus_scope':'same pinned xKV-vendored RULER PaulGraham essay snapshot as completed single2, not an invented upstream download'})
summary={'status':'READY_NEW_MULTIKEY128K100_CPU_FIXTURE_NO_GPU','items':100,'documents':100,'unique_item_ids':len({x['item_id'] for x in items}),'unique_document_hashes':len({x['document_token_sha256'] for x in documents}),'full_BPE_concat_matches':100,'ranges':{key:{'min':min(row[key] for row in lengths),'max':max(row[key] for row in lengths)} for key in lengths[0] if key!='sample_index'},'fixture':bound(O/'fixture.json'),'labels':bound(O/'scoring_only/labels.json'),'input_manifest':bound(O/'input_manifest.json'),'torch_imported':'torch' in sys.modules,'inference_or_scoring_executed':False,'same_scorer_and_postprocess_as_single2':True,'source_reserve128_inference_cap48':True}
assert not summary['torch_imported'];write(O/'preparation_summary.json',summary)
print(json.dumps(summary,indent=2))
