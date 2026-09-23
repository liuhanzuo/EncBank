"""CPU projection of the unmodified official generated cohort; no inference/scoring."""
from pathlib import Path
import ast,collections,datetime,hashlib,importlib.metadata,json,math,os,sys
O=Path(__file__).resolve().parent;ROOT=Path('/srv/encbank/legacy_workspace')
assert os.environ['PYTHONHASHSEED']=='42' and os.environ['USE_TORCH']=='0'
sys.path.insert(0,str(O/'dependencies/site'))
from transformers import AutoTokenizer
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sth(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def th(v):return sth(json.dumps(v,separators=(',',':'),ensure_ascii=False))
def write(path,value):
 assert not path.exists(),path
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def bound(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
receipt=json.loads((O/'generation_attempt2/execution_receipt.json').read_text())
assert receipt['actual_exit_code']==0 and receipt['actual_parent_wait'] and receipt['process_exit_observed']
rawpath=O/'scoring_only/official_raw_attempt2/niah_single_2/validation.jsonl'
raw=[json.loads(s) for s in rawpath.read_text(encoding='utf-8').splitlines()]
boundaries=[json.loads(s) for s in (O/'source_boundaries_v2.jsonl').read_text(encoding='utf-8').splitlines()]
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
 assert len(fullids)+128==raw[i]['length'] and raw[i]['length']<=8192
 assert 1+len(fullids)+48<=8192
 assert 151643 not in fullids and 151645 not in fullids
 chunks=[docids[j:j+512] for j in range(0,len(docids),512)]
 assert [v for c in chunks for v in c]==docids and all(0<len(c)<=512 for c in chunks)
 chosen=ns['iter_bm25_indices']([CPUIds(c) for c in chunks],bareids,topk=12,iter_rounds=0,iter_hop_topk=4)
 assert chosen==sorted(set(chosen)) and len(chosen)<=12 and all(0<=j<len(chunks) for j in chosen)
 item_id=f'ruler_niah_single_2_8k_seed42_{i:03d}';doc_id=f'{item_id}_document'
 docs_sha=th(docids);query_sha=th(qids);full_prompt=[151643]+fullids
 documents.append({'document_id':doc_id,'source_ordinal':i,'document_text':document,'document_token_ids':docids,'document_token_sha256':docs_sha,
  'prefix_token_ids':[151643],'chunks':chunks,'chunk_token_sha256':[th(c) for c in chunks], 'chunk_sizes':[len(c) for c in chunks],
  'raw_context_char_bounds':m['raw_context_char_bounds'],'document_query_char_boundary':len(document),
  'write_input_scope':'fixed task instruction + generated context + fixed newline; no question or answer prefix'})
 items.append({'sample_index':i,'item_id':item_id,'document_id':doc_id,'task':'niah_single_2','subset':'validation',
  'query_text':query,'bare_question_text':bare,'query_token_ids':qids,'bare_question_token_ids':bareids,
  'query_token_sha256':query_sha,'bare_question_token_sha256':th(bareids),'selected_chunk_indices':chosen,
  'selected_chunk_token_sha256':[th(chunks[j]) for j in chosen],
  'full_prompt_text':original_full,'full_prompt_token_ids':full_prompt,'full_prompt_token_sha256':th(full_prompt),
  'plain_prompt_without_bos_sha256':sth(original_full),'prefix_token_ids':[151643],
  'max_new_tokens':48,'eos_token_id':151645,'eos_token_ids':[151645],
  'official_prompt_tokens_without_bos':len(fullids),'official_max_seq_length':8192,'official_generation_reserve':128,
  'inference_prompt_tokens_including_bos':len(full_prompt),'selected_read_tokens_including_bos_query':1+sum(len(chunks[j]) for j in chosen)+len(qids),
  'input_pair_sha256':th({'document_token_sha256':docs_sha,'query_token_sha256':query_sha})})
 lengths.append({'sample_index':i,'document_tokens':len(docids),'query_tokens':len(qids),'bare_question_tokens':len(bareids),
  'official_prompt_tokens':len(fullids),'official_prompt_plus_reserve':raw[i]['length'],'inference_full_prompt_plus_cap':len(full_prompt)+48,
  'chunks':len(chunks),'selected_chunks':len(chosen),'selected_read_tokens':items[-1]['selected_read_tokens_including_bos_query']})
assert len({x['item_id'] for x in items})==len({x['full_prompt_token_sha256'] for x in items})==100
assert len({x['document_token_sha256'] for x in documents})==100
protocol={'task':'niah_single_2','official_ruler_commit':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a','seed':42,'PYTHONHASHSEED':42,
 'replica_scope':'new fixed official-generator cohort for local Qwen3-8B/custom4000 matching replica, not public CoMem checkpoint or same published examples',
 'official_max_seq_length':8192,'official_generation_reserve':128,'inference_generation_cap':48,'prompt_format':'plain, no chat template',
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
evalsrc=O/'official_ruler/scripts/eval/synthetic/constants.py'
evaltree=ast.parse(evalsrc.read_text());scorenode=next(n for n in evaltree.body if isinstance(n,ast.FunctionDef) and n.name=='string_match_all')
write(O/'scorer_interface.json',{'source':bound(evalsrc),'function':'string_match_all','source_line':scorenode.lineno,'end_line':scorenode.end_lineno,
 'ast_sha256':sth(ast.dump(scorenode,include_attributes=False)), 'arguments':{'preds':'list[str] in fixture order','refs':'list[list[str]], from scoring_only/labels.json references'},
 'return':'float, mean fraction of case-insensitive gold substrings found per item times100, rounded2 decimals',
 'single_answer_scope':'100questions each have1reference, so equals case-insensitive answer-substring accuracy, NOT F1',
 'example_call_without_data':"string_match_all(preds, [labels_by_id[item['item_id']]['references'] for item in fixture['items']])",'executed_on_model_outputs':False})
selector_binding={'source':bound(selector),'functions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':sth(ast.dump(n,include_attributes=False))} for n in nodes],
 'execution':'exact AST-extracted original functions; list subclass supplies only .tolist(); no torch import and no source edits'}
write(O/'selector_binding.json',selector_binding)
bindings=[bound(p) for p in sorted((O/'official_ruler').rglob('*')) if p.is_file()]
for name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json']:
 if (model/name).exists():bindings.append(bound(model/name))
adapter=ROOT/'models/CoMem-Qwen3-8B-final4000-20260909/adapter_config.json';bindings.append(bound(adapter))
for name in ['source_receipt.json','dependency_patch_v2.json','generation_plan_v2.json','generation_attempt2/execution_receipt.json','generate_official_v2.py','source_boundaries_v2.jsonl','assemble_fixture.py']:
 bindings.append(bound(O/name))
bindings.extend([bound(rawpath),bound(selector),bound(O/'dependencies/tenacity-9.1.2-py3-none-any.whl'),bound(O/'dependencies/wonderwords-2.2.0-py3-none-any.whl'),bound(O/'dependencies/punkt_tab.zip')])
write(O/'input_manifest.json',{'bindings':bindings,'not_rehashed':'full model/adapter weight files; runtime must bind separately from existing verified manifests',
 'generated_corpus_provenance':'pinned NVIDIA generator requires generated PaulGrahamEssays.json not tracked by its git; exact xKV-vendored RULER corpus snapshot reused, SHA and original path in source_receipt.json',
 'official_index_boundary':'official niah.py overwrites index with answer char position. New ordinal IDs preserve all rows without treating original index as unique; raw/index/gold position preserved scorer-only.',
 'versions':{k:importlib.metadata.version(k) for k in ['transformers','tokenizers','numpy','nltk','wonderwords','tenacity','PyYAML']},'python':sys.version})
summary={'status':'READY_FIXED_CPU_INPUTS_PENDING_FORMAL_RUNTIME','time':datetime.datetime.now().astimezone().isoformat(),'items':100,'documents':100,'source_boundary_matches':100,'token_concat_exact_matches':100,
 'unique_item_ids':100,'unique_full_prompt_hashes':100,'unique_document_token_hashes':100,
 'ranges':{k:{'min':min(r[k] for r in lengths),'max':max(r[k] for r in lengths)} for k in lengths[0] if k!='sample_index'},
 'all_chunks_selected_count':sum(r['chunks']==r['selected_chunks'] for r in lengths),'torch_imported':'torch' in sys.modules,
 'fixture':bound(O/'fixture.json'),'labels':bound(O/'scoring_only/labels.json'),'selector':bound(O/'selector_binding.json'),'scorer':bound(O/'scorer_interface.json'),
 'source_manifest':bound(O/'input_manifest.json'),'prior_generation_preflight_failure':bound(O/'generation_attempt1/execution_receipt.json'),
 'limitations':['New official-generator cohort, not exact published CoMem examples','Local Qwen3-8B/custom final4000 adapter matching replica, not public continued Base identity','No inference, quality score, capacity or timing evidence','8K is official budget8192 with128 generation reserve; runtimecap48 and explicit BOS recorded separately; no forced exact-length padding','One RULER task/length cell only; not complete RULER or all CoMem main benchmarks']}
assert not summary['torch_imported'];write(O/'lengths.json',lengths);write(O/'preparation_summary.json',summary)
print(json.dumps({k:summary[k] for k in ['status','items','documents','ranges','all_chunks_selected_count','fixture','labels','torch_imported']},indent=2))
