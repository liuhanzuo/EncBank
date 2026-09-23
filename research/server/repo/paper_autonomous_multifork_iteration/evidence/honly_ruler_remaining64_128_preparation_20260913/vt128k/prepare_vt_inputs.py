"""CPU-only official CoMem VT functions; no scientific generator function changed."""
from pathlib import Path
import ast, collections, datetime, hashlib, json, math, os, random, re, string, sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE/'inputs'
assert os.environ['PYTHONHASHSEED']=='42' and os.environ['USE_TORCH']=='0'
assert not OUT.exists();OUT.mkdir();(OUT/'scoring_only').mkdir()
SRC=ROOT/'tmp_external_baselines/comem_official/eval/ruler.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def bind(p):return {'path':Path(p).relative_to(ROOT).as_posix(),'sha256':sha(p)}
tree=ast.parse(SRC.read_text(encoding='utf-8'))
names={'_gen_chain','_make_vt','_make_vt_icl','_render_vt','_build_sample','_bare_question','_string_match_all_one'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'VT_TEMPLATE','VT_ANSWER_PREFIX','NOISE_HAYSTACK'} for t in n.targets)]
assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names
ns={'random':random,'string':string};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC),'exec'),ns)
dep=ROOT/'paper_autonomous_multifork_iteration/evidence/comem_honly_formal_20260911/inputs/dependencies/site'
sys.path.insert(0,str(dep));from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-8B'),local_files_only=True,trust_remote_code=False)
task='variable_tracking';length='128k';budget=131072;cap=60
base_seed=42+(hash((task,length))%100000)
plan={'status':'FROZEN_BEFORE_GENERATION','source':bind(SRC),'AST_function_names':sorted(names),'generator_functions_unmodified':True,'task':task,'length':length,'target_prompt_tokens':budget,'generation_cap':cap,'source_cap_expression':'max(args.max_new_tokens,60), default max_new_tokens=48','source_original_generator':'CoMem embedded generation, not legacy xKV NIAH generator','PYTHONHASHSEED':42,'seed':42,'resolved_base_seed':base_seed,'n':100,'ICL_examples':1,'num_hops':4,'num_chains':1,'prompt_length_boundary':'Official target refers to prompt tokens, full native unscaled positions exceed original40960 window; no artificial 128K totalcap','source_method_scope':'New fixed official-generated cohort, not published examples or published checkpoint'}
save(OUT/'generation_plan.json',plan)
icl=ns['_make_vt_icl'](random.Random(base_seed+777),4)
items=[];labels=[];raw=[];lengths=[]
for i in range(100):
 prompt,answers,gold=ns['_build_sample'](task,budget,tok,random.Random(base_seed*1000+i),icl)
 assert gold is None and len(answers)==5 and len(set(answers))==5
 split=prompt.rfind('\nQuestion:');assert split>len(icl)
 document=prompt[:split+1];query=prompt[split+1:]
 assert query.startswith('Question: Find all variables') and query.count(' Answer:')==1
 # Official CoMem retrieval query is its complete trailing question/answer-prefix line.
 bare=ns['_bare_question'](prompt);assert bare==query.strip()
 enc=lambda s:tok.encode(s,add_special_tokens=False)
 d,q,b=enc(document),enc(query),enc(bare);full=enc(prompt)
 assert d+q==full and len(full)<=budget and len(full)+1+cap<=budget+1+cap
 assert tok.encode(prompt,add_special_tokens=True)==full and 151643 not in full and 151645 not in full
 ident=f'ruler_variable_tracking_128k_seed42_{i:03d}'
 items.append({'id':ident,'dataset':'ruler_variable_tracking_128k','source_id':ident,'source_ordinal':i,'document_id':ident+'_document','document_token_ids':d,'query_token_ids':q,'bare_question_token_ids':b,'prefix_token_ids':[151643],'max_new_tokens':cap,'eos_token_id':151645,'full_prompt_with_BOS_token_sha256':th([151643]+full)})
 labels.append({'id':ident,'references':answers})
 raw.append({'id':ident,'input':prompt,'outputs':answers,'official_source_seed':base_seed*1000+i})
 lengths.append({'id':ident,'document_tokens':len(d),'query_tokens':len(q),'bare_question_tokens':len(b),'logical_prompt_tokens':len(full)+1,'logical_prompt_plus_cap':len(full)+1+cap})
assert len({th(x['document_token_ids']) for x in items})==100
save(OUT/'inference_fixture.json',{'schema':'inference_only_official_CoMem_VT_128k100_v1','items':items})
save(OUT/'scoring_only/labels.json',{'fixture_sha256':sha(OUT/'inference_fixture.json'),'items':labels})
(OUT/'scoring_only/official_raw.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in raw),encoding='utf-8')
save(OUT/'lengths.json',lengths)
manifest={'status':'PASS_100_official_VT_CPU_inputs_no_GPU','source':bind(SRC),'generator':bind(__file__),'plan':bind(OUT/'generation_plan.json'),'fixture':bind(OUT/'inference_fixture.json'),'labels':bind(OUT/'scoring_only/labels.json'),'raw':bind(OUT/'scoring_only/official_raw.jsonl'),'items':100,'unique_documents':100,'exact_document_query_BPE_concatenations':100,'query_independent_document':True,'tokenizer_files':{n:sha(ROOT/'models/Qwen3-8B'/n) for n in ['tokenizer.json','tokenizer_config.json','config.json']},'ranges':{k:{'minimum':min(x[k] for x in lengths),'maximum':max(x[k] for x in lengths)} for k in lengths[0] if k!='id'},'torch_imported':'torch' in sys.modules,'scoring_or_model_execution':False}
assert not manifest['torch_imported'];save(OUT/'manifest.json',manifest);print(json.dumps(manifest,indent=2))
