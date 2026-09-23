"""CPU-only public BABILong exact official prompt projection; no model or scorer."""
import ast,collections,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def text_sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def token_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def save(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
 plan=json.loads((H/'preparation_plan.json').read_text('utf-8'));task=plan['task'];length=plan['length']
 for name,digest in plan['input_source_sha256'].items():assert sha(ROOT/name)==digest,name
 prompt=ROOT/plan['prompts_source'];tree=ast.parse(prompt.read_text('utf-8'))
 assert all(isinstance(n,(ast.Assign,ast.FunctionDef)) for n in tree.body)
 env={};exec(compile(tree,str(prompt),'exec'),env)
 cfg={k:env['DEFAULT_PROMPTS'][task][k] for k in ('instruction','examples','post_prompt')}
 cfg.update(template=env['DEFAULT_TEMPLATE'],chat_template=False,system_prompt='')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-8B',local_files_only=True,use_fast=True)
 assert tok.is_fast and tok.eos_token_id==151645
 public=json.loads((ROOT/plan['public_data']).read_text('utf-8'));assert len(public)==100
 items=[];labels=[];texts=[];lens=[]
 for i,sample in enumerate(public):
  assert set(sample)=={'input','question','target'} and all(isinstance(v,str) for v in sample.values())
  # Target never participates in prompt rendering, document construction or tokenization.
  full=env['get_formatted_input'](sample['input'],sample['question'],cfg['examples'],cfg['instruction'],cfg['post_prompt'],template=cfg['template'])
  sentinel='QCOMEM_BOUNDARY_SENTINEL_NEVER_SENT_TO_MODEL'
  rendered=env['get_formatted_input'](sample['input'],sentinel,cfg['examples'],cfg['instruction'],cfg['post_prompt'],template=cfg['template'])
  marker='Question: '+sentinel;assert rendered.endswith(marker)
  prefix=rendered[:-len(marker)];assert prefix.endswith('>\n\n')
  document=prefix[:-3];assert full.startswith(document)
  query=full[len(document):];assert query=='>\n\nQuestion: '+sample['question'].rstrip()
  assert document+'?'+sentinel!=full and sentinel not in full
  bare=sample['question'].strip()
  dids=tok.encode(document,add_special_tokens=False);qids=tok.encode(query,add_special_tokens=False);bids=tok.encode(bare,add_special_tokens=False)
  original=tok.encode(full,add_special_tokens=True);plain=tok.encode(full,add_special_tokens=False)
  assert original==plain and dids+qids==plain,('Exact whole-prompt token boundary failed',i)
  assert len(plain)+1+20<=40960 and dids and qids and bids
  item_id=f'babilong_{task}_{length}_{i:03d}';docid='babilong_doc_'+text_sha(document)
  row={'id':item_id,'source_id':item_id,'source_ordinal':i,'dataset':f'babilong_{task}_{length}','document_id':docid,'document_token_ids':dids,'query_token_ids':qids,'bare_question_token_ids':bids,'prefix_token_ids':[151643],'eos_token_id':151645,'max_new_tokens':20,'full_prompt_with_BOS_token_sha256':token_sha([151643]+plain),'metadata':{'task':task,'length':length,'document_text_sha256':text_sha(document),'query_text_sha256':text_sha(query),'bare_question_text_sha256':text_sha(bare),'source_context_text_sha256':text_sha(sample['input']),'source_full_prompt_text_sha256':text_sha(full),'source_full_prompt_token_sha256':token_sha(original),'separator_contract':'final context closing > plus two newlines belongs to query prefix uniformly; all text preserved'}}
  items.append(row);labels.append({'id':item_id,'document_id':docid,'target':sample['target'],'question':sample['question'],'task':task})
  texts.append({'id':item_id,'document_id':docid,'document_text':document,'query_text':query,'bare_question_text':bare})
  lens.append({'id':item_id,'document_tokens':len(dids),'query_tokens':len(qids),'bare_question_tokens':len(bids),'full_prompt_with_BOS_tokens':len(plain)+1,'full_prompt_with_BOS_plus_generation_reserve':len(plain)+21,'document_chunks_512':(len(dids)+511)//512})
 O=H/'inputs';O.mkdir(exist_ok=False);(O/'scoring_only').mkdir()
 save(O/'inference_fixture.json',{'schema':'BABILong_public_official_prompt_complete_input_v1','items':items})
 save(O/'scoring_only/labels.json',{'schema':'BABILong_official_compare_answers_CPU_only_v1','fixture_sha256':sha(O/'inference_fixture.json'),'items':labels})
 with (O/'raw_text.jsonl').open('x',encoding='utf-8',newline='\n') as f:
  for row in texts:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 save(O/'lengths.json',{'items':lens})
 counts=collections.Counter(x['document_id'] for x in items)
 ranges={k:{'minimum':min(x[k] for x in lens),'maximum':max(x[k] for x in lens)} for k in lens[0] if k!='id'}
 manifest={'status':'CPU_INPUTS_PREPARED_NOT_QUALITY','task':task,'length':length,'items':100,'unique_documents':len(counts),'document_group_sizes':list(counts.values()),'expected_Writes_per_arm':len(counts),'expected_phases_per_arm':4*len(counts)+201,'ranges':ranges,'source_revision':plan['source_revision'],'source_input':plan['public_data'],'source_input_sha256':sha(ROOT/plan['public_data']),'prompts_sha256':sha(prompt),'full_prompt_exact_original_tokenization':True,'question_independent_document_construction':True,'target_used_for_prompt_or_selection':False,'prompt_configuration':cfg,'no_crop_or_ROPE_change':True,'python':sys.version,'versions':{k:importlib.metadata.version(k) for k in ('transformers','tokenizers')},'plan_sha256':sha(H/'preparation_plan.json'),'preparation_source_sha256':sha(Path(__file__)),'files':{p:{'sha256':sha(O/p),'bytes':(O/p).stat().st_size} for p in ('inference_fixture.json','scoring_only/labels.json','raw_text.jsonl','lengths.json')},'limits':['One public100-question task/length cell, not21-cell macro','Across-length instances may share underlying bAbI questions; no pooled independence claim','Shared exact documents use one entry and all their queries; no artificial grouping']}
 save(O/'manifest.json',manifest);print(json.dumps({'status':manifest['status'],'documents':len(counts),'ranges':ranges,'manifest_sha256':sha(O/'manifest.json')}))
if __name__=='__main__':main()
