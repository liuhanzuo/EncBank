"""Execute unmodified official niah.py; collect read-only source boundaries."""
from pathlib import Path
import os,sys,json,runpy,hashlib,importlib.metadata,ast,datetime
O=Path(__file__).resolve().parent;ROOT=Path(r'/srv/encbank/legacy_workspace');N=ROOT/'paper_autonomous_multifork_iteration/evidence/comem_honly_formal_20260911/inputs'
assert os.environ.get('PYTHONHASHSEED')=='42'
assert os.environ.get('USE_TORCH')=='0'
DATA=N/'official_ruler/scripts/data';SRC=DATA/'synthetic/niah.py'
sys.path[:0]=[str(N/'dependencies/site'),str(DATA),str(DATA/'synthetic')]
import yaml
from transformers import AutoTokenizer
config=yaml.safe_load((N/'official_ruler/scripts/synthetic.yaml').read_text())['niah_multikey_1']
task=runpy.run_path(str(DATA/'synthetic/constants.py'))['TASKS'][config['task']]
templates=runpy.run_path(str(DATA/'template.py'))['Templates']
assert config['args']=={'type_haystack':'essay','type_needle_k':'words','type_needle_v':'numbers','num_needle_k':4,'num_needle_v':1,'num_needle_q':1}
assert task['tokens_to_generate']==128 and templates['base']=='{task_template}'
template=templates['base'].format(task_template=task['template'])+task['answer_prefix']
save_root=O/'scoring_only/official_raw_attempt1'
assert not save_root.exists()
args=[str(SRC),'--save_dir',str(save_root),'--save_name','niah_multikey_1','--subset','validation',
 '--tokenizer_path',str(ROOT/'models/Qwen3-8B'),'--tokenizer_type','hf','--max_seq_length','131072',
 '--tokens_to_generate','128','--num_samples','100','--random_seed','42','--template',template]
for k,v in config['args'].items():args.extend(['--'+k,str(v)])
metadata={}
def key(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def profile(frame,event,arg):
 if event=='return' and frame.f_code.co_name=='generate_input_output' and Path(frame.f_code.co_filename).resolve()==SRC.resolve():
  f=frame.f_locals;full=f['input_text'];left,right=f['template'].split('{context}')
  prefix=left.format(type_needle_v=f['type_needle_v'],query=f['query'])
  suffix=right.format(type_needle_v=f['type_needle_v'],query=f['query'])
  assert full==prefix+f['context']+suffix
  assert suffix.startswith('\n') and not suffix.startswith('\n\n')
  # Move the fixed separator into the document so its newline BPE is independent of query words.
  document=prefix+f['context']+'\n';query=suffix[1:]
  metadata[key(full)]={'full_text_sha256':key(full),'document_text':document,'query_text':query,
    'raw_context_char_bounds':[len(prefix),len(prefix)+len(f['context'])],
    'document_query_char_boundary':len(document),'bare_question_text':query[:query.index('?')+1],
    'source_function':'generate_input_output','source_key':f['query']}
 return profile
plan={'status':'FROZEN_BEFORE_NEW_DATA_GENERATION','task':'niah_multikey_1','official_git_revision':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a',
 'model_tokenizer':'models/Qwen3-8B','replica_scope':'new official-generator cohort using local matching checkpoint/custom4000 adapter; not public paper checkpoint or sameexamples',
 'argv':args,'random_seed':42,'PYTHONHASHSEED':42,'official_max_seq_length':131072,'official_generation_reserve':128,'inference_generation_cap':48,
 'task_config':config,'template':template,'metadata_capture':'read-only Python profile return hook; official source bytes unchanged; no RNG calls',
 'prefix_contract':'single explicit model-config BOS; not in document chunks; no chat template',
 'document_contract':'official fixed task instruction + generated full context + fixed newline separator; independently tokenized',
 'query_contract':'official entire question+answerprefix; independently tokenized, no truncation; barequestion excludes answerprefix',
 'retrieval':{'selector':'iter_bm25','topk':12,'iter_hop_topk':4,'iter_rounds':0,'source':'tmp_external_baselines/comem_official/comem/selectors.py'},
 'versions':{k:importlib.metadata.version(k) for k in ['transformers','tokenizers','numpy','nltk','wonderwords','PyYAML','tenacity']}}
(O/'generation_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
sys.argv=args
sys.setprofile(profile)
try:runpy.run_path(str(SRC),run_name='__main__')
finally:sys.setprofile(None)
raw=save_root/'niah_multikey_1/validation.jsonl'
rows=[json.loads(line) for line in raw.read_text().splitlines()]
assert len(rows)==100
selected=[]
for ordinal,row in enumerate(rows):
 full=row['input']+row['answer_prefix'];m=metadata[key(full)]
 selected.append(dict(m,ordinal=ordinal))
(O/'source_boundaries.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in selected),encoding='utf-8')
assert 'torch' not in sys.modules
print(json.dumps({'status':'OFFICIAL_GENERATION_COMPLETE','rows':len(rows),'source_boundary_rows':len(selected),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'torch_imported':False}))
