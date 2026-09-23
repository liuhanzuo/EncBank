"""Real tokenizer boundary checks; no model inference and no verifier contents."""
import ast,json
from pathlib import Path
from transformers import AutoTokenizer
from common import ROOT,PLAN as P,save
from layout import partition,fill_recent
for path in ROOT.rglob('*.py'):ast.parse(path.read_text())
tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
messages=[dict(role='system',content='Return valid JSON with terminal commands.'),dict(role='user',content='Process a log file safely.')]
anchor=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)
previous=[];rows=[]
for step in range(4):
    ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    chunks,query=partition(ids,anchor,P['recent_tokens'],512)
    assert chunks[:len(previous)]==previous
    assert all(len(c)==512 for c in chunks)
    assert anchor+sum(chunks,[])+query[len(anchor):]==ids
    rows.append(dict(step=step,chunks=len(chunks),query_tokens=len(query)))
    previous=chunks
    messages.extend([dict(role='assistant',content='{"commands": [{"keystrokes": "cat input.log\\n", "duration": 0.1}], "task_complete": false}'),dict(role='user',content=('event=ready 零 α delimiter </s>\n'*500))])
assert rows[0]['chunks']==0 and rows[-1]['chunks']>=12,rows
assert fill_recent([1,3],20,12)==[1,3]+list(range(10,20))
assert fill_recent([],3,12)==[0,1,2]
save(ROOT/'cpu_check.json',dict(passed=True,tokenizer_boundary_cases=rows,model_calls=0))
print(json.dumps(dict(passed=True,rows=rows)))
