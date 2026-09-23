"""No-network regression check for explicit timeouts and persistent retry budget."""
import ast,json,tempfile,types
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    tree=ast.parse((root/'judge_watch.py').read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('retryable','classify')]
    namespace=dict(Path=Path,json=json,time=types.SimpleNamespace(sleep=lambda _:None),PROMPT='{question}')
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<retry-test>','exec'),namespace)
    failure=dict(ok=False,returncode=None,turn_completed=False,errors=[],answer='')
    success=dict(ok=True,returncode=0,turn_completed=True,errors=[],answer='CORRECT')
    assert namespace['retryable'](failure)
    assert not namespace['retryable']({**failure,'errors':[dict(message='401 Unauthorized')]})
    assert not namespace['retryable']({**failure,'returncode':1,'errors':[dict(message='invalid request')]})
    calls=[]
    def write(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r))
    def call(prompt,out,timeout):
        calls.append((prompt,timeout));write(out/'result.json',failure);return failure
    namespace.update(write_json=write,call=call)
    with tempfile.TemporaryDirectory(dir=root,prefix='retry-check-') as tmp:
        p=Path(tmp);entry=p/'digest'
        write(entry/'attempt-old/result.json',failure)
        namespace['classify'](dict(question='fixed'),entry/'attempt-new')
        assert len(calls)==2 and all(timeout==180 for _,timeout in calls)
        namespace['classify'](dict(question='fixed'),entry/'attempt-another')
        assert len(calls)==2,'Exceeded three total attempts across restarts'
        entry2=p/'successful';write(entry2/'attempt-old/result.json',success)
        assert namespace['classify'](dict(question='fixed'),entry2/'attempt-new')==success
        assert len(calls)==2,'Repeated a successful saved judgment'
    print('Passed: timeouts recognized; auth failures stopped; three total attempts across restarts; saved successes reused.')

if __name__=='__main__':main()
