"""Single persistent owner: first formal batch, then the remaining original batches."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
H=Path(__file__).resolve().parent
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def main():
    expected=sys.argv[1];assert sha(H/'plan.json')==expected
    os.environ['QCOMEM_JUDGE_API_KEY']=json.loads(sys.stdin.buffer.read())['key']
    assert os.environ['QCOMEM_JUDGE_API_KEY']
    state={'status':'RUNNING','pid':os.getpid(),'started_at':now(),'plan_sha256':expected,'automatic_retry':False,'steps':[]}
    save(H/'coordinator_status.json',state)
    try:
        for phase in ('first','remaining'):
            argv=[sys.executable,'-X','utf8','-B',str(H/'hold_launch.py'),'--phase',phase,'--expected-plan-sha256',expected]
            child=subprocess.Popen(argv,cwd=H)
            code=child.wait()
            state['steps'].append({'phase':phase,'pid':child.pid,'argv':argv,'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True,'finished_at':now()})
            save(H/'coordinator_status.json',state)
            if code:raise RuntimeError('Formal phase failed; preserved without retry')
        state['status']='ALL_PHASES_PARENT_WAIT0_COMPLETED'
    except BaseException as error:
        state.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__)
        raise
    finally:
        state['finished_at']=now();save(H/'coordinator_status.json',state)
if __name__=='__main__':main()
