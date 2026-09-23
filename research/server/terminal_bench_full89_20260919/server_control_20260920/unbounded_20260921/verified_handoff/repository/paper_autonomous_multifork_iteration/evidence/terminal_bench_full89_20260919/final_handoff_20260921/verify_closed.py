"""Closed-task saved-output verification; never runs a model or edits an active plan."""
from pathlib import Path
from collections import Counter
import datetime,hashlib,json
ROOT=Path('/srv/encbank/legacy_workspace');H=Path(__file__).resolve().parent;B=H.parent;RT=ROOT/'.runtime/terminal_bench_full89_20260919'
NAMES=dict(dense='dense_no_task_deadline_r6_20260921',top12='encbank_k12_no_task_deadline_r6_20260920',top48='encbank_k48_no_task_deadline_r6_20260920')
def load(p):return json.loads(p.read_text(encoding='utf8'))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
snapshot=load(H/'live_snapshot.json');arms={}
for arm,name in NAMES.items():
    D=B/name;P=load(D/'plan.json');base=load(D/'partition_final.json');retained=base['retained_normal'];old={x['task'] for x in retained};box=RT/('local_rpc_'+name)/P['arm']
    for n,d in load(D/'recovery_check.json')['source_sha256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==d,n
    reqs={}
    for f in box.glob('*.request.json'):
        q=load(f);reqs.setdefault(q['task'],[]).append((f,q))
    released={x['task_id']:x for x in snapshot['remote']['services'][arm]['status'].get('transport',{}).get('events',[]) if x['event']=='session_release'}
    rows=[];excluded=[]
    for rp in sorted((D/'execution/receipts').glob('*.json')):
        receipt=load(rp);task=receipt['task'];assert task in P['tasks'] and task not in old
        files=[Path(f) for f in receipt['result_paths']];assert len(files)<=1
        raw=load(files[0]) if files else {};exc=raw.get('exception_info') or {}
        good=receipt['exit_code']==0 and receipt.get('actual_parent_wait') and receipt.get('actual_process_handle_wait') and not receipt['transport_errors'] and raw.get('verifier_result') is not None and not exc
        if not good:
            classification='SSH_TRANSPORT_INTERRUPTION' if receipt['transport_errors'] else ('CONTEXT_CAPACITY_BOUNDARY' if exc.get('exception_type')=='ContextLengthExceededError' else 'OTHER_INCOMPLETE')
            excluded.append(dict(task=task,classification=classification,exception=exc,verifier=raw.get('verifier_result'),transport_errors=receipt['transport_errors'],result=ref(files[0]) if files else None,receipt=ref(rp)))
            continue
        f=files[0];f.resolve().relative_to((RT/'results'/name).resolve());assert raw['finished_at'] and raw['task_name'].split('/')[-1]==task
        requests=reqs.get(task,[]);assert len(requests)==receipt['requests'] and len(requests)>0 and sorted(q['step'] for _,q in requests)==list(range(len(requests)))
        answers=[];tokens=0;capped=0;session_ids=set()
        for qf,q in sorted(requests,key=lambda x:x[1]['step']):
            rid=q['request_id'];ap=box/(rid+'.response.json');bp=box/(rid+'.broker.json')
            assert not (box/(rid+'.error.json')).exists() and ap.exists() and bp.exists(),(task,rid)
            data=ap.read_bytes();a=json.loads(data);broker=load(bp);proof=broker['proof']
            assert broker['status']=='delivered' and proof['sha256']==hashlib.sha256(data).hexdigest() and proof['bytes']==len(data)
            assert a['status']=='ok' and all(a[k]==q[k] for k in ['request_id','task','task_id','step'])
            assert a['generated_tokens']==len(a['generated_ids']) and a['generated_tokens']<=P['max_new_tokens']
            assert a['prompt_tokens']==len(a['prompt_ids']) and a.get('full_history_tokens',a['prompt_tokens'])+a['generated_tokens']<=P['context_tokens']
            session_ids.add(q['task_id']);tokens+=a['generated_tokens']
            if arm!='dense':
                assert a['state_hashes_unchanged'] is True and a['configured_top_k_chunks']==P['top_k_chunks']
                assert a['configured_auto_rounds']==-(-P['top_k_chunks']//P['iter_hop_topk'])
                assert a['actual_selected_chunks']==len(a['selected_history_chunks'])<=min(a['candidate_history_chunks'],P['top_k_chunks'])
                assert len(set(a['selected_history_chunks']))==len(a['selected_history_chunks'])
                assert a['selected_history_tokens']<=512*a['actual_selected_chunks'] and a['configured_decode_batch_size']==8
                capped+=a['hit_generation_cap']
            else:capped+=a.get('finish_reason')=='length'
            answers.append(dict(request=ref(qf),response=ref(ap),broker=ref(bp)))
        releases=[released[s] for s in session_ids if s in released]
        if arm!='dense':assert len(releases)==len(session_ids),(arm,task,'await actual release event')
        rows.append(dict(task=task,path=str(f),sha256=ref(f)['sha256'],verifier=raw['verifier_result'],exception_type=None,receipt=ref(rp),
            requests=len(requests),generated_tokens=tokens,capped_responses=capped,session_release_events=releases,answers=answers))
    total=retained+rows;assert len({r['task'] for r in total})==len(total)
    previously_registered={'dense':74,'top12':14,'top48':23}[arm]
    arms[arm]=dict(root=name,normal_count=len(total),passed=sum(x['verifier']['rewards']['reward'] for x in total),
        failed=sum(x['verifier']['rewards']['reward']==0 for x in total),added_since_last_registration=len(total)-previously_registered,
        retained_normal=total,excluded=excluded,blocked_environment=base['blocked_environment'],old_pending_adjudication=base['pending_adjudication'],
        source=ref(D/'recovery_check.json'),plan=ref(D/'plan.json'),source_partition=ref(D/'partition_final.json'),
        result_scope='Per-task parent wait. Old Dense cancelled; original Encbank services still running. Not full89 score.')
out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS',live_snapshot=ref(H/'live_snapshot.json'),arms=arms,model_calls=0)
dest=H/'closed_verification.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps({k:dict(normal=v['normal_count'],passed=v['passed'],failed=v['failed'],added=v['added_since_last_registration'],excluded=dict(Counter(x['classification'] for x in v['excluded']))) for k,v in arms.items()},indent=2))
