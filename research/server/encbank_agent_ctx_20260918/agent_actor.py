"""Real closed-loop AppWorld CPU actor; the GPU worker cannot access evaluation gold."""
import argparse, json, os, re, time, traceback
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
R=Path(a.run).resolve(); R.relative_to(H.resolve())
plan=json.loads((H/'agent_plan.json').read_text()); box=R/'mailbox'
os.environ.update(APPWORLD_ROOT=str(H/'appworld_public'), APPWORLD_CACHE=str(R/'app_cache'),IPYTHONDIR=str(R/'ipython'),CUDA_VISIBLE_DEVICES='-1')
from appworld import AppWorld
import freezegun.api as real_clock
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
seq=0; reports=[]
for task in plan['task_ids']:
    experiment=R.name+'_'+task
    spec=json.loads((H/f'appworld_public/data/tasks/{task}/specs.json').read_text())
    messages=[dict(role='system',content=(H/'agent_prompt.txt').read_text()),dict(role='user',content='Task and supervisor context:\n'+json.dumps(spec,ensure_ascii=False))]
    events=[];started=real_clock.real_perf_counter();termination='step_limit';world=None
    try:
        world=AppWorld(task_id=task,experiment_name=experiment,load_ground_truth=False,
            random_seed=plan['seed'],max_interactions=plan['max_steps'],max_api_calls_per_interaction=plan['max_api_calls_per_step'],
            raise_on_unsafe_syntax=True,raise_on_unsafe_execution=True,raise_on_failure=True,
            import_utils=False,add_login_shortcut=False,allow_datetime_change=False,
            parse_datetimes=False,wrap_response=False,unwrap_response=False,munchify_response=False)
        assert world.task.ground_truth is None
        for step in range(plan['max_steps']):
            seq+=1;key=f'{seq:05d}'
            save(box/(key+'.request.json'),dict(task_id=task,step=step,messages=messages))
            deadline=real_clock.real_monotonic()+900
            while not (box/(key+'.response.json')).exists():
                if (R/'worker_failure.json').exists(): raise RuntimeError('GPU worker failed; see separate record')
                if real_clock.real_monotonic()>deadline: raise TimeoutError('model response timeout')
                time.sleep(.2)
            response=json.loads((box/(key+'.response.json')).read_text())
            if response['status']!='ok': termination=response['status'];break
            text=response['text'];blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,flags=re.S)
            code='\n'.join(blocks) if blocks else text.strip()
            messages.append(dict(role='assistant',content=text))
            t=real_clock.real_perf_counter()
            try: output=world.execute(code)
            except Exception as exc: output='Execution error: '+type(exc).__name__+': '+str(exc)
            tool_ms=1000*(real_clock.real_perf_counter()-t)
            messages.append(dict(role='user',content='Execution output:\n'+str(output)))
            events.append(dict(step=step,request=key,code=code,output=str(output),tool_ms=tool_ms,model=response))
            save(R/(task+'.trace.json'),dict(task_id=task,events=events,messages=messages))
            save(R/'actor_progress.json',dict(task_id=task,step=step+1,completed_tasks=len(reports),status='running'))
            if world.task_completed(): termination='complete_task_called';break
        report=dict(task_id=task,experiment_name=experiment,termination=termination,steps=len(events),wall_seconds=real_clock.real_perf_counter()-started,
            output_db_directory=str(world.output_db_home_path_on_disk),task_success=None)
    except BaseException as exc:
        save(R/'actor_failure.json',dict(task_id=task,type=type(exc).__name__,traceback=traceback.format_exc()))
        raise
    finally:
        if world is not None: world.close()
    reports.append(report);save(R/'actor_results.json',dict(status='running',rows=reports))
save(R/'actor_results.json',dict(status='completed',rows=reports,official_evaluation_pending=True))
save(box/'stop.json',dict(reason='all episodes closed'))
