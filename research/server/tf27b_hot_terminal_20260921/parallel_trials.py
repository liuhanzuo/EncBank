import concurrent.futures,json,sys,time
import controller
from common import ROOT,PLAN as P,save,verify_sources,sha
verify_sources();phase=sys.argv[1];controller.QUALIFY=phase=='qualify'
if controller.QUALIFY:
    for path,expected in json.loads((ROOT/'task_manifest.json').read_text()).items():assert sha(path)==expected,path
tasks=P['tasks'][:1] if controller.QUALIFY else P['tasks']
def one(task):
    arm='qualify' if controller.QUALIFY else P['arm']
    try:
        code=controller.trial(task,arm)
        return dict(task=task,arm=arm,exit_code=code)
    finally:
        if not controller.QUALIFY:save(ROOT/'mailbox'/task/'release.json',dict(epoch=time.time(),
            trial_parent_wait_finished=(ROOT/'jobs'/('wait-'+task+'--'+arm+'.json')).exists()))
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=1 if controller.QUALIFY else P['task_concurrency']) as pool:
    jobs=[pool.submit(one,t) for t in tasks]
    for future in concurrent.futures.as_completed(jobs):
        rows.append(future.result());save(ROOT/'jobs'/(phase+'_progress.json'),rows)
save(ROOT/'jobs'/(phase+'_complete.json'),dict(rows=rows,passed=all(x['exit_code']==0 for x in rows)))
assert all(x['exit_code']==0 for x in rows),rows
