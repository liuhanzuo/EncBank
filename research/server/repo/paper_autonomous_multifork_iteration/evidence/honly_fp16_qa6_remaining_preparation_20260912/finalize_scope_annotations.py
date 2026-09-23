"""Correct inherited template labels before handoff; preserve prior candidate snapshots."""
import ast,datetime,difflib,json,os,shlex,subprocess,sys
from pathlib import Path
from assemble import HERE,ROOT,REMOTE,CASES,read,sha,bind
def now():return datetime.datetime.now().astimezone().isoformat()
def put(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def text(p,s):
    if p.suffix=='.py':ast.parse(s)
    p.write_text(s,encoding='utf-8',newline='\n')
def main():
    final=[]
    checker=HERE/'check_cell_v2.py';assert not checker.exists()
    text(checker,(HERE/'check_cell.py').read_text('utf-8').replace('focused_checks_attempt1/report.json','focused_checks_attempt2/report.json'))
    for ds in CASES:
        H=HERE/ds;P=H/'package';hist=H/'history_pre_scope_annotation_correction';assert not hist.exists();hist.mkdir()
        names=['package/analyze.py','package/protocol.py','package/run_quality.py','package/resource_worker.py','package/plan.json','package/upload_manifest.json','package/batch.sbatch','remote_commands.json','readiness.json','source_delta.json','README.md']
        for name in names:
            dest=hist/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((H/name).read_bytes())
        original=read(P/'plan.json');changes=[]
        descriptions={'analyze.py':'Saved-output official LongBench QA F1; require all six successful native exits first.', 'protocol.py':'Six-method LongBench QA native Slurm bindings; check-only imports no model framework.', 'run_quality.py':'LongBench grouped immutable entries and full-query natural QA; no model import at module load.', 'resource_worker.py':'Qualified single-GPU resource envelope; six FP16 QA methods and natural timing.'}
        for name,description in descriptions.items():
            path=P/name;before=path.read_text('utf-8');after=before
            end=after.index('"""',3)+3;after='"""'+description+'"""'+after[end:]
            after=after.replace(' unscaled extrapolation, official LongBench',' unchanged within-window inputs, official LongBench')
            if before!=after:
                text(path,after);changes.append({'old':bind(hist/'package'/name),'new':bind(path),'only_docstring_or_scope_annotation':True})
                text(H/'source_diffs'/(name+'.scope_annotation.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(hist/'package'/name),tofile=str(path))))
        plan=read(P/'plan.json');plan['frozen_at']=now()
        for name in descriptions:plan['source_sha256'][(P/name).relative_to(ROOT).as_posix()]=sha(P/name)
        plan['input_regime']='All bound full prompt-plus-cap budgets fit original max_position_embeddings40960. No extrapolation, truncation, YaRN or model configuration change is claimed.'
        plan['scope_annotation_correction']={'prior_unsubmitted_candidate_plan':bind(hist/'package/plan.json'),'reason':'Inherited template prose incorrectly called these within-window cohorts extrapolation and some docstrings called QA VT. Only those annotations corrected; original scientific execution code and fixtures unchanged.','source_changes':changes}
        put(P/'plan.json',plan)
        text(P/'batch.sbatch',(P/'batch.sbatch').read_text('utf-8').replace(sha(hist/'package/plan.json'),sha(P/'plan.json')))
        m=read(P/'upload_manifest.json');paths={ROOT[x['relative_path']] if False else ROOT/x['relative_path'] for x in m['files']}
        paths.add(hist/'package/plan.json');paths.update(hist/'package'/n for n in descriptions);paths.add(Path(__file__))
        m.update(plan_sha256=sha(P/'plan.json'),files=[{'relative_path':x.relative_to(ROOT).as_posix(),'local_path':str(x),'remote_path':REMOTE+'/'+x.relative_to(ROOT).as_posix(),'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(paths)])
        put(P/'upload_manifest.json',m)
        cmds=read(H/'remote_commands.json');cmds['plan_sha256']=sha(P/'plan.json');cmds['upload_manifest']['sha256']=sha(P/'upload_manifest.json');cmds['roots_and_staged_check_command']='QENCBANK_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',cmds['remote_package']+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')]);put(H/'remote_commands.json',cmds)
        delta=read(H/'source_delta.json');delta.update(status='FINAL_SCOPE_ANNOTATIONS_CORRECTED_CPU_RECHECK_PENDING',plan=bind(P/'plan.json'),manifest=bind(P/'upload_manifest.json'),scope_annotation_correction=plan['scope_annotation_correction']);put(H/'source_delta.json',delta)
        out=H/'focused_checks_attempt2';assert not out.exists();out.mkdir();argv=[sys.executable,'-X','utf8','-B',str(checker),'--dataset',ds];start=now()
        with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=so,stderr=se);code=child.wait()
        receipt={'argv':argv,'holding_parent_pid':os.getpid(),'child_pid':child.pid,'started_at':start,'finished_at':now(),'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True,'script':bind(checker),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log')};put(out/'execution_receipt.json',receipt);assert code==0,(ds,code)
        ready=read(H/'readiness.json');ready.update(plan=bind(P/'plan.json'),manifest=bind(P/'upload_manifest.json'),source_delta=bind(H/'source_delta.json'),focused_report=bind(out/'report.json'),actual_focused_exit=bind(out/'execution_receipt.json'),remote_commands=bind(H/'remote_commands.json'),source_sha256=plan['source_sha256'],scope_annotation_correction=plan['scope_annotation_correction'],prior_actual_focused_exit=bind(H/'focused_checks_attempt1/execution_receipt.json'));put(H/'readiness.json',ready)
        desc=(H/'README.md').read_text('utf-8').replace(sha(hist/'package/plan.json'),sha(P/'plan.json')).replace(sha(hist/'package/upload_manifest.json'),sha(P/'upload_manifest.json'));text(H/'README.md',desc+'\nInherited VT/extrapolation template annotations were corrected before handoff. Final actual held CPU check attempt2 exited 0; the prior unsubmitted candidate metadata is retained in history.\n')
        final.append({'dataset':ds,'status':ready['status'],'readiness':bind(H/'readiness.json'),'plan':ready['plan'],'manifest':ready['manifest'],'N':ready['N'],'D':ready['D'],'cap':ready['cap'],'actual_focused_exit':ready['actual_focused_exit'],'exact_future_held_CLI':ready['exact_future_held_CLI']})
    oldroll=HERE/'preparation_rollup.json';(HERE/'preparation_rollup_initial.json').write_bytes(oldroll.read_bytes())
    put(oldroll,{'status':'ALL_FOUR_CPU_READY_NOT_SUBMITTED','at':now(),'cells':final,'MFQA_FP16_already_complete_do_not_rerun':True,'Narrative_FP16_job25168_active_do_not_duplicate':True,'no_remote_actions':True,'final_scope_annotations_corrected':True})
    print(json.dumps(final))
if __name__=='__main__':main()
