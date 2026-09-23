"""EXIT-trap receipt; final Slurm accounting must still be checked externally."""
import os,socket,sys,datetime
from protocol import HERE,ROOT,read,save,sha,local
plan=read(HERE/'plan.json');out=local(plan['batch_output'])
save(out/'shell_exit.json',{'actual_shell_exit_code':int(sys.argv[1]),'shell_exit_trap_observed':True,'slurm_controller_exit_verified':False,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'node':socket.gethostname(),'plan_sha256':sha(HERE/'plan.json'),'finished_at':datetime.datetime.now().astimezone().isoformat(),'no_service_restore_obligation':'No local Ollama/service was changed in this remote allocation.'})
