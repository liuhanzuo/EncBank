"""One Slurm-assigned GPU, actual physical identity, no foreign-process mutation."""
import csv,io,os,socket,subprocess,time
from protocol import RESOURCE
def query():
 assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS') is not None,'Must run inside Slurm allocation'
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(visible)==1 and visible[0] not in ('','-1','NoDevFiles')
 fields=['index','uuid','name','pci.bus_id','pci.device_id','memory.total','memory.free','driver_version']
 argv=['nvidia-smi','--id='+visible[0],'--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']
 result=subprocess.run(argv,capture_output=True,text=True,timeout=30);assert result.returncode==0,result.stderr
 rows=list(csv.reader(io.StringIO(result.stdout)));assert len(rows)==1
 gpu=dict(zip(fields,[x.strip() for x in rows[0]]));gpu['total_bytes']=int(gpu['memory.total'])*2**20;gpu['free_bytes']=int(gpu['memory.free'])*2**20
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
 assert apps.returncode==0,apps.stderr
 entries=[r for r in csv.reader(io.StringIO(apps.stdout)) if len(r)>=4 and r[0].strip()==gpu['uuid']]
 foreign=[{'gpu_uuid':r[0].strip(),'pid':int(r[1].strip()),'process_name':r[2].strip(),'used_gpu_memory':r[3].strip()} for r in entries if int(r[1].strip())!=os.getpid()]
 return {'hostname':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_job_gpus':os.environ.get('SLURM_JOB_GPUS'),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu':gpu,'foreign_gpu_compute_processes':foreign,'actual_commands':[argv,apps.args],'allowed':not foreign and gpu['free_bytes']>=RESOURCE['minimum_free_bytes'] and gpu['total_bytes']>=RESOURCE['minimum_free_bytes']}
def require_identity(snapshot,identity):
 assert snapshot['allowed'],'GPU admission denied; no other processes are terminated'
 if identity is not None:
  for k in ('uuid','pci.bus_id','memory.total','driver_version'):assert snapshot['gpu'][k]==identity[k],('Physical GPU changed',k)
 return snapshot['gpu']
def stable_admission(identity,record):
 samples=[];begin=time.monotonic()
 while True:
  value=query();samples.append(value)
  elapsed=time.monotonic()-begin
  record(samples,elapsed)  # Preserve the exact rejected sample before the unchanged assertion.
  require_identity(value,identity)
  if elapsed>=RESOURCE['stable_idle_seconds']:return samples
  # A slow record callback can cross the deadline: obtain another fresh sample.
  time.sleep(max(0,min(5,RESOURCE['stable_idle_seconds']-(time.monotonic()-begin))))
