from pathlib import Path
import os,json,sys
root=Path(__file__).resolve().parent
os.environ.update(APPWORLD_ROOT=str(root/'appworld_public'),APPWORLD_CACHE=str(root/'transport_check_cache'),IPYTHONDIR=str(root/'transport_check_ipython'),CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0,str(root))
from action_parser import parse_action,VERSION
from appworld import AppWorld
trace=json.loads((root.parent/'comem_agent_ctx_r3_20260918/run_dense/0d8a4ee_1.trace.json').read_text())
text=trace['events'][0]['model']['text'];code,kind=parse_action(text)
world=None
try:
 world=AppWorld(task_id='0d8a4ee_1',experiment_name='r4_transport_check',load_ground_truth=False,random_seed=20260917,max_interactions=1,max_api_calls_per_interaction=50,raise_on_unsafe_syntax=True,raise_on_unsafe_execution=True,raise_on_failure=True,import_utils=False,add_login_shortcut=False,allow_datetime_change=False,parse_datetimes=False,wrap_response=False,unwrap_response=False,munchify_response=False)
 assert world.task.ground_truth is None
 result=world.execute(code)
 assert 'Execution failed' not in str(result) and 'supervisor' in str(result).lower(),str(result)
 print(json.dumps(dict(status='SAVED_NATIVE_TOOL_PUBLIC_API_EXECUTION_PASS',parser_version=VERSION,action_format=kind,code=code,output=str(result),ground_truth_loaded=False,model_requests=0)))
finally:
 if world is not None:world.close()
