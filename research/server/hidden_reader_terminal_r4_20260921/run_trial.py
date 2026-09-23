"""One actual Harbor Trial per process, including its original verifier."""
import asyncio,json,os,sys,time,traceback
import harbor_unbounded
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
from common import ROOT,PLAN,save,verify_sources
async def main():
    task,arm=sys.argv[1:3];qualify=arm=='qualify';verify_sources()
    os.environ.update(BAND_TASK=task,BAND_ARM=arm)
    directory=ROOT/('qualification' if qualify else 'results')/(task+'--'+arm)
    config=dict(task={'path':str(__import__('pathlib').Path(PLAN['task_root'])/task)},
        trial_name=task+'__band_'+arm,trials_dir=str(directory),install_only=qualify,
        agent={'import_path':'tb_agent:JointBandTerminus','model_name':'Qwen3-8B/COMem-'+arm,
            'kwargs':{'enable_summarize':False,'proactive_summarization_threshold':0,'interleaved_thinking':False,
                'collect_rollout_details':True,'store_all_messages':True,'record_terminal_session':False}},
        environment=json.loads((ROOT/'environment_template.json').read_text()),verifier={})
    save(ROOT/'configs'/(task+'--'+arm+'.json'),config)
    began=time.time();trial=await Trial.create(TrialConfig.model_validate(config));result=await trial.run()
    save(directory/'actual_result.json',result.model_dump(mode='json'))
    env=trial.agent_environment
    closure=env._root/'closure.json' if hasattr(env,'_root') else None
    save(directory/'execution_receipt.json',dict(start_epoch=began,end_epoch=time.time(),elapsed_seconds=time.time()-began,
        environment_root=str(env._root) if hasattr(env,'_root') else None,
        closure=json.loads(closure.read_text()) if closure and closure.exists() else None,
        exception=result.exception_info.model_dump(mode='json') if result.exception_info else None,
        verifier=result.verifier_result.model_dump(mode='json') if result.verifier_result else None,
        install_only=qualify,model_calls=len(list((ROOT/'mailbox'/task).glob(arm+'_*.request.json')))))
    if result.exception_info:raise RuntimeError(result.exception_info.exception_type+': '+result.exception_info.exception_message)
if __name__=='__main__':asyncio.run(main())
