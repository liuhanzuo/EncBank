"""Prepare a full automation update for the app tool; does not edit app config."""
import json,tomllib
from pathlib import Path
root=Path(__file__).resolve().parent
automation=Path('/srv/encbank/client/.codex/automations/encbank-followups-progress-3h/automation.toml')
old=tomllib.loads(automation.read_text(encoding='utf-8'))
launch=json.loads((root/'launch.json').read_text())
prompt=old['prompt']
start=prompt.index('Four-benchmark evaluation45014 is already running')
end=prompt.index(' Do not resubmit or duplicate completed/active shards.',start)
prompt=prompt[:start]+'''Four-benchmark evaluation45014 and dependent LoCoMo array59046 remain sequential (afterok:45014). Both were reduced to ArrayTaskThrottle=3 on September16 to reserve ONE GPU for the newly authorized Qwen3-8B CacheBlend-style LoRA experiment. Do not restore them to4 while its GPU pipeline is active. Existing four running tasks were not interrupted; the new one-GPU job waits for any one to finish before entering the reserved fourth slot. The CPU release job restores4 only AFTER the CacheBlend pipeline terminates. Never exceed FOUR GPUs overall or remove the existing array dependency without another mechanism guaranteeing the cap.'''+prompt[end:]
prompt=prompt.replace('Once all current five-benchmark runs and judging are finished and user informed, delete this obsolete heartbeat and explain why it stopped.',
 'Delete the heartbeat only after BOTH the five-benchmark/Astra work AND the independent CacheBlend-style training/evaluation/delivery are complete and the user has been informed. Explain why the monitor stopped.')
prompt+='''

NEW authorized independent CacheBlend-style baseline: local F:/Paper_Evolve/exp/encbank_cacheblend_lora_20260916, remote /srv/encbank/encbank_cacheblend_lora_20260916 on gpu-node1. Run its collect.py and inspect STATUS.json. Pipeline job PIPELINE, dependent CPU release job RELEASE, both already submitted. Do not deploy or submit duplicates. Read PROTOCOL_zh.md. CPU tests passed locally and remotely before submission; pipeline also checks actual8B forward parity initially and after training. This is a differentiable version of the prior two-bootstrap-layer fixed15% HF CacheBlend-style control, not the native engine. Cached writer and sparse repair both receive gradients; backbone frozen. Fresh suffix LoRA12-35, rank/alpha32,168modules/58,195,968parameters. Same exact PG19 64books/7,231,927token stream as principal Encbank, same initial skip8windows,4096tokens/step,seed42,teacher top64 support-normalized bidirectional KL .6/.4,AdamW1e-4/warmup50/cosine/clip1,4000steps/16.384Mprocessedtokens. Preserve original training tokenizer-BOS-else-EOS and evaluationBOS151643; actual runtime versions recorded. Matched scientific settings/budget, not equal FLOPs/hours. Never initialize from Encbank adapter or tune ratio/checkpoint on test outcomes.

After training, same pipeline generates four fresh arms on exact prior500examples (LongEval8/16/32k100each,budget16; Qasper200,budget128): CacheBlend frozen, CacheBlend own LoRA, Encbank frozen, Encbank principal own LoRA. Identical saved tokens/retrieval/order,chunk512/top12,plain prompts/greedy/scorers. 2000records total; report.py independently decodes/rescores complete support. This support is the paired KV appendix experiment, not all five main-table benchmarks. Do not conflate LongEval16 here with newer-model48. Failure JSONs and pipeline logs must be inspected; if a real implementation defect occurs, fix only the defect without changing method/budget/samples, preserve failure evidence, and resume last250-step checkpoint with restored optimizer/RNG/cursor. Before any recovery submission, inspect live jobs and current array throttles: the CPU release job may already have restored4 after failure, so reserve a fourth slot safely again.

When CacheBlend COMPLETE.json confirms2000verifiedrecords and4000trainingsteps, run collect_artifacts.py locally to collect results.csv, per_sample_scores.csv, training_curve.csv, summary.json, actual raw predictions/protocol/correctness/metadata and final adapter. Do not download optimizer or intermediate adapters. Report actual numbers and paired training gain, links to CSV/adapter, and caveat that this is CacheBlend-style. Do not fabricate results while pending or integrate unverified numbers into the paper. Confirm release job restored array throttle4 if still useful; if release failed, restore only after confirming CacheBlend GPU job is terminal. Add outcomes to memory.md and avoid duplicate notifications.
'''.replace('PIPELINE',str(launch['pipeline_job'])).replace('RELEASE',str(launch['release_job']))
payload=dict(mode='update',id=old['id'],kind='heartbeat',name='MidCache 评测、CacheBlend 训练与 Astra Judge（3小时）',
 prompt=prompt,rrule=old['rrule'],status=old['status'],targetThreadId=old['target_thread_id'])
if 'notification_policy' in old:payload['notificationPolicy']=old['notification_policy']
(root/'monitor_update.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared app-tool update')
