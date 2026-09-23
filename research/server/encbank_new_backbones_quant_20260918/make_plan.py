"""Commit explicit experiment scope and CPU scheduling dependency graph."""
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
MODELS=['Qwen3.5-9B','Qwen3.8-27B']

def main():
    tasks=[]
    for i,name in enumerate(MODELS):
        tasks.append(dict(id=f'quant-cell-m{i}',model=name,kind='evaluation',depends=[],
            command=['evaluate_quant.py','--mode','quant-cell','--model-index',str(i)],
            complete=f'results/quant-cell/{name}/cell/complete.json'))
    for i,name in enumerate(MODELS):
        tasks.append(dict(id=f'train-r128-m{i}',model=name,kind='training',depends=[],
            command=['train_large.py','--model-index',str(i)],complete=f'training/{name}/complete.json',
            prestart_duration_hours=[10,14] if i==0 else [17,24]))
    for i,name in enumerate(MODELS):
        for shard in range(4):
            tasks.append(dict(id=f'quant-full-m{i}-s{shard}',model=name,kind='evaluation',depends=[f'quant-cell-m{i}'],
                command=['evaluate_quant.py','--mode','quant-full','--model-index',str(i),'--shard',str(shard)],
                complete=f'results/quant-full/{name}/shard{shard}/complete.json'))
    for mode in ('large-mid','large-final'):
        for i,name in enumerate(MODELS):
            for shard in range(4):
                tasks.append(dict(id=f'{mode}-m{i}-s{shard}',model=name,kind='evaluation',depends=[f'train-r128-m{i}'],
                    command=['evaluate_quant.py','--mode',mode,'--model-index',str(i),'--shard',str(shard)],
                    complete=f'results/{mode}/{name}/shard{shard}/complete.json'))
    plan=dict(id='newbackbones-hprecision-and-r128-20260918',remote_root=REMOTE,
        scope='Five existing benchmark cohorts. Same native prompt-pack fixtures; no independent-document claim.',
        quantization=dict(bits=[16,8,4],group_size=64,adapter='existing rank32 alpha32 final4000',
             full_source_store=True,H16_reuse='Only after 100 matched formal examples per model pass exact H and generated-ID parity.',
             new_predictions=29144),
        training=dict(rank=128,alpha=128,steps=8000,checkpoint4000='LongEval diagnostic',
             final8000='all five benchmarks',unchanged='j, PG19 corpus/order, window4096, chunk512, teacher, KL, optimizer/lr/warmup/seed',
             schedule='cosine horizon8000 (step4000 is midpoint; not original4000-step cosine endpoint)',new_predictions=15472),
        judging=dict(model='gpt-6-astra',reasoning='low',protocol='midcache-locomo-gpt6-astra-v1',
             reuse='exact question/gold/prediction/category/status/protocol cache',report=['C1-4','C5','full1986']),
        resources=dict(max_running_plus_pending_GPUs=4,gpus_per_job=1,partition='gpu',gres='gpu:nvidia_l20d:1',
             shared_preexisting_jobs=['102932','102933','102934','102935'],
             counted_name_prefixes=['qencbank-','qcm-','midcache-','encbank-','encbank-'],
             note='Do not cancel existing AppWorld requests or unrelated jobs; no offload.'),tasks=tasks)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    names=[p.name for p in ROOT.iterdir() if p.suffix in ('.py','.sh','.json') and p.name not in
           ('package_manifest.json','cpu_checks.json','preflight_progress.json','verified_inputs.json','status.json')]
    files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    (ROOT/'package_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(f'{len(tasks)} serially admitted singleton jobs; 4 shared GPU requests maximum.')

if __name__=='__main__':main()
