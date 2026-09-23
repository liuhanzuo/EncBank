"""Short logits-distillation pilot; select by validation including step zero."""
import gc,hashlib,json,random,statistics,time

def train_heads(api,run,root):
    import torch
    import torch.nn.functional as F
    capture,Provider,prefill,evaluate=[api[k] for k in ['capture','Provider','prefill','evaluate']]
    dataset,dump,status=[api[k] for k in ['dataset','dump','status']]
    prior=root.parent/'hidden_reader_pilot_20260921/h12/heads.pt'
    checkpoint=torch.load(prior,map_location='cuda',weights_only=False)
    params={l:tuple(torch.nn.Parameter(t.float()) for t in pair) for l,pair in checkpoint['weights'].items()}
    del checkpoint
    lr={'lr1e5':1e-5,'lr5e5':5e-5}[run]
    all_params=[p for pair in params.values() for p in pair]
    optimizer=torch.optim.AdamW(all_params,lr=lr,weight_decay=0.)
    def export():return {l:tuple(t.detach().to(torch.bfloat16).clone() for t in pair) for l,pair in params.items()}
    def validate(step):
        rows=[dict(id=case['id'],**evaluate(case,params)) for case in dataset['validation']]
        return dict(step=step,kl=statistics.mean(r['kl'] for r in rows),documents=rows)
    validations=[validate(0)];best=validations[0]['kl'];best_step=0;best_weights=export()
    dump('distill_validation.json',validations)
    rng=random.Random(20260921);trace=[];begin=time.perf_counter()
    step=0
    for epoch in range(3):
        cases=dataset['train'].copy();rng.shuffle(cases)
        for case in cases:
            step+=1
            bundle=capture(case['memory'])
            teacher=Provider(bundle,teacher=True)
            reference,*rest=prefill(teacher,case['continuation'][:128])
            logp=F.log_softmax(reference.float(),dim=-1).detach()
            del teacher,reference,rest
            with torch.enable_grad():
                student=Provider(bundle,params)
                actual,*rest=prefill(student,case['continuation'][:128])
                logq=F.log_softmax(actual.float(),dim=-1)
                loss=(logp.exp()*(logp-logq)).sum(-1).mean()
                assert loss.requires_grad and torch.isfinite(loss),loss
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(all_params,1.)
                assert torch.isfinite(norm) and norm>0,norm
                optimizer.step()
            trace.append(dict(step=step,epoch=epoch,id=case['id'],train_kl=float(loss.detach()),grad_norm=float(norm)))
            del bundle,student,actual,rest,logp,logq,loss,norm
            if step%8==0:
                dump('distill_trace.json',trace);status('DISTILL',step=step,total=96,train_kl=trace[-1]['train_kl'])
            if step%16==0:
                row=validate(step);validations.append(row)
                if row['kl']<best:best=row['kl'];best_step=step;best_weights=export()
                dump('distill_validation.json',validations)
                status('DISTILL_VALIDATE',step=step,validation_kl=row['kl'],best_step=best_step)
    seconds=time.perf_counter()-begin
    info=dict(lr=lr,updates=step,selected_step=best_step,initial_validation_kl=validations[0]['kl'],
        selected_validation_kl=best,seconds=seconds,initializer_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        train_documents=32,validation_documents=4,test_documents=8,loss='mean teacher-to-student logits KL, T=1',
        trainable='23 affine H12-to-KV heads only; frozen backbone, LoRA and anchor KV13 projection',
        selection='validation KL over checkpoints 0,16,...96; no test selection')
    dump('distill_training.json',info)
    del optimizer,params,all_params;gc.collect();torch.cuda.empty_cache()
    return best_weights,info
