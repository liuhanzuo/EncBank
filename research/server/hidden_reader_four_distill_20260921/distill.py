"""Four independent-write depths, frozen teacher, validation-selected KD heads."""
import gc,hashlib,json,random,statistics,time

def train_heads(api,config,root):
    import torch
    import torch.nn.functional as F
    capture,Provider,prefill,evaluate=[api[k] for k in ['capture','Provider','prefill','evaluate']]
    dataset,dump,status=[api[k] for k in ['dataset','dump','status']]
    prior=root.parent/'hidden_reader_localanchors_20260921/four_local/heads.pt'
    prior_summary=json.loads((prior.parent/'results/summary.json').read_text())
    initializer_sha=hashlib.sha256(prior.read_bytes()).hexdigest()
    assert initializer_sha==prior_summary['checkpoint_sha256']
    checkpoint=torch.load(prior,map_location='cuda',weights_only=False)
    params={l:tuple(torch.nn.Parameter(t.float()) for t in pair) for l,pair in checkpoint['weights'].items()}
    assert len(params)==20 and not set(params)&{12,16,20,24}
    del checkpoint
    if config['anchor_residual']:
        for l in [16,20,24]:
            params[l]=(torch.nn.Parameter(torch.zeros((2048,4096),device='cuda')),
                torch.nn.Parameter(torch.zeros(2048,device='cuda')))
    all_params=[p for pair in params.values() for p in pair]
    optimizer=torch.optim.AdamW(all_params,lr=config['lr'],weight_decay=0.)
    def export():return {l:tuple(t.detach().to(torch.bfloat16).clone() for t in pair) for l,pair in params.items()}
    def validate(step):
        rows=[dict(id=case['id'],**evaluate(case,params)) for case in dataset['validation']]
        return dict(step=step,kl=statistics.mean(r['kl'] for r in rows),documents=rows)
    validations=[validate(0)]
    assert abs(validations[0]['kl']-0.097931157797575)<1e-5,validations[0]
    best=validations[0]['kl'];best_step=0;best_weights=export()
    dump('distill_validation.json',validations)
    rng=random.Random(20260921);trace=[];begin=time.perf_counter();step=0;epoch=0
    while step<config['updates']:
        cases=dataset['train'].copy();rng.shuffle(cases)
        for case in cases:
            if step>=config['updates']:break
            step+=1
            bundle=capture(case['memory']);teacher=Provider(bundle,teacher=True)
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
                if step==1:
                    gradient_rows={str(l):[float(p.grad.norm()) for p in pair] for l,pair in params.items()}
                    assert all(all(x>0 for x in vals) for vals in gradient_rows.values())
                    dump('gradient_check.json',dict(heads=gradient_rows,all_heads_receive_gradient=True,
                        backbone_trainable_parameters=sum(p.numel() for p in api['model'].parameters() if p.requires_grad)))
                optimizer.step()
            trace.append(dict(step=step,epoch=epoch,id=case['id'],train_kl=float(loss.detach()),grad_norm=float(norm)))
            del bundle,student,actual,rest,logp,logq,loss,norm
            if step%16==0:
                dump('distill_trace.json',trace);status('DISTILL',step=step,total=config['updates'],train_kl=trace[-1]['train_kl'])
            if step%32==0 or step==config['updates']:
                row=validate(step);validations.append(row)
                if row['kl']<best:best=row['kl'];best_step=step;best_weights=export()
                dump('distill_validation.json',validations)
                status('DISTILL_VALIDATE',step=step,validation_kl=row['kl'],best_step=best_step)
        epoch+=1
    seconds=time.perf_counter()-begin
    info=dict(**config,actual_updates=step,selected_step=best_step,initial_validation_kl=validations[0]['kl'],
        selected_validation_kl=best,seconds=seconds,initializer_sha256=initializer_sha,
        train_documents=len(dataset['train']),validation_documents=len(dataset['validation']),test_documents=len(dataset['test']),
        fresh_test_documents=len(dataset.get('fresh_test',[])),loss='mean teacher-to-student logits KL, T=1',
        trainable_heads=len(params),trainable_parameters=sum(p.numel() for p in all_params),
        trainable='20 affine per-layer KV heads; optional3 zero-initialized residual heads at KV17/21/25. Frozen backbone, original LoRA, writer and anchor projections.',
        selection='validation KL over step0 and every32 steps; no test selection')
    dump('distill_training.json',info)
    del optimizer,params,all_params;gc.collect();torch.cuda.empty_cache()
    return best_weights,info
