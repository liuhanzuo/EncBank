"""Actual original CUDA pack/GEMV checks, with CPU reference used ONLY as oracle."""
import gc,hashlib,weakref
from pathlib import Path
from protocol import save,sha

def digest_tensor(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()

def tensors_digest(state):
    return [{'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':digest_tensor(t)} for n,t in state.tensor_items()]

def cpu_unpack(torch,codes,scale,mn,bits):
    shifts=torch.arange(32//bits,dtype=torch.int64)*bits
    integer=((codes.long().unsqueeze(-1)>>shifts)&((1<<bits)-1)).flatten(-2)
    return integer,integer.double()*scale.double().repeat_interleave(32,-1)+mn.double().repeat_interleave(32,-1)

def one_case(torch,kernels,case,generator,out):
    from bridge import repeat_heads
    bits,inner,outer=case['bits'],case['inner'],case['outer']
    cpu_weight=torch.randint(-32,33,(1,8,inner,outer),generator=generator,dtype=torch.int16).to(torch.float16)/16
    cpu_weight[...,:32]=2.5  # constant group: preserve original zero-scale behavior
    cpu_input=torch.randint(-16,17,(1,32,1,inner),generator=generator,dtype=torch.int16).to(torch.float16)/32
    weight=cpu_weight.cuda();query=cpu_input.cuda()
    codes,scale,mn=kernels.pack(weight,32,bits);torch.cuda.synchronize()
    output=kernels.gemv(32,query,repeat_heads(codes,4),repeat_heads(scale,4),repeat_heads(mn,4),bits);torch.cuda.synchronize()
    c,s,z,y=(x.cpu() for x in (codes,scale,mn,output))
    raw={'cpu_weight':cpu_weight,'cpu_input':cpu_input,'actual_cuda_codes':c,'actual_cuda_scale':s,'actual_cuda_min':z,'actual_cuda_output':y}
    rawpath=out/f"{bits}bit_{case['kind']}.pt";torch.save(raw,rawpath)
    integer,dequant=cpu_unpack(torch,c,s,z,bits)
    grouped=cpu_weight.reshape(1,8,inner,outer//32,32)
    refmin=grouped.amin(-1);refscale=(grouped.amax(-1)-refmin)/((1<<bits)-1)
    quotient=((grouped-refmin[...,None])/refscale[...,None]).clamp(0,(1<<bits)-1).round()
    valid=refscale.ne(0).unsqueeze(-1).expand_as(grouped).reshape_as(cpu_weight)
    reference_codes=quotient.to(torch.int32).reshape_as(cpu_weight)
    mismatches=int((integer[valid]!=reference_codes[valid]).sum())
    constant=~valid;constant_error=float((dequant[constant]-cpu_weight.double()[constant]).abs().max())
    reference=cpu_input.double()@dequant.repeat_interleave(4,1)
    error=y.double()-reference
    record={**case,'status':'observed_pending_gates','raw_path':str(rawpath.name),'raw_sha256':sha(rawpath),'defined_code_mismatches':mismatches,'constant_group_code_comparison':'not required: original NaN-to-int conversion preserved and raw codes retained','constant_decoded_max_abs_error':constant_error,'scale_equal':torch.equal(s,refscale),'min_equal':torch.equal(z,refmin),'finite_output':bool(torch.isfinite(y).all()),'gemv_max_abs_error':float(error.abs().max()),'gemv_rms_error':float(error.square().mean().sqrt()),'allocated_packed_bytes':sum(x.untyped_storage().nbytes() for x in (codes,scale,mn))}
    save(out/f"{bits}bit_{case['kind']}.json",record)
    assert mismatches==0 and record['scale_equal'] and record['min_equal'] and constant_error==0,record
    assert record['finite_output'] and record['gemv_max_abs_error']<=.01 and record['gemv_rms_error']<=.001,record
    refs=[weakref.ref(t) for t in (weight,query,codes,scale,mn,output)]
    del weight,query,codes,scale,mn,output
    gc.collect();torch.cuda.synchronize()
    record.update(status='passed',own_cuda_tensor_objects_released=all(r() is None for r in refs))
    assert record['own_cuda_tensor_objects_released']
    save(out/f"{bits}bit_{case['kind']}.json",record)
    return record

def one_tail(torch,kernels,case,generator,out):
    from bridge import LayerState
    bits=case['bits'];entry=LayerState(bits)
    key=torch.randn((1,8,127,128),generator=generator,dtype=torch.float16).cuda()
    value=torch.randn((1,8,127,128),generator=generator,dtype=torch.float16).cuda()
    entry.prefill(key,value,kernels);entry.frozen=True;del key,value
    original=tensors_digest(entry)
    steps=[tuple(torch.randn(shape,generator=generator,dtype=torch.float16).cuda() for shape in ((1,32,1,128),(1,8,1,128),(1,8,1,128))) for _ in range(3)]
    outputs=[];requests=[]
    for index in range(2):
        request=entry.fork()
        noalias=all(t.untyped_storage().data_ptr()!=dict(entry.tensor_items())[n].untyped_storage().data_ptr() for n,t in request.tensor_items())
        actual=[]
        for query,key,value in steps:
            attention,probability=request.decode_attention(query,key,value,None,kernels,4)
            torch.cuda.synchronize();actual.append(attention.cpu())
            assert torch.isfinite(attention).all() and torch.isfinite(probability).all()
        assert request.sequence_length==130 and request.key_tail.shape[2]==2 and request.value_tail.shape[2]==128
        assert request.key_codes.shape[-1]*(32//bits)==128 and request.value_codes.shape[2]==2
        invariant=tensors_digest(entry)==original
        refs=[weakref.ref(t) for _,t in request.tensor_items()];request.release();gc.collect();torch.cuda.synchronize()
        released=all(r() is None for r in refs)
        requests.append({'index':index,'final_tokens':130,'no_entry_alias':noalias,'entry_unchanged':invariant,'request_tensors_released':released})
        save(out/f'{bits}bit_tail.json',{'status':'running','requests':requests,'entry_initial':original})
        assert noalias and invariant and released
        outputs.append(actual)
    identical=all(torch.equal(a,b) for a,b in zip(*outputs))
    torch.save({'actual_two_request_outputs':outputs},out/f'{bits}bit_tail.pt')
    refs=[weakref.ref(t) for _,t in entry.tensor_items()];entry.release()
    del steps,query,key,value,attention,probability
    gc.collect();torch.cuda.synchronize()
    released=all(r() is None for r in refs)
    record={**case,'status':'passed','requests':requests,'entry_initial':original,'repeat_outputs_bitwise_equal':identical,'entry_tensors_released':released,'raw_path':f'{bits}bit_tail.pt','raw_sha256':sha(out/f'{bits}bit_tail.pt')}
    save(out/f'{bits}bit_tail.json',record)
    assert identical and released
    return record

def run(torch,kernels,plan,out,profiler):
    assert kernels.performance_backend is True
    assert torch.cuda.current_stream()==torch.cuda.default_stream()
    generator=torch.Generator(device='cpu').manual_seed(plan['seed'])
    result={'status':'running','scope':'kernel compatibility only; no model/benchmark','cases':[],'tail_cases':[],'cpu_reference_is_oracle_only':True,'official_cuda_pack_and_gemv_used':True}
    save(out/'result.json',result)
    for case in plan['cases']:
        with profiler.phase('synthetic_pack_and_gemv',bits=case['bits'],kind=case['kind']):result['cases'].append(one_case(torch,kernels,case,generator,out))
        save(out/'result.json',result)
    for case in plan['tail_cases']:
        with profiler.phase('tail_two_requests_and_release',bits=case['bits']):result['tail_cases'].append(one_tail(torch,kernels,case,generator,out))
        save(out/'result.json',result)
    result.update(status='all_synthetic_checks_passed_parent_exit_pending',fixed_query_cost_protocol='future six-arm block still must freeze common tokenwise schedule; no profile metrics here')
    save(out/'result.json',result)
    return result
