"""Eval-only paired D0 QA quality on one externally leased remote RTX 3090.

This entry point never trains, selects documents, builds prompts or reports
inference speed. Reader histories are independent and stop at natural EOS.
Torch/model imports in the CLI happen only after the remote lease is validated.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

HERE=Path(__file__).resolve().parent
BRANCHES=("decode_v2","backend_v3")
SCHEMA="backend-quality-record-v1"


def _safe_failure(value):
    if isinstance(value,dict):
        return {key:_safe_failure(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_safe_failure(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):
        return {"nonfinite":repr(value)}
    return value


def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                      ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _write_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def select_rows(rows,mode,seed=42):
    if seed!=42 or mode not in ("smoke","full"):
        raise ValueError("This fixed pilot uses seed42 and smoke/full only")
    if len(rows)!=99 or len({row["id"] for row in rows})!=99:
        raise ValueError("Both smoke and full must start from the fixed 99-example dev set")
    ordered=sorted(rows,key=lambda row:hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    return ordered[:8] if mode=="smoke" else ordered


def _flags():
    import torch
    return {name:bool(getattr(torch.backends.cuda,name)()) for name in
        ("math_sdp_enabled","flash_sdp_enabled","mem_efficient_sdp_enabled","cudnn_sdp_enabled")}


def _tensor_identity(tensor):
    return {"shape":list(tensor.shape),"dtype":str(tensor.dtype),"device":str(tensor.device),
        "stride":list(tensor.stride()),"data_ptr":tensor.data_ptr(),"version":tensor._version}


def _input_identity(sink,memories):
    return {"sink":None if sink is None else _tensor_identity(sink),
            "memories":[_tensor_identity(memory) for memory in memories]}


def free_generate(reader,sink,memories,prompt_ids,probe_indices,*,stop_ids,max_new_tokens=128):
    """No gold-answer argument exists; all appended tokens come from this reader."""
    import torch
    if reader.training or not 1<=max_new_tokens<=128 or not stop_ids:
        raise ValueError("Generation requires eval reader, nonempty stop IDs and budget1..128")
    state=logits=None
    generated=[]
    route=None
    initial_flags=_flags()
    try:
        with torch.no_grad():
            logits,state=reader.prefill(sink,memories,prompt_ids,probe_indices=probe_indices)
            route=copy.deepcopy(state.route_stats)
            initial_route=copy.deepcopy(route)
            finish_reason="max_new_tokens"
            for index in range(max_new_tokens):
                if logits.ndim!=3 or logits.shape[0]!=1 or not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("Invalid or nonfinite generation logits")
                token=int(logits[0,-1].argmax().item())
                generated.append(token)
                if token in stop_ids:
                    finish_reason="eos"
                    break
                if index+1<max_new_tokens:
                    logits=reader.decode_step(token,state)
                if state.route_stats!=initial_route:
                    raise RuntimeError("Route changed during free generation")
            if _flags()!=initial_flags:
                raise RuntimeError("SDPA flags changed during generation")
            return {"generated_ids":generated,"finish_reason":finish_reason,
                "eos_token_id":generated[-1] if finish_reason=="eos" else None,
                "stop_token_ids":sorted(stop_ids),"max_new_tokens":max_new_tokens,
                "route_stats":route,"backend_flags":initial_flags}
    except Exception as exc:
        # Preserve partial generation without retaining GPU tensors/tracebacks.
        try:
            exc.quality_partial={"generated_ids":list(generated),"route_stats":route}
        except Exception:
            pass
        raise
    finally:
        state=logits=None


def visible_prediction(tokenizer,generation):
    ids=generation["generated_ids"]
    visible=ids[:-1] if generation["finish_reason"]=="eos" else ids
    return tokenizer.decode(visible,skip_special_tokens=True).strip()


def profile_generation(reader,sink,memories,prompt_ids,probe_indices,generation,*,out_dir,metadata=None):
    """Independent short replay with normal free history, never the main state.

    The successful free answer sets only the maximum diagnostic length. The
    profile uses its own greedy tokens and refuses any decode after EOS.
    """
    from reader_profiler import run_reader_profile
    stop_ids=set(generation["stop_token_ids"])
    steps=min(3,len(generation["generated_ids"])-1)
    def prefill():
        return reader.prefill(sink,memories,prompt_ids,probe_indices=probe_indices)
    def decode(token,state):
        if token in stop_ids:
            raise RuntimeError("Independent profile reached EOS before its expected prefix; refuse post-EOS decode")
        return reader.decode_step(token,state)
    receipt=run_reader_profile(prefill,decode,out_dir,decode_steps=steps,
        device=str(reader.device),metadata={**(metadata or {}),
            "scope":"Independent free-greedy first-example backend probe, stops no later than natural EOS; not quality or performance",
            "main_generated_tokens":len(generation["generated_ids"]),"stop_token_ids":sorted(stop_ids)})
    expected=generation["generated_ids"][:steps+1]
    diagnostic={"status":receipt["status"],"profile":receipt,
        "normal_generation_prefix":expected,"profile_generated_ids":receipt["generated_ids"],
        "prefix_equal":receipt["generated_ids"]==expected,
        "formal_inference_timing":False,"formal_inference_memory":False}
    _write_json(Path(out_dir)/"quality_profile_receipt.json",diagnostic)
    if not diagnostic["prefix_equal"]:
        raise RuntimeError("Profile greedy prefix differs from unprofiled free generation; evidence retained")
    return diagnostic


def source_hashes():
    names=("evaluate_backend_quality.py","backend_quality_results.py","reader_profiler.py",
           "optimized_sparse_reader.py","backend_sparse_reader.py","sparse_reader.py",
           "train_sparse.py","prepare_data.py","remote_gpu_guard.py","remote_sparse_queue.py")
    paths={name:HERE/name for name in names}
    paths.update({"evaluate_sft.py":HERE.parent/"beacon_encbank_20260909"/"evaluate_sft.py",
        "train_8b_baseline.py":HERE.parent/"encbank_v2_benchmarks_20260908"/"train_8b_baseline.py",
        "Encbank/encbank/model.py":HERE.parents[1]/"Encbank"/"encbank"/"model.py"})
    return {name:digest(path) for name,path in paths.items()}


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("model","init-adapter","train","dev","out","lease"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=("smoke","full"),default="smoke")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max-new-tokens",type=int,default=128)
    return parser.parse_args()


def _validate_platform(args):
    if sys.platform!="linux":
        raise RuntimeError("Quality runner is remote Linux only; never local5090")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible or len(visible.split(","))!=1:
        raise RuntimeError("Externally lease exactly one explicit CUDA_VISIBLE_DEVICES GPU")
    if args.seed!=42 or args.max_new_tokens!=128:
        raise ValueError("Fixed pilot requires seed42 and natural EOS up to128 tokens")


def run(args,lease_receipt):
    """Called only after root's external GPU lease validation; no optimizer exists."""
    import torch
    import transformers
    # The existing loading/scoring modules are reused only after lease admission.
    sys.path.insert(0,str(HERE))
    from train_sparse import validate_prepared
    from prepare_data import read_rows
    from train_8b_baseline import attach_lora,restore_flat
    from evaluate_sft import eos_token_ids,token_f1,exact_match
    from encbank.model import Encbank
    from backend_quality_results import aggregate_paired
    from optimized_sparse_reader import OptimizedSparseEncbankReader
    from backend_sparse_reader import BackendAlignedSparseEncbankReader
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.integrations.sdpa_attention as sdpa
    from remote_gpu_guard import validate_worker_lease

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    run_id=uuid.uuid4().hex
    records=[]
    profile_receipts={}
    active_row=None
    active_branch=None
    ordered_ids=recipe_sha=source=None
    model=readers=encbank=sink=memories=None
    metadata={"run_id":run_id,"status":"starting","formal_inference_timing":False,
              "formal_inference_memory":False,"lease":lease_receipt}
    _write_json(out/"status.json",metadata)
    try:
        train,dev=read_rows(args.train),read_rows(args.dev)
        validate_prepared(train,dev)
        ordered=select_rows(dev,args.mode,args.seed)
        if any(not row["references"] or not all(isinstance(value,str) for value in row["references"])
               or len(row["selected_chunk_indices"])!=len(row["document_chunks"]) for row in ordered):
            raise ValueError("Prepared references/chunk provenance are invalid")
        ordered_ids=[row["id"] for row in ordered]
        source=source_hashes()
        recipe={"model":str(args.model.resolve()),"init_adapter_sha256":digest(args.init_adapter),
            "train_sha256":digest(args.train),"dev_sha256":digest(args.dev),"mode":args.mode,
            "seed":42,"max_new_tokens":128,"j":12,"m":16,"probe_mode":"dense","retain_ratio":1.,
            "rank":32,"alpha":32.,"ordered_ids":ordered_ids,"branches":list(BRANCHES),
            "decoding":"independent-free-greedy-natural-eos","teacher_forced_ce":False}
        recipe_sha=_canonical_hash(recipe)
        metadata.update(status="loading",recipe=recipe,recipe_sha256=recipe_sha,
                        source_sha256=source,ordered_ids=ordered_ids)
        _write_json(out/"metadata.json",metadata)
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=16:
            torch.set_num_interop_threads(16)
        after_import_admission=validate_worker_lease(args.lease,require_idle=True)
        metadata["admissions"]={"before_torch_import":lease_receipt,
                                "after_import_before_cuda":after_import_admission}
        torch.manual_seed(42)
        device=torch.device("cuda:0")
        torch.cuda.set_device(device)
        if "3090" not in torch.cuda.get_device_name(device) or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This first quality pilot requires the leased RTX3090 with BF16 support")
        # Exactly the existing writer recipe, common to both custom readers.
        sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
        validate_worker_lease(args.lease,require_idle=False)
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
            attn_implementation="sdpa",local_files_only=True).to(device).eval()
        validate_worker_lease(args.lease,require_idle=False)
        if (model.config.model_type!="qwen3" or model.config.attention_dropout!=0.
                or model.config.num_hidden_layers!=36 or model.config.num_key_value_heads!=8):
            raise ValueError("Expected Qwen3-8B,36 layers,8 KV heads,zero dropout")
        longest=max(sum(map(len,r["document_chunks"]))+len(r["prompt_ids"])+128+1 for r in ordered)
        if longest>model.config.max_position_embeddings:
            raise ValueError("Prepared input/generation exceeds model window; no truncation")
        modules=attach_lora(model,12,32,32.,torch.float32)
        saved=torch.load(args.init_adapter,map_location="cpu",weights_only=False)
        if any(saved.get(key)!=value for key,value in {"j":12,"rank":32,"alpha":32.}.items()):
            raise ValueError("Strong adapter recipe differs from D0 plan")
        restore_flat(modules,saved["named"])
        del saved
        if any(module.A.dtype!=torch.float32 or module.B.dtype!=torch.float32 for module in modules.values()):
            raise RuntimeError("LoRA masters must remain FP32")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encbank=Encbank(model,resume_j=12)
        kwargs=dict(fusion_layer=16,retain_ratio=1.,probe_mode="dense",gradient_checkpointing=False)
        readers={"decode_v2":OptimizedSparseEncbankReader(encbank,**kwargs).eval(),
                 "backend_v3":BackendAlignedSparseEncbankReader(encbank,**kwargs).eval()}
        stops=eos_token_ids(tokenizer,model)
        sink_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if sink_id is None or not stops:
            raise ValueError("Missing sink or stopping token")
        baseline_flags=_flags()
        metadata.update(status="running",recipe=recipe,recipe_sha256=recipe_sha,source_sha256=source,
            ordered_ids=ordered_ids,torch=str(torch.__version__),transformers=transformers.__version__,
            cuda_runtime=torch.version.cuda,host=platform.node(),gpu=torch.cuda.get_device_name(device),
            physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],backend_flags=baseline_flags,
            backbone_dtype=str(model.model.embed_tokens.weight.dtype),lora_master_dtype="torch.float32",
            autocast_dtype="torch.bfloat16",stop_token_ids=sorted(stops),
            metric_protocol="max-reference normalized token F1/EM,0-to-1; held-out Qasper-train pilot,not official test",
            dispatch_scope="Short separate profile of first selected question only;3090 does not establish5090 dispatch/quality")
        _write_json(out/"metadata.json",metadata)
        with (out/"records.jsonl").open("x",encoding="utf-8",buffering=1) as stream, \
             torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            sink=encbank.write_chunk([int(sink_id)]).detach()
            for ordinal,row in enumerate(ordered):
                validate_worker_lease(args.lease,require_idle=False)
                active_row=row
                memories=[encbank.write_chunk(chunk).detach() for chunk in row["document_chunks"]]
                shared_before=_input_identity(sink,memories)
                for branch in BRANCHES:
                    active_branch=branch
                    common={"schema":SCHEMA,"run_id":run_id,"recipe_sha256":recipe_sha,"source_sha256":source,
                        "ordinal":ordinal,"id":row["id"],"document_id":row["document_id"],"source":row["source"],
                        "branch":branch,"references":row["references"],"selected_chunk_indices":row["selected_chunk_indices"],
                        "prompt_ids":row["prompt_ids"],"probe_indices":row["probe_indices"],
                        "candidate_tokens":sum(map(len,row["document_chunks"])),"max_new_tokens":128,
                        "formal_inference_timing":False,"formal_inference_memory":False}
                    if "question" in row:
                        common["question"]=row["question"]
                    try:
                        if _flags()!=baseline_flags:
                            raise RuntimeError("SDPA flags differ between readers")
                        generated=free_generate(readers[branch],sink,memories,row["prompt_ids"],row["probe_indices"],
                            stop_ids=stops,max_new_tokens=128)
                        if _input_identity(sink,memories)!=shared_before:
                            raise RuntimeError("Shared writer h_j mutated by a reader")
                        prediction=visible_prediction(tokenizer,generated)
                        record={**common,**generated,"status":"complete","prediction":prediction,
                            "token_f1":max(token_f1(prediction,answer) for answer in row["references"]),
                            "exact_match":max(exact_match(prediction,answer) for answer in row["references"]),
                            "shared_writer_hj_unchanged":True}
                        # Reject nonfinite route/score metadata before treating
                        # the record as complete; preserve it safely on failure.
                        json.dumps(record,allow_nan=False)
                    except Exception as exc:
                        record=_safe_failure({**common,"status":"failed","error":{"type":type(exc).__name__,"message":str(exc),
                            "traceback":traceback.format_exc()},**getattr(exc,"quality_partial",{})})
                        records.append(record)
                        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        raise
                    records.append(record)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    _write_json(out/"status.json",{"status":"running","run_id":run_id,"records_completed":len(records),
                        "target_records":2*len(ordered),"id":row["id"],"branch":branch,
                        "formal_inference_timing":False,"updated_utc":datetime.now(timezone.utc).isoformat()})
                    if ordinal==0:
                        profile_receipts[branch]=profile_generation(readers[branch],sink,memories,row["prompt_ids"],
                            row["probe_indices"],generated,out_dir=out/"profiles"/branch,
                            metadata={"run_id":run_id,"id":row["id"],"branch":branch,"recipe_sha256":recipe_sha})
                        if not profile_receipts[branch]["profile"].get("backend_operator_evidence"):
                            raise RuntimeError("First-question profile did not capture SDPA operator evidence")
                        if _input_identity(sink,memories)!=shared_before or _flags()!=baseline_flags:
                            raise RuntimeError("Independent profile changed shared h_j or backend flags")
                    gc.collect()
                memories=None
                active_row=active_branch=None
            summary=aggregate_paired(records,ordered_ids,run_id=run_id,recipe_sha256=recipe_sha,source_sha256=source)
            summary["profiles"]=profile_receipts
            _write_json(out/"summary.json",summary)
            if summary.get("status")!="complete":
                raise RuntimeError("Paired completeness/identity summary failed")
        _write_json(out/"status.json",{"status":"complete","run_id":run_id,"records_completed":len(records),
            "target_records":len(ordered)*2,"formal_inference_timing":False,
            "updated_utc":datetime.now(timezone.utc).isoformat()})
        return summary
    except Exception as exc:
        failure={"status":"failed","run_id":run_id,"records_completed":len(records),
            "active_id":None if active_row is None else active_row["id"],"active_branch":active_branch,
            "error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()},
            "formal_inference_timing":False,"formal_inference_memory":False,
            "updated_utc":datetime.now(timezone.utc).isoformat()}
        if ordered_ids is not None and recipe_sha is not None and source is not None:
            try:
                partial=aggregate_paired(records,ordered_ids,run_id=run_id,
                    recipe_sha256=recipe_sha,source_sha256=source)
                partial["profiles"]=profile_receipts
                partial["outer_execution_status"]="failed"
                _write_json(out/"partial_summary.json",partial)
                failure["partial_summary_path"]=str(out/"partial_summary.json")
            except Exception as summary_error:
                failure["partial_summary_error"]={"type":type(summary_error).__name__,"message":str(summary_error)}
        _write_json(out/"failure.json",failure)
        _write_json(out/"status.json",failure)
        raise
    finally:
        memories=sink=readers=encbank=model=None
        gc.collect()


def main():
    args=parse_args()
    _validate_platform(args)
    # Root supplies this independent guard. No fallback permits direct loading.
    from remote_gpu_guard import validate_worker_lease
    lease_receipt=validate_worker_lease(args.lease,require_idle=True)
    run(args,lease_receipt)


if __name__=="__main__":
    main()
