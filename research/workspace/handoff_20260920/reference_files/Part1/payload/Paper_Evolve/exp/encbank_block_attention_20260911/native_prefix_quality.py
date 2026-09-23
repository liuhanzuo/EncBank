"""Group-level native Encbank/prefix free-QA checks; no loading, queue or timers.

The caller owns device admission, preparation, group selection and persistence.
Generation consumes prepared prompt/probe IDs and shared prewritten h_j only;
references are used after generation for scoring. No answer history is shared.
"""
from __future__ import annotations
import copy
import weakref

from backend_quality_results import score_prediction
from evaluate_backend_quality import _flags,_input_identity,visible_prediction


BRANCHES=("native_cold","native_prefix")
PACK_FIELDS=("source","document_id","selected_chunk_indices","document_chunks")
ROW_FIELDS=("id","document_id","source","references","selected_chunk_indices",
            "document_chunks","prompt_ids","probe_indices","question")


def _require(value,message):
    if not value:
        raise RuntimeError(message)


def _counts(reader):
    return {"build_count":reader.build_count,"hit_count":reader.hit_count,"miss_count":reader.miss_count}


def _prefix_signature(prefix):
    if prefix is None:
        return None
    prefix.assert_intact()
    return (id(prefix),prefix.tensor_signatures)


def _check_prefix_signature(prefix,expected):
    _require(_prefix_signature(prefix)==expected,"Shared prefix was changed or replaced by a request")


def _audit_state(reader,state,is_prefix):
    """Metadata/versions only; never clone or export complete document KV."""
    cm=reader.encbank
    heads=cm.config.num_key_value_heads
    query=state.query_position
    upper_bytes=0
    for i in range(reader.j):
        layer=state.bottom_cache.layers[i]
        for tensor in (layer.keys,layer.values):
            _require(tensor.shape[1]==heads and tensor.shape[-2]==query,"Bottom query cache shape differs")
    if is_prefix:
        prefix=state.top_cache.prefix
        prefix.assert_intact()
        _require(state.pack_position==prefix.token_count+query,"Query/pack position mismatch")
        for i in range(reader.j,reader.L):
            layer=state.top_cache.layers[i]
            for field,short in (("keys","k"),("values","v")):
                tensor=getattr(layer,field)
                _require(tensor.shape[1]==heads and tensor.shape[-2]==query,"Upper query cache contains extra tokens/heads")
                size=tensor.numel()*tensor.element_size()
                _require(tensor.untyped_storage().nbytes()==size,"Query view retains extra document storage")
                if i in prefix.pairs:
                    document=getattr(prefix.pairs[i],short)
                    _require(tensor.untyped_storage().data_ptr()!=document.untyped_storage().data_ptr(),
                             "Query cache aliases read-only document storage")
                upper_bytes+=size
        _require(state.route_stats["request_upper_query_kv_bytes"]==upper_bytes,"Query bytes accounting differs")
        return {"query_private":True,"prefix_unchanged":True,"request_upper_query_kv_bytes":upper_bytes,
                "prefix_kv_bytes":prefix.prefix_kv_bytes,"prefix_storage_bytes":prefix.prefix_storage_bytes,
                "hj_tensor_bytes":prefix.hj_tensor_bytes,"hj_storage_bytes":prefix.hj_storage_bytes,
                "query_position":query,"pack_position":state.pack_position,
                "persistent_kv_heads":heads}
    for i in range(reader.j,reader.L):
        layer=state.top_cache.layers[i]
        for tensor in (layer.keys,layer.values):
            _require(tensor.shape[1]==heads and tensor.shape[-2]==state.pack_position,"Native cold top cache shape differs")
    return {"query_private":True,"prefix_unchanged":True,"query_position":query,
            "pack_position":state.pack_position,"persistent_kv_heads":heads,
            "document_kv_bytes":state.route_stats["document_kv_bytes"],
            "storage_scope":"Cold native request owns its combined document+query top KV"}


def _generate(reader,sink,documents,row,*,stop_ids,max_new_tokens,check_guard,audit_state,immutable_check):
    """Independent natural greedy; state/audit callbacks never receive references."""
    import torch
    generated=[]
    state=logits=first_logits=None
    state_ref=None
    cache_refs=()
    route=None
    before=_flags()
    try:
        if check_guard is not None:check_guard()
        with torch.no_grad():
            logits,state=reader.prefill(sink,documents,row["prompt_ids"],probe_indices=row["probe_indices"])
            state_ref=weakref.ref(state)
            cache_refs=tuple(weakref.ref(cache) for cache in (state.bottom_cache,state.top_cache))
            route=copy.deepcopy(state.route_stats)
            stable_route={key:value for key,value in route.items() if key!="request_upper_query_kv_bytes"}
            observation=audit_state(state)
            immutable_check()
            finish="max_new_tokens"
            decode_calls=0
            for index in range(max_new_tokens):
                if logits.ndim!=3 or logits.shape[:2]!=(1,1) or not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("Invalid or nonfinite natural-generation logits")
                if first_logits is None:first_logits=logits.detach().to("cpu").clone()
                token=int(logits[0,-1].argmax().item())
                generated.append(token)
                if token in stop_ids:
                    finish="eos"
                    break
                if index+1<max_new_tokens:
                    if check_guard is not None and decode_calls%8==0:check_guard()
                    logits=reader.decode_step(token,state)
                    decode_calls+=1
                    observation=audit_state(state)
                    immutable_check()
                    actual={key:value for key,value in state.route_stats.items() if key!="request_upper_query_kv_bytes"}
                    _require(actual==stable_route,"Route changed during natural generation")
            _require(_flags()==before,"Backend flags changed during generation")
            immutable_check()
            route=copy.deepcopy(state.route_stats)
            result={"generated_ids":generated,"finish_reason":finish,
                    "eos_token_id":generated[-1] if finish=="eos" else None,
                    "stop_token_ids":sorted(stop_ids),"max_new_tokens":max_new_tokens,
                    "route_stats":route,"backend_flags":before}
        state=logits=None
        _require(state_ref() is None,"Generator retained request state after completion")
        _require(all(reference() is None for reference in cache_refs),"Generator retained a private request cache")
        observation.update(request_state_released=True,decode_calls=decode_calls)
        return result,first_logits,observation
    except Exception as exc:
        exc.quality_partial={"generated_ids":list(generated),"route_stats":route,
                             "stop_token_ids":sorted(stop_ids),"max_new_tokens":max_new_tokens}
        raise
    finally:
        state=logits=first_logits=None


def _first_logit_comparison(reference,candidate):
    import torch
    _require(reference.shape==candidate.shape,"First-logit shape differs")
    error=candidate.double()-reference.double()
    mean=error.mean()
    return {"scope":"Same prepared prompt; independent native and prefix prefill, observation only",
            "max_abs":float(error.abs().max()),"rms":float(error.square().mean().sqrt()),
            "signed_mean_error":float(mean),"centered_rms":float((error-mean).square().mean().sqrt()),
            "reference_dtype":str(reference.dtype),"candidate_dtype":str(candidate.dtype),
            "greedy_equal":bool(torch.equal(reference.argmax(-1),candidate.argmax(-1))),
            "numeric_gate_applied":False}


def run_native_prefix_group(native,prefix,sink,docs,rows,tokenizer,*,stop_ids,max_new_tokens=128,
                            on_record=None,check_guard=None):
    """One prepared whole-pack group, with immediate per-branch record callbacks.

    Starts a new prefix epoch (without resetting cumulative counters). The first
    prefix request misses/builds; later questions hit. Callbacks own JSONL output.
    A failed branch is emitted with partial generated IDs and then re-raised.
    """
    from native_infra_readers import NativeEncbankReader
    from native_prefix_reader import NativePrefixEncbankReader
    _require(type(native) is NativeEncbankReader and isinstance(prefix,NativePrefixEncbankReader),
             "Expected native cold and native prefix readers")
    _require(native.model is prefix.model and native.encbank is prefix.encbank,"Readers must share the exact same model/Encbank")
    _require(not native.model.training and not prefix.model.training,"Eval mode required")
    _require(type(max_new_tokens) is int and 1<=max_new_tokens<=128,"Natural generation budget must be1..128")
    _require(stop_ids and all(type(value) is int and value>=0 for value in stop_ids),"Nonempty valid stop IDs required")
    docs=list(docs);rows=list(rows)
    _require(bool(rows),"A whole-pack group must contain at least one question")
    _require(len({row["id"] for row in rows})==len(rows),"Duplicate question IDs")
    for row in rows:
        _require(all(row.get(field)==rows[0].get(field) for field in PACK_FIELDS),"Rows do not share exact ordered prepared pack")
        _require(len(row["document_chunks"])==len(docs)==len(row["selected_chunk_indices"]),"Prepared chunk count differs")
        _require([len(chunk) for chunk in row["document_chunks"]]==[int(doc.shape[1]) for doc in docs],"Prepared/h_j lengths differ")
        _require(row["references"] and all(isinstance(value,str) for value in row["references"]),"Invalid references")
        _require(row["prompt_ids"] and row["probe_indices"] and
                 all(type(i) is int and 0<=i<len(row["prompt_ids"]) for i in row["probe_indices"]),"Invalid prompt/probes")
        total=(0 if sink is None else int(sink.shape[1]))+sum(int(doc.shape[1]) for doc in docs)+len(row["prompt_ids"])
        _require(total+max_new_tokens-1<=native.encbank.config.max_position_embeddings,"Requested generation exceeds declared model window")
    prefix.invalidate()
    start_counts=_counts(prefix)
    shared_inputs=_input_identity(sink,docs)
    expected_prefix=None
    records=[]
    for query_index,row in enumerate(rows):
        first_native=None
        for branch,reader in (("native_cold",native),("native_prefix",prefix)):
            before=_counts(prefix)
            common={**{key:copy.deepcopy(row[key]) for key in ROW_FIELDS if key in row},
                    "branch":branch,"formal_inference_timing":False,"formal_inference_memory":False,
                    "decoding":"independent-free-greedy-natural-eos"}
            try:
                def immutable_check():
                    _require(_input_identity(sink,docs)==shared_inputs,"Shared writer h_j changed")
                    if expected_prefix is not None:_check_prefix_signature(prefix.prefix,expected_prefix)
                def audit_state(state):
                    nonlocal expected_prefix
                    observation=_audit_state(reader,state,branch=="native_prefix")
                    if branch=="native_prefix":
                        _require(prefix.prefix is state.top_cache.prefix,"Reader and request prefix differ")
                        if expected_prefix is None:expected_prefix=_prefix_signature(state.top_cache.prefix)
                        _check_prefix_signature(state.top_cache.prefix,expected_prefix)
                    return observation
                generated,first_logits,observation=_generate(reader,sink,docs,row,stop_ids=set(stop_ids),
                    max_new_tokens=max_new_tokens,check_guard=check_guard,
                    audit_state=audit_state,
                    immutable_check=immutable_check)
                after=_counts(prefix)
                observation.update(cross_request_kv_reuse=branch=="native_prefix",
                    **{key+"_before":value for key,value in before.items()},
                    **{key+"_after":value for key,value in after.items()})
                if branch=="native_prefix":
                    expected_hit=query_index>0
                    _require(generated["route_stats"]["prefix_cache_hit"]==expected_hit,"Unexpected whole-pack hit/miss")
                    _require(after["build_count"]-before["build_count"]==int(not expected_hit) and
                             after["miss_count"]-before["miss_count"]==int(not expected_hit) and
                             after["hit_count"]-before["hit_count"]==int(expected_hit),"Unexpected prefix lifecycle counters")
                    observation["prefix_cache_hit"]=expected_hit
                    expected_prefix=_prefix_signature(prefix.prefix)
                    comparison=_first_logit_comparison(first_native,first_logits)
                else:
                    _require(after==before,"Cold native request changed prefix counters")
                    first_native=first_logits
                    comparison=None
                prediction=visible_prediction(tokenizer,generated)
                record={**common,**generated,"status":"complete","prediction":prediction,
                        **score_prediction(prediction,row["references"]),"cache_observation":observation,
                        "first_logit_comparison":comparison,"shared_writer_hj_unchanged":True}
            except Exception as exc:
                record={**common,"status":"failed","error":{"type":type(exc).__name__,"message":str(exc)},
                        **getattr(exc,"quality_partial",{}),"cache_observation":{"counts_before":before,"counts_after":_counts(prefix)}}
                records.append(record)
                if on_record is not None:on_record(record)
                raise
            records.append(record)
            if on_record is not None:on_record(record)
            first_logits=None
        first_native=None
    after=_counts(prefix)
    _require(after["build_count"]-start_counts["build_count"]==1 and
             after["miss_count"]-start_counts["miss_count"]==1 and
             after["hit_count"]-start_counts["hit_count"]==len(rows)-1,"Group lifecycle counters differ")
    return records
