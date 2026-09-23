"""Server-only KV reconstruction microbenchmark; no benchmark task answers used."""
import collections
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
MODEL = "/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B"
ADAPTER = "/srv/encbank/comem_infra_recheck_20260912/adapter"
MISSES = [0, 1, 2, 4, 8, 12]
WARMUPS, REPEATS = 2, 7


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def status(phase, **kw):
    value = dict(phase=phase, at=datetime.datetime.now().astimezone().isoformat(), **kw)
    dump("status.json", value)
    print(json.dumps(value), flush=True)


def main():
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must have its own Slurm allocation"
    assert not (OUT / "summary.json").exists(), "Do not overwrite a completed measurement"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from engine import Reader
    from peft import PeftModel
    from unittest.mock import patch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.manual_seed(20260921)
    assert torch.cuda.device_count() == 1
    prop = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    assert free > 80 * 2**30, "Insufficient free memory on allocated GPU; do not disturb other jobs"
    torch.cuda.set_per_process_memory_fraction(64 * 2**30 / total, 0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    adapter_sha = hashlib.sha256((Path(ADAPTER) / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert adapter_sha == "1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13"
    environment = dict(job=os.environ["SLURM_JOB_ID"], pid=os.getpid(), host=platform.node(),
        gpu=prop.name, gpu_uuid=str(prop.uuid), cc=[prop.major, prop.minor], total_memory_bytes=total,
        free_before_load_bytes=free, allocator_cap_bytes=64 * 2**30,
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
        model=MODEL, adapter=ADAPTER, adapter_sha256=adapter_sha,
        dtype="BF16 backbone; original FP32 unmerged LoRA", j=12, chunk_tokens=512,
        cpu_affinity=sorted(os.sched_getaffinity(0)), warmups=WARMUPS, repeats=REPEATS,
        attention="SDPA flash/efficient backend, explicit repeat-KV as in previous deployment probe",
        inference_only=True, local_model_files_downloaded=False)
    dump("environment.json", environment)
    status("LOAD_MODEL")
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    with patch("peft.tuners.lora.model.dispatch_torchao", return_value=None):
        wrapper = PeftModel.from_pretrained(model, ADAPTER, autocast_adapter_dtype=True)
    model = wrapper.base_model.model.eval()
    model.requires_grad_(False)
    assert all(p.device.type == "cuda" for p in model.parameters())
    assert model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
    assert model.config.num_key_value_heads == 8 and model.config.head_dim == 128
    reader = Reader(model, 12, tokenizer=tok)
    layers = list(range(12, 36))
    environment["model_load_seconds"] = time.perf_counter() - started
    dump("environment.json", environment)
    docs = json.loads((ROOT / "vendor" / "workloads.json").read_text())

    def synchronize():
        torch.cuda.synchronize()
        return time.perf_counter()

    def timed(fn):
        synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        a.record()
        result = fn()
        b.record()
        b.synchronize()
        return result, dict(wall_ms=(time.perf_counter()-begin)*1000, cuda_ms=a.elapsed_time(b))

    def fresh():
        return DynamicCache(config=model.config)

    def kv(cache, layer):
        return cache.layers[layer].keys, cache.layers[layer].values

    def cache_bytes(cache):
        return sum(t.numel()*t.element_size() for i in layers for t in kv(cache, i))

    def upper(h, cache=None, past=0):
        cache = fresh() if cache is None else cache
        n = h.shape[1]
        positions = torch.arange(past, past+n, device="cuda")[None].expand(h.shape[0], -1)
        mask = (torch.arange(past+n, device="cuda")[None, :] <= positions[0, :, None])[None, None]
        result = reader._run_layers(h, slice(12,36), mask, positions,
            reader.rotary_emb(h, position_ids=positions), past_key_values=cache, use_cache=True)
        return cache, result

    def clone_prefix(cache, length):
        result = fresh()
        for i in layers:
            k, v = kv(cache, i)
            result.update(k[:, :, :length].clone(), v[:, :, :length].clone(), i)
        return result

    def split_rows(caches):
        blocks = []
        for cache in caches:
            for row in range(kv(cache, 12)[0].shape[0]):
                block = fresh()
                for i in layers:
                    k, v = kv(cache, i)
                    block.update(k[row:row+1].clone(), v[row:row+1].clone(), i)
                blocks.append(block)
        return blocks

    def assemble(sink_cache, blocks):
        result = fresh()
        for i in layers:
            result.update(*[torch.cat([kv(sink_cache,i)[c]] + [kv(x,i)[c] for x in blocks], dim=2)
                for c in [0,1]], i)
        return result

    def compare(a, b):
        max_abs = 0.0
        sq, refsq, numel = 0.0, 0.0, 0
        for i in layers:
            for x,y in zip(kv(a,i),kv(b,i)):
                assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
                xf,yf=x.float(),y.float()
                d=xf-yf
                max_abs=max(max_abs,float(d.abs().max()))
                sq+=float(d.square().sum());refsq+=float(yf.square().sum());numel+=d.numel()
        result=dict(max_abs=max_abs, rms=(sq/numel)**.5, relative_rms=(sq/max(refsq,1e-30))**.5)
        assert result["relative_rms"] < .03, result
        return result

    records, validations, decode_records = [], [], []
    record_file = (OUT / "measurements.jsonl").open("w", encoding="utf-8")
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for di, doc in enumerate(docs[:3]):
            status("PREPARE_DOCUMENT", document=di)
            chosen = doc["queries"][0]["selected"]
            assert len(chosen)==12 and chosen==sorted(chosen)
            ids=torch.tensor([doc["source"][i*512:(i+1)*512] for i in chosen], device="cuda")
            query_ids=doc["queries"][0]["query"]
            assert len(query_ids)==512 and ids.shape==(12,512)
            h=reader.write_chunk(ids)
            sink_h=reader.write_chunk([model.config.bos_token_id])
            packed=torch.cat([sink_h,h.reshape(1,6144,-1)],dim=1)
            full_cache,_=upper(packed)
            independent_cache,_=upper(h)
            hot=split_rows([independent_cache])
            sink_cache,_=upper(sink_h)
            del independent_cache
            assert cache_bytes(hot[0]) == 48*2**20
            assert cache_bytes(full_cache) == 6145*96*1024
            dump("input_"+str(di)+".json",dict(document=di,selected=chosen,query_tokens=512,
                selected_tokens_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                H_bytes=h.numel()*h.element_size(),KV_bytes_per_chunk=cache_bytes(hot[0])))

            # Check two independent chunks, serial versus batch; a different attention
            # graph from joint-document prefill, never a claim of original answer parity.
            serial=[]
            for ix in range(2):
                c,_=upper(h[ix:ix+1]);serial.append(c)
            batch,_=upper(h[:2]);batch_parts=split_rows([batch])
            checks=[compare(x,y) for x,y in zip(batch_parts,serial)]
            del serial,batch,batch_parts,c
            prefix_checks=[]
            for miss in [0,1,4,12]:
                hit=12-miss;cache=clone_prefix(full_cache,1+hit*512)
                if miss:
                    cache,_=upper(h[hit:].reshape(1,miss*512,-1),cache,1+hit*512)
                prefix_checks.append(dict(misses=miss,**compare(cache,full_cache)))
                del cache
            validations.append(dict(document=di,independent_batch_vs_serial=checks,prefix_vs_full=prefix_checks))
            dump("validation.json",dict(passed=True,relative_rms_limit=.03,checks=validations,
                scope="Same-attention-graph BF16 cache consistency only; no task quality claim"))

            def point(mode, miss):
                hit=12-miss
                torch.cuda.reset_peak_memory_stats()
                begin=synchronize()
                zero=dict(wall_ms=0.0,cuda_ms=0.0)
                restore,store,assembly=zero.copy(),zero.copy(),zero.copy()
                if mode=="exact_prefix":
                    result,restore=timed(lambda:clone_prefix(full_cache,1+hit*512))
                    if miss:
                        (result,hidden),rebuild=timed(lambda:upper(h[hit:].reshape(1,miss*512,-1),result,1+hit*512))
                        del hidden
                    else:rebuild=zero.copy()
                else:
                    def build():
                        if mode=="independent_batch":
                            cache,_=upper(h[hit:]);return [cache]
                        caches=[]
                        for ix in range(hit,12):
                            cache,_=upper(h[ix:ix+1]);caches.append(cache)
                        return caches
                    if miss:
                        new,rebuild=timed(build)
                        additions,store=timed(lambda:split_rows(new))
                        del new
                    else:additions=[];rebuild=zero.copy()
                    active=hot[:hit]+additions
                    result,assembly=timed(lambda:assemble(sink_cache,active))
                elapsed=(synchronize()-begin)*1000
                row=dict(document=di,mode=mode,misses=miss,hits=hit,rebuild=rebuild,restore=restore,
                    store=store,assembly=assembly,total_wall_ms=elapsed,active_document_KV_bytes=cache_bytes(result),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                return row

            points=[(mode,m) for mode in ["exact_prefix","independent_serial","independent_batch"] for m in MISSES]
            random.Random(20260921+di).shuffle(points)
            for mode,miss in points:
                status("MEASURE",document=di,mode=mode,misses=miss,completed=len(records))
                for repeat in range(WARMUPS+REPEATS):
                    row=point(mode,miss)
                    if repeat>=WARMUPS:
                        row["repeat"]=repeat-WARMUPS
                        records.append(row);record_file.write(json.dumps(row)+"\n");record_file.flush()
                gc.collect()

            # Reference: ordinary CoMem generation with the joint 12-chunk cache.
            # Always emit 512 tokens, ignoring EOS, for a fixed-shape timing comparison.
            status("DECODE_512",document=di)
            for warmup,count in [(True,32),(False,512)]:
                top=clone_prefix(full_cache,6145)
                def prefill_query():
                    qh,lower,qpos=reader.write_prefill(query_ids)
                    _,last=upper(qh,top,6145)
                    logits=reader.lm_head(reader.norm(last[:,-1:]))
                    return lower,qpos,logits
                (lower,qpos,logits),query_time=timed(prefill_query)
                first=int(logits[:,-1].argmax(-1).item())
                generated=[first];token=torch.tensor([first],device="cuda")
                begin=synchronize()
                for step in range(1,count):
                    logits=reader.decode_step(token,lower,top,qpos+step-1,6657+step-1)
                    value=int(logits[:,-1].argmax(-1).item())
                    generated.append(value);token.fill_(value)
                elapsed=(synchronize()-begin)*1000
                if not warmup:
                    decode_records.append(dict(document=di,generated_tokens=count,decode_forwards=count-1,
                        decode_ms=elapsed,ms_per_decode_token=elapsed/(count-1),query_prefill=query_time,
                        token_ids=generated,active_upper_KV_bytes=cache_bytes(top)))
                    dump("decode.json",decode_records)
                del top,lower,logits,token
            del h,sink_h,packed,full_cache,hot,sink_cache,ids
            gc.collect();torch.cuda.empty_cache()
    record_file.close()

    def stats(values):
        values=sorted(values)
        return dict(n=len(values),median=statistics.median(values),min=values[0],max=values[-1],
            p95=values[min(len(values)-1,int(.95*(len(values)-1)))])
    grouped=collections.defaultdict(list)
    for row in records:grouped[(row["mode"],row["misses"])].append(row)
    cells=[]
    for (mode,miss),rows in sorted(grouped.items()):
        cells.append(dict(mode=mode,misses=miss,n=len(rows),
            rebuild_wall_ms=stats([x["rebuild"]["wall_ms"] for x in rows]),
            rebuild_cuda_ms=stats([x["rebuild"]["cuda_ms"] for x in rows]),
            restore_wall_ms=stats([x["restore"]["wall_ms"] for x in rows]),
            store_wall_ms=stats([x["store"]["wall_ms"] for x in rows]),
            assembly_wall_ms=stats([x["assembly"]["wall_ms"] for x in rows]),
            total_wall_ms=stats([x["total_wall_ms"] for x in rows]),
            peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in rows)))
    summary=dict(complete=True,environment=environment,cells=cells,formal_observations=len(records),
        decode_512_ms=stats([x["decode_ms"] for x in decode_records]),
        decode_ms_per_token=stats([x["ms_per_decode_token"] for x in decode_records]),
        scope="Microbenchmark, 3 saved PG19 inputs, one model process; no live agent trajectory or retrieval/eviction policy benchmark",
        reconstruction_boundary="GPU H through all 24 upper blocks with KV retention; excludes retrieval, initial H Write, LM head and query",
        independent_scope="Independent chunk attention and chunk-local positions; changes document attention graph versus current joint CoMem",
        prefix_scope="Unchanged ordered 12-chunk context; cached prefix and missing suffix; includes actual prefix clones",
        timer_scope="CUDA synchronized wall time plus CUDA events; fixed active pack 12x512+sink; output is 512 tokens with 511 decode forwards",
        source_manifest=json.loads((ROOT/"source_manifest.json").read_text()))
    dump("summary.json",summary)
    status("COMPLETE",formal_observations=len(records))


if __name__=="__main__":
    try:
        main()
    except BaseException:
        dump("failure.json",dict(traceback=traceback.format_exc(),at=datetime.datetime.now().astimezone().isoformat()))
        raise
