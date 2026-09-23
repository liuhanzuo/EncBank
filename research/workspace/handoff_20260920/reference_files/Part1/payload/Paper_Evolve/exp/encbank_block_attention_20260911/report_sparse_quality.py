"""Read-only queue-aware final quality report; standard library only.

Reuse aggregate_sparse's prepared-data, training-log, score and route audits,
plus the real queue completion predicate. Never load checkpoints or run models.
Only --out creates new report files; all queue/training/data inputs stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random

import aggregate_sparse as audit
from backend_quality_results import score_prediction
from remote_sparse_queue import completed_job

ARMS=("D0","A","B","D1")
ALGORITHM_FIELDS={"arm","probe_mode","retain_ratio"}


def expected_budget(train):
    consumed=[]
    epoch=0
    while len(consumed)<1000:
        order=list(range(len(train)))
        random.Random(42+epoch).shuffle(order)
        consumed.extend(train[index] for index in order)
        epoch+=1
    consumed=consumed[:1000]
    ids=[row["id"] for row in consumed]
    return {"steps":250,"grad_accum":4,"seed":42,"examples_consumed":1000,
        "raw_tokens":sum(sum(map(len,row["document_chunks"]))+len(row["prompt_ids"])+len(row["answer_ids"]) for row in consumed),
        "target_tokens":sum(len(row["answer_ids"]) for row in consumed),
        "ordered_training_ids_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),
        "raw_token_scope":"Trainer count: document_chunks + prompt_ids + answer_ids per consumed example; reusable sink excluded"}


def _prepared(train_path,dev_path):
    paths={"train":Path(train_path),"dev":Path(dev_path)}
    raw={split:[json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
         for split,path in paths.items()}
    recipe={}
    for split,path in paths.items():
        recipe.update({split:str(path),split+"_sha256":audit.sha256(path),
                       split+"_ids":[row["id"] for row in raw[split]]})
    train=audit.read_prepared(recipe,"train",None,())
    dev=audit.read_prepared(recipe,"dev",None,())
    audit.require(len(dev)==99,"Final report requires the entire 99-example prepared dev set")
    audit.require(not ({row["document_id"] for row in train}&{row["document_id"] for row in dev}),
                  "Prepared train/dev documents overlap")
    return train,dev,recipe


def _paired(reference,candidate):
    audit.require([r["id"] for r in reference]==[r["id"] for r in candidate],"Paired final IDs differ")
    deltas={metric:[right[metric]-left[metric] for left,right in zip(reference,candidate)]
            for metric in ("token_f1","exact_match")}
    return {"examples":99,"ids":[row["id"] for row in reference],
        "delta_f1":sum(deltas["token_f1"])/99,"delta_em":sum(deltas["exact_match"])/99,
        "f1_win_tie_loss":{"win":sum(x>0 for x in deltas["token_f1"]),
                           "tie":sum(x==0 for x in deltas["token_f1"]),
                           "loss":sum(x<0 for x in deltas["token_f1"])}}


def build_report(queue_path,train_path,dev_path):
    queue_path=Path(queue_path)
    result={"schema":"queue-aware-sparse-final-quality-v1","status":"pending",
        "generated_utc":datetime.now(timezone.utc).isoformat(),"queue_path":str(queue_path),
        "formal_inference_timing":False,"formal_inference_memory":False,
        "score_scale":"0-to-1 in JSON; percentages in Markdown",
        "protocol":"Document-held-out Qasper official-train pilot;99 questions,not official test",
        "arms":{arm:{"phase":None,"status":"pending","final":None,"actual_budget":None,"reason":None}
                for arm in ARMS},"final_vs_D0":{},"errors":[],"comparison_valid":False,
        "audit_reuse":["remote_sparse_queue.completed_job: completion/checkpoint/gradient/final-eval gate",
            "aggregate_sparse.read_prepared: data hashes,IDs,complete targets",
            "aggregate_sparse.training_audit: every step/order/cumulative raw and supervised tokens",
            "aggregate_sparse.evaluation_audit:99 IDs,answer scores/means,CE,route/physical KV accounting"],
        "audit_added":["Queue train phase,250x4/seed42,j12/m16 and full99-only restriction",
            "Source adapter/data hashes and full recipe comparability except the three algorithm fields",
            "Independent stdlib score_prediction cross-check;pairwise complete D0 contrasts and finish/rho statistics"],
        "limitations":["Checkpoint existence/nonempty is checked on source server without deserialization.",
            "Finish rates use stored trainer finish_reason;old records do not persist the complete stop-token set.",
            "Questions share documents;no independence or non-inferiority claim is made."]}
    validated={}
    try:
        queue=audit.read_json(queue_path)
        config=queue["config"]
        for key,value in {"steps":250,"grad_accum":4,"j":12,"m":16,"rho":.5,"arms":list(ARMS)}.items():
            audit.require(config.get(key)==value,f"Queue {key} differs from fixed four-arm plan")
        train,dev,data_recipe=_prepared(train_path,dev_path)
        expected=expected_budget(train)
        adapter_path=Path(config["init_adapter"])
        audit.require(adapter_path.is_file(),"Source initial adapter unavailable; run report on source server")
        adapter_sha=audit.sha256(adapter_path)
        result.update(queue_updated_utc=queue.get("updated_utc"),queue_reason=queue.get("reason"),
            expected_budget=expected,prepared={"train_examples":len(train),"dev_examples":len(dev),
                "train_documents":len({row["document_id"] for row in train}),
                "dev_documents":len({row["document_id"] for row in dev}),
                "train_sha256":data_recipe["train_sha256"],"dev_sha256":data_recipe["dev_sha256"]},
            init_adapter_sha256=adapter_sha)
        for arm in ARMS:
            item=result["arms"][arm]
            try:
                job=queue["jobs"]["train/"+arm]
                directory=Path(job["out"])
                phase=job.get("phase")
                item.update(phase=phase,status=phase,out=str(directory))
                audit.require(job["stage"]=="train" and job["arm"]==arm and job["steps"]==250
                    and job["grad_accum"]==4 and job["dev_ids"]==data_recipe["dev_ids"],"Wrong train job/IDs/budget")
                if phase!="complete":
                    audit.require(phase in ("queued","launching","running","failed"),"Unknown queue phase")
                    item["reason"]=job.get("error") if phase=="failed" else "Full training/final evaluation not complete"
                    continue
                audit.require(job.get("returncode")==0 and not job.get("guard_failure"),"Queue completion has a failure/unknown exit code")
                audit.require(completed_job(job),"Queue completion predicate rejected source artifacts")
                metadata=audit.read_json(directory/"metadata.json")
                state=audit.read_json(directory/"status.json")
                recipe,model=metadata["recipe"],metadata["model"]
                expected_recipe={"arm":arm,"steps":250,"grad_accum":4,"j":12,"m":16,"seed":42,
                    "rho":.5,"smoke":False,"max_new_tokens":128,"final_eval_limit":100,
                    "probe_mode":"block" if arm in ("A","B") else "dense",
                    "retain_ratio":1. if arm in ("D0","A") else .5,"rank":32,"alpha":32.,
                    "init_adapter_sha256":adapter_sha,"model":config["model"],"init_adapter":config["init_adapter"],
                    **{key:value for key,value in data_recipe.items() if key.endswith(("_ids","_sha256"))}}
                for key,value in expected_recipe.items():
                    audit.require(recipe.get(key)==value,f"Final recipe {key} mismatch")
                audit.require(model["num_hidden_layers"]==36 and model["num_key_value_heads"]==8,
                              "Final model is not the expected 36-layer/8-KV-head architecture")
                audit.require(metadata["formal_inference_timing"] is False and metadata["formal_inference_memory"] is False,
                              "Training quality cannot be formal inference infra")
                budget=audit.training_audit(directory,state,recipe,train)
                for key in ("steps","examples_consumed","raw_tokens","target_tokens"):
                    audit.require(budget[key]==expected[key],f"Actual {key} differs from fixed training order")
                checked,summary=audit.evaluation_audit(directory/"eval_step250.json",dev,recipe,model,99)
                raw=audit.read_json(directory/"eval_step250.json")["records"]
                for record in raw:
                    recalculated=score_prediction(record["prediction"],record["references"])
                    for metric in ("token_f1","exact_match"):
                        audit.close(record[metric],recalculated[metric],"Shared normalized scorer "+metric)
                counts={reason:sum(row["finish_reason"]==reason for row in raw) for reason in ("eos","max_new_tokens")}
                final={"examples":99,"token_f1":summary["token_f1"],"exact_match":summary["exact_match"],
                    "mean_actual_retain_ratio":summary["mean_actual_retain_ratio"],"finish_counts":counts,
                    "finish_rates":{name:count/99 for name,count in counts.items()}}
                item.update(status="complete",final=final,actual_budget=budget,reason=None,source_gpu=metadata["gpu"])
                validated[arm]={"recipe":{k:v for k,v in recipe.items() if k not in ALGORITHM_FIELDS},
                                "model":model,"budget":budget,"rows":checked}
            except (audit.Pending,ValueError,KeyError,TypeError,OSError,IndexError,AttributeError,ZeroDivisionError) as exc:
                item.update(status="invalid",final=None,reason=str(exc))
                result["errors"].append(f"{arm}: {exc}")
        if validated:
            anchor="D0" if "D0" in validated else next(arm for arm in ARMS if arm in validated)
            for arm in list(validated):
                mismatched=[key for key in ("recipe","model","budget") if validated[arm][key]!=validated[anchor][key]]
                if mismatched:
                    item=result["arms"][arm]
                    item.update(status="unmatched",final=None,reason=f"Differs from {anchor}: {', '.join(mismatched)}")
                    result["errors"].append(f"{arm}: "+item["reason"])
                    del validated[arm]
        if "D0" in validated:
            result["final_vs_D0"]={arm:_paired(validated["D0"]["rows"],validated[arm]["rows"])
                                   for arm in ARMS[1:] if arm in validated}
        result["comparison_valid"]=len(validated)==4
        result["status"]=("invalid" if result["errors"] else "failed" if any(x["status"]=="failed" for x in result["arms"].values())
            else "complete" if result["comparison_valid"] else "partial" if validated else "pending")
    except (audit.Pending,ValueError,KeyError,TypeError,OSError,IndexError) as exc:
        result["status"]="invalid"
        result["errors"].append(str(exc))
    return result


def render_markdown(result):
    lines=["# Encbank sparse：四臂最终质量", "",result["protocol"],"",
        f"状态：**{result['status']}**。四臂完全匹配：**{result['comparison_valid']}**。",
        "", "仅完整250步、99题的最终结果可填分；无数据使用破折号。", "",
        "| Arm | Status | F1 (%) | EM (%) | Mean actual rho | EOS / limit |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        item=result["arms"][arm]
        final=item["final"]
        values=(f"{final['token_f1']*100:.2f} | {final['exact_match']*100:.2f} | {final['mean_actual_retain_ratio']:.4f} | "
                f"{final['finish_counts']['eos']} / {final['finish_counts']['max_new_tokens']}" if final else "— | — | — | —")
        lines.append(f"| {arm} | {item['status']} | {values} |")
    if result.get("expected_budget"):
        b=result["expected_budget"]
        p=result["prepared"]
        lines += ["",f"固定训练预算：250 × 4 = 1000样本；raw tokens {b['raw_tokens']:,}，监督token {b['target_tokens']:,}。",
            f"开发集99题来自{p['dev_documents']}篇文档；题目并非99个独立文档。", "",
            "| Arm | Actual raw tokens | Actual target tokens |", "| --- | ---: | ---: |"]
        for arm in ARMS:
            b=result["arms"][arm]["actual_budget"]
            lines.append(f"| {arm} | {b['raw_tokens'] if b else '—'} | {b['target_tokens'] if b else '—'} |")
    if result["final_vs_D0"]:
        lines += ["", "已匹配的完整结果，相对D0：", "",
            "| Arm | Paired N | ΔF1 (pp) | ΔEM (pp) | F1 win / tie / loss |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for arm,delta in result["final_vs_D0"].items():
            counts=delta["f1_win_tie_loss"]
            lines.append(f"| {arm} | 99 | {delta['delta_f1']*100:+.2f} | {delta['delta_em']*100:+.2f} | {counts['win']} / {counts['tie']} / {counts['loss']} |")
    lines += ["", "不含正式5090速度或显存结果；未反序列化checkpoint。", ""]
    lines += [f"- {arm}：{item['reason']}" for arm,item in result["arms"].items() if item.get("reason")]
    lines += ["- 审计错误："+error for error in result["errors"]]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("queue","train","dev"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=build_report(args.queue,args.train,args.dev)
    if args.out is not None:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/"SPARSE_QUALITY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (args.out/"SPARSE_QUALITY.md").write_text(render_markdown(result),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    raise SystemExit(1 if result["status"]=="invalid" else 0)


if __name__=="__main__":
    main()
