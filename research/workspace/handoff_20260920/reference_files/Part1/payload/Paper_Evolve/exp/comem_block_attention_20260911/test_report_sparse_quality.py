"""Standard-library fixtures for queue/final-score and budget completeness."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import aggregate_sparse as audit
from report_sparse_quality import build_report,render_markdown
from test_aggregate_sparse import make_row,make_evaluation,publish


def fixture(root,complete=True):
    data=root/"data"
    data.mkdir()
    train=[make_row("train",0),make_row("train",1,3)]
    dev=[make_row("dev",i,1+i%3) for i in range(99)]
    for i,row in enumerate(dev):
        row["document_id"]=f"dev-document-{i%35}"
    for split,rows in (("train",train),("dev",dev)):
        (data/(split+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    adapter=root/"initial-adapter.pt"
    adapter.write_bytes(b"test fixture, never deserialized")
    config={"steps":250,"grad_accum":4,"j":12,"m":16,"rho":.5,
        "arms":["D0","A","B","D1"],"init_adapter":str(adapter),"model":"/source/Qwen3-8B"}
    queue={"config":config,"jobs":{}}
    consumed=[]
    for epoch in range(500):
        order=[0,1]
        random.Random(42+epoch).shuffle(order)
        consumed.extend(train[index] for index in order)
    logs=[]
    raw=targets=0
    for step in range(1,251):
        batch=consumed[(step-1)*4:step*4]
        raw+=sum(sum(map(len,row["document_chunks"]))+len(row["prompt_ids"])+len(row["answer_ids"]) for row in batch)
        targets+=sum(len(row["answer_ids"]) for row in batch)
        logs.append({"step":step,"cursor":4*step,"raw_tokens":raw,"target_tokens":targets,
                     "loss":1.,"grad_norm":.1,"routes":[{"id":row["id"]} for row in batch]})
    for arm in config["arms"]:
        destination=root/"train"/arm
        queue["jobs"]["train/"+arm]={"stage":"train","arm":arm,"phase":"complete" if complete else "queued",
            "steps":250,"grad_accum":4,"dev_ids":[row["id"] for row in dev],"out":str(destination),"returncode":0}
        if not complete:
            continue
        recipe={"arm":arm,"j":12,"m":16,"rho":.5,"seed":42,"smoke":False,"steps":250,"grad_accum":4,
            "rank":32,"alpha":32.,"lr":2e-5,"max_new_tokens":128,"eval_limit":25,"final_eval_limit":100,
            "probe_mode":"block" if arm in ("A","B") else "dense",
            "retain_ratio":1. if arm in ("D0","A") else .5,"model":config["model"],"init_adapter":str(adapter),
            "init_adapter_sha256":audit.sha256(adapter),"reader_sha256":"a"*64}
        for split,rows in (("train",train),("dev",dev)):
            recipe.update({split:str(data/(split+".jsonl")),split+"_sha256":audit.sha256(data/(split+".jsonl")),
                           split+"_ids":[row["id"] for row in rows]})
        model={"num_hidden_layers":36,"num_key_value_heads":8,"head_dim":128,"hidden_size":4096,
               "num_attention_heads":32,"vocab_size":152000}
        metadata={"recipe":recipe,"model":model,"gpu":"NVIDIA GeForce RTX3090",
                  "formal_inference_timing":False,"formal_inference_memory":False}
        state={"arm":arm,"phase":"complete","complete":True,"step":250,"target_steps":250,"cursor":1000,
            "raw_tokens":raw,"target_tokens":targets,"gradient_check":{"reader_norm":.2,
                "frozen_base_has_grad":False,"total_trainable_tensors":336,"trainable_tensors_with_grad":336}}
        evaluation=make_evaluation(dev,recipe)
        evaluation["summary"]["max_new_tokens"]=128
        for row in evaluation["records"]:
            for key in ("route_stats","ce_route_stats"):
                row[key]["document_kv_bytes"]=4096*sum(row[key]["document_kv_tokens_by_layer"].values())
        publish(destination/"metadata.json",metadata)
        publish(destination/"status.json",state)
        publish(destination/"eval_step250.json",evaluation)
        (destination/"last.pt").write_bytes(b"test fixture only; no weights loaded")
        (destination/"train.jsonl").write_text("".join(json.dumps(log)+"\n" for log in logs),encoding="utf-8")
    publish(root/"queue.json",queue)
    return data


class SparseQualityReport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="sparse-quality-report-")
        self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def report(self):
        return build_report(self.root/"queue.json",self.root/"data/train.jsonl",self.root/"data/dev.jsonl")
    def change(self,name,fn,arm="B"):
        path=self.root/"train"/arm/name
        value=json.loads(path.read_text())
        fn(value)
        publish(path,value)
    def rejected_b(self):
        result=self.report()
        self.assertIsNone(result["arms"]["B"]["final"])
        self.assertNotIn("B",result["final_vs_D0"])
        self.assertFalse(result["comparison_valid"])
        return result

    def test_all_queued_stay_blank_and_planned_budget_is_not_actual(self):
        fixture(self.root,False)
        before=(self.root/"queue.json").read_bytes()
        result=self.report()
        self.assertEqual(result["status"],"pending",result)
        self.assertTrue(all(row["status"]=="queued" and row["final"] is None
                            and row["actual_budget"] is None for row in result["arms"].values()))
        self.assertEqual(result["expected_budget"]["raw_tokens"],15000)
        self.assertEqual(result["expected_budget"]["target_tokens"],2000)
        self.assertEqual(result["prepared"]["dev_documents"],35)
        self.assertIn("| D0 | queued | — | —",render_markdown(result))
        self.assertEqual(before,(self.root/"queue.json").read_bytes())

    def test_full_matching_four_arms_and_pairwise_without_every_arm(self):
        fixture(self.root)
        result=self.report()
        self.assertEqual(result["status"],"complete",result)
        self.assertTrue(result["comparison_valid"])
        self.assertEqual(result["arms"]["B"]["final"]["mean_actual_retain_ratio"],.5)
        self.assertEqual(result["final_vs_D0"]["B"]["f1_win_tie_loss"],{"win":0,"tie":99,"loss":0})
        path=self.root/"queue.json"
        queue=json.loads(path.read_text())
        queue["jobs"]["train/A"]["phase"]="running"
        queue["jobs"]["train/D1"]["phase"]="queued"
        publish(path,queue)
        result=self.report()
        self.assertEqual(set(result["final_vs_D0"]),{"B"})
        self.assertIsNone(result["arms"]["A"]["final"])
        self.assertIsNone(result["arms"]["D1"]["final"])

    def test_25_question_or_missing_final_cannot_fill(self):
        fixture(self.root)
        self.change("eval_step250.json",lambda value:value.update(records=value["records"][:25]))
        self.rejected_b()
        (self.root/"train/B/eval_step250.json").rename(self.root/"train/B/eval_step0.json")
        self.rejected_b()

    def test_duplicate_final_question_is_rejected(self):
        fixture(self.root)
        def duplicate(value):
            value["records"][1]=copy.deepcopy(value["records"][0])
        self.change("eval_step250.json",duplicate)
        self.rejected_b()

    def test_token_budget_and_training_log_order_are_rejected(self):
        fixture(self.root)
        self.change("status.json",lambda value:value.update(raw_tokens=value["raw_tokens"]+1))
        self.rejected_b()
        self.change("status.json",lambda value:value.update(raw_tokens=value["raw_tokens"]-1))
        path=self.root/"train/B/train.jsonl"
        logs=[json.loads(line) for line in path.read_text().splitlines()]
        logs[3]["routes"][0]["id"]="wrong-training-example"
        path.write_text("".join(json.dumps(row)+"\n" for row in logs))
        self.rejected_b()

    def test_wrong_mean_and_wrong_per_question_score_rejected(self):
        fixture(self.root)
        path=self.root/"train/B/eval_step250.json"
        original=json.loads(path.read_text())
        self.change("eval_step250.json",lambda value:value["summary"].update(token_f1=.123))
        self.rejected_b()
        publish(path,original)
        changed=copy.deepcopy(original)
        first=changed["records"][0]
        first["prediction"]=(first["references"][0] if first["token_f1"]!=1. else "not the recorded answer")
        publish(path,changed)
        self.rejected_b()

    def test_other_training_recipe_difference_blocks_only_unmatched_arm(self):
        fixture(self.root)
        self.change("metadata.json",lambda value:value["recipe"].update(lr=3e-5))
        result=self.rejected_b()
        self.assertEqual(result["arms"]["B"]["status"],"unmatched")
        self.assertEqual(set(result["final_vs_D0"]),{"A","D1"})

    def test_failed_or_smoke_artifacts_never_promote(self):
        fixture(self.root)
        path=self.root/"queue.json"
        queue=json.loads(path.read_text())
        queue["jobs"]["train/B"].update(phase="failed",error="OOM")
        publish(path,queue)
        result=self.rejected_b()
        self.assertEqual(result["arms"]["B"]["status"],"failed")
        queue["jobs"]["train/B"]["phase"]="complete"
        publish(path,queue)
        self.change("metadata.json",lambda value:value["recipe"].update(smoke=True))
        self.rejected_b()


if __name__=="__main__":
    unittest.main(verbosity=2)
