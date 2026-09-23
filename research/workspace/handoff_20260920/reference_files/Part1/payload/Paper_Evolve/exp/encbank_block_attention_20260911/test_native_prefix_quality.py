"""Bounded CPU checks for natural QA generation and whole-pack group lifecycle."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import torch

from native_prefix_quality import _generate,run_native_prefix_group
from test_native_prefix_reader import setup,precision_context


QUALITY_OBSERVATIONS=[]
GROUP_PROTOCOL_OBSERVATIONS=[]


class Container:
    pass


class ScriptedReader:
    def __init__(self,sequence):
        self.sequence=sequence
        self.cursor=0
        self.inputs=[]
        self.prefills=[]
    def logits(self):
        result=torch.full((1,1,50),-10.)
        result[0,0,self.sequence[min(self.cursor,len(self.sequence)-1)]]=10.
        return result
    def prefill(self,sink,docs,prompt,probe_indices=None):
        self.cursor=0
        self.prefills.append((prompt,probe_indices))
        state=Container()
        state.bottom_cache=Container();state.top_cache=Container()
        state.route_stats={"selected_indices":[0]}
        return self.logits(),state
    def decode_step(self,token,state):
        self.inputs.append(token)
        self.cursor+=1
        return self.logits()


class Tokenizer:
    def decode(self,ids,skip_special_tokens=True):
        return " ".join(map(str,ids))
    def encode(self,*args,**kwargs):
        raise AssertionError("Prepared prompts must never be tokenized again")


def rows():
    base={"document_id":"shared-document","source":"tiny-qwen-fixture",
          "document_chunks":[[3,4,5],[6,7,8],[9,10,11]],"selected_chunk_indices":[0,1,2],
          "references":["31","32 33"]}
    return [{**copy.deepcopy(base),"id":"question-one","question":"First independent question?",
             "prompt_ids":[21,22,23],"probe_indices":[0,2]},
            {**copy.deepcopy(base),"id":"question-two","question":"A different question?",
             "prompt_ids":[24,25,26,27],"probe_indices":[1,2]}]


def generate_fake(reader,budget=128,guard=None):
    return _generate(reader,None,[],rows()[0],stop_ids={0},max_new_tokens=budget,check_guard=guard,
                     audit_state=lambda state:{},immutable_check=lambda:None)


class NativePrefixQualityTests(unittest.TestCase):
    def test_real_groups_flow_through_fixed_plan_and_paired_summary(self):
        from backend_quality_results import score_prediction
        from prefix_quality_protocol import (LOCATION_FIELDS,build_prefix_quality_plan,
                                             summarize_prefix_quality)
        prepared=[]
        for group_index in range(2):
            for question_index,row in enumerate(rows()):
                row.update(id=f"pack-{group_index}-q-{question_index}",
                           document_id=f"document-{group_index}",
                           source="allenai/qasper:train:v0.3",split="dev")
                if group_index:
                    row["document_chunks"]=[[12,13],[14,15,16],[17,18,19]]
                    row["prompt_ids"]=[value+13 for value in row["prompt_ids"]]
                prepared.append(row)
        for index in range(95):
            row=copy.deepcopy(prepared[0])
            row.update(id=f"singleton-{index}",document_id=f"singleton-document-{index}")
            prepared.append(row)
        plan=build_prefix_quality_plan(prepared,mode="smoke")
        self.assertEqual((len(plan["groups"]),len(plan["ordered_ids"])),(2,4))
        native,prefix,sink,_=setup("bf16")
        collected=[]
        locations={row["id"]:row for row in plan["ordered_rows"]}
        def on_record(record):
            location=locations[record["id"]]
            self.assertTrue(all(key not in record or record[key]==location[key] for key in LOCATION_FIELDS))
            collected.append({**record,**{key:location[key] for key in LOCATION_FIELDS}})
        with precision_context(prefix,"bf16"):
            for group in plan["groups"]:
                group_rows=[plan["ordered_rows"][index] for index in group["row_ordinals"]]
                # The h_j pack is freshly written from this exact group's IDs.
                # Questions in the group share those same objects, never another
                # group's writer output or a changed retrieval selection.
                documents=[native.write_chunk(chunk) for chunk in group_rows[0]["document_chunks"]]
                run_native_prefix_group(native,prefix,sink,documents,group_rows,Tokenizer(),
                    stop_ids={0},max_new_tokens=6,on_record=on_record)
        summary=summarize_prefix_quality(plan,collected,stop_token_ids=[0],max_new_tokens=6)
        self.assertEqual(summary["status"],"complete",summary)
        self.assertEqual(summary["records_received"],8)
        self.assertEqual((prefix.build_count,prefix.hit_count,prefix.miss_count),(2,2,2))
        self.assertEqual(summary["metrics"]["prefix_builds"],2)
        self.assertEqual(summary["metrics"]["prefix_hits"],2)
        for record in collected:
            expected=score_prediction(record["prediction"],locations[record["id"]]["references"])
            self.assertEqual({key:record[key] for key in expected},expected)
        for branch in ("native_cold","native_prefix"):
            branch_rows=[record for record in collected if record["branch"]==branch]
            for metric in ("token_f1","exact_match"):
                expected=100*math.fsum(record[metric] for record in branch_rows)/4
                self.assertAlmostEqual(summary["metrics"]["branches"][branch][metric+"_percent"],expected,places=12)
        lookup={(record["id"],record["branch"]):record for record in collected}
        for question in summary["metrics"]["per_question"]:
            for metric in ("token_f1","exact_match"):
                expected=(lookup[(question["id"],"native_prefix")][metric]
                          -lookup[(question["id"],"native_cold")][metric])
                self.assertEqual(question[metric+"_delta"],expected)
        json.dumps(summary,allow_nan=False)
        GROUP_PROTOCOL_OBSERVATIONS.append({"scope":"CPU tiny BF16+FP32 LoRA; synthetic legal protocol fixture, not Qasper results",
                                            "records":collected,"summary":summary})

    def test_natural_histories_eos_and_full_budget(self):
        a,b=ScriptedReader([10,11,0]),ScriptedReader([12,0])
        left,_,left_observation=generate_fake(a)
        right,_,_=generate_fake(b)
        self.assertEqual(left["generated_ids"],[10,11,0])
        self.assertEqual(right["generated_ids"],[12,0])
        self.assertEqual(a.inputs,[10,11]);self.assertEqual(b.inputs,[12])
        self.assertEqual(left["finish_reason"],"eos")
        self.assertEqual(left["eos_token_id"],0)
        self.assertTrue(left_observation["request_state_released"])
        immediate=ScriptedReader([0])
        result,_,observation=generate_fake(immediate)
        self.assertEqual(result["generated_ids"],[0]);self.assertEqual(immediate.inputs,[])
        self.assertEqual(observation["decode_calls"],0)
        guards=[]
        limited=ScriptedReader([10])
        result,_,observation=generate_fake(limited,guard=lambda:guards.append(True))
        self.assertEqual(len(result["generated_ids"]),128)
        self.assertEqual(len(limited.inputs),127)
        self.assertEqual(result["finish_reason"],"max_new_tokens")
        self.assertIsNone(result["eos_token_id"])
        self.assertEqual(observation["decode_calls"],127)
        self.assertEqual(len(guards),17) # Prefill plus before decode1,9,...121.

    def test_real_two_questions_fp32_and_bf16_lora_free_answers(self):
        for precision in ("fp32","bf16"):
            with self.subTest(precision=precision):
                native,prefix,sink,docs=setup(precision)
                emitted=[]
                with precision_context(prefix,precision):
                    records=run_native_prefix_group(native,prefix,sink,docs,rows(),Tokenizer(),
                        stop_ids={0},max_new_tokens=6,on_record=emitted.append)
                self.assertEqual(records,emitted)
                self.assertEqual([(r["id"],r["branch"]) for r in records],
                    [(row["id"],branch) for row in rows() for branch in ("native_cold","native_prefix")])
                self.assertEqual([records[i]["cache_observation"]["prefix_cache_hit"] for i in (1,3)],[False,True])
                self.assertEqual((prefix.build_count,prefix.miss_count,prefix.hit_count),(1,1,1))
                for record in records:
                    self.assertEqual(record["status"],"complete")
                    self.assertTrue(record["cache_observation"]["request_state_released"])
                    self.assertTrue(record["cache_observation"]["query_private"])
                    self.assertTrue(record["cache_observation"]["prefix_unchanged"])
                    self.assertTrue(record["shared_writer_hj_unchanged"])
                    self.assertFalse(record["formal_inference_timing"])
                    self.assertFalse(record["formal_inference_memory"])
                    self.assertEqual(record["cache_observation"]["persistent_kv_heads"],8)
                    self.assertGreaterEqual(len(record["generated_ids"]),1)
                for record in (records[1],records[3]):
                    self.assertFalse(record["first_logit_comparison"]["numeric_gate_applied"])
                json.dumps(records,allow_nan=False)
                QUALITY_OBSERVATIONS.extend({"precision":precision,**record} for record in records)

    def test_singleton_and_next_group_each_start_with_miss(self):
        native,prefix,sink,docs=setup()
        with precision_context(prefix):
            for index in (0,1):
                records=run_native_prefix_group(native,prefix,sink,docs,rows()[index:index+1],Tokenizer(),
                    stop_ids={0},max_new_tokens=2)
                cache=records[1]["cache_observation"]
                self.assertFalse(cache["prefix_cache_hit"])
                self.assertEqual(cache["build_count_after"]-cache["build_count_before"],1)
                self.assertEqual(cache["hit_count_after"]-cache["hit_count_before"],0)
            self.assertEqual(prefix.build_count,2)
            self.assertEqual(prefix.miss_count,2)
            self.assertEqual(prefix.hit_count,0)

    def test_mismatched_prepared_pack_rejected_before_generation(self):
        native,prefix,sink,docs=setup()
        for field in ("document_chunks","selected_chunk_indices","document_id"):
            changed=rows()
            if field=="document_chunks":changed[1][field][0][0]=99
            elif field=="selected_chunk_indices":changed[1][field]=[1,0,2]
            else:changed[1][field]="different-document"
            with self.subTest(field=field),precision_context(prefix):
                with self.assertRaises(RuntimeError):
                    run_native_prefix_group(native,prefix,sink,docs,changed,Tokenizer(),stop_ids={0},max_new_tokens=2)
        self.assertEqual(prefix.build_count,0)

    def test_failure_emits_failed_branch_with_partial_generation(self):
        native,prefix,sink,docs=setup()
        emitted=[]
        original=prefix.prefill
        def nonfinite(*args,**kwargs):
            logits,state=original(*args,**kwargs)
            return logits*float("nan"),state
        with precision_context(prefix),patch.object(prefix,"prefill",side_effect=nonfinite):
            with self.assertRaises(FloatingPointError):
                run_native_prefix_group(native,prefix,sink,docs,rows()[:1],Tokenizer(),
                    stop_ids={0},max_new_tokens=3,on_record=emitted.append)
        self.assertEqual(len(emitted),2)
        self.assertEqual(emitted[0]["status"],"complete")
        self.assertEqual(emitted[1]["status"],"failed")
        self.assertEqual(emitted[1]["branch"],"native_prefix")
        self.assertEqual(emitted[1]["generated_ids"],[])
        self.assertEqual(emitted[1]["error"]["type"],"FloatingPointError")


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    unittest.main(verbosity=2)
