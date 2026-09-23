"""CPU/tiny checks for independent EOS generation and isolated dispatch probes."""
import gc
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from evaluate_backend_quality import (free_generate,profile_generation,select_rows,
                                     visible_prediction,_input_identity,_validate_platform)

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"beacon_encbank_20260909"))
from evaluate_sft import token_f1,exact_match,eos_token_ids


class State:
    def __init__(self):
        self.route_stats={"selected_indices":[0],"block_scores":[.5]}


class FakeReader:
    training=False
    def __init__(self,sequence):
        self.device=torch.device(os.environ.get("SPARSE_TEST_DEVICE","cpu"))
        self.sequence=sequence
        self.cursor=0
        self.inputs=[]
        self.prefills=[]
        self.state_refs=[]
    def _logits(self):
        value=torch.full((1,1,300),-10.,device=self.device)
        value[0,0,self.sequence[min(self.cursor,len(self.sequence)-1)]]=10.
        return value
    def prefill(self,sink,memories,prompt_ids,probe_indices=None):
        self.cursor=0
        self.prefills.append((prompt_ids,probe_indices))
        state=State()
        self.state_refs.append(weakref.ref(state))
        return self._logits(),state
    def decode_step(self,token,state):
        self.inputs.append(token)
        self.cursor+=1
        return self._logits()


class FakeTokenizer:
    eos_token_id=0
    unk_token_id=299
    def convert_tokens_to_ids(self,token):
        return 2 if token=="<|im_end|>" else 299
    def decode(self,ids,skip_special_tokens=True):
        return " "+" ".join(map(str,ids))+" "
    def encode(self,*args,**kwargs):
        raise AssertionError("No retokenization allowed")


class BackendQualityGeneration(unittest.TestCase):
    def test_independent_histories_and_natural_eos(self):
        a,b=FakeReader([10,11,0]),FakeReader([12,0])
        prompt,probe=[42,43],[1]
        ra=free_generate(a,None,[],prompt,probe,stop_ids={0})
        rb=free_generate(b,None,[],prompt,probe,stop_ids={0})
        self.assertEqual(a.inputs,[10,11])
        self.assertEqual(b.inputs,[12])
        self.assertEqual(ra["generated_ids"],[10,11,0])
        self.assertEqual(rb["generated_ids"],[12,0])
        self.assertIs(a.prefills[0][0],prompt)
        self.assertIs(b.prefills[0][1],probe)
        self.assertEqual(visible_prediction(FakeTokenizer(),ra),"10 11")
        self.assertTrue(all(r() is None for r in a.state_refs+b.state_refs))

    def test_eos_first_token_and_exact_budget(self):
        early=FakeReader([0])
        result=free_generate(early,None,[],[9],[0],stop_ids={0})
        self.assertEqual(result["finish_reason"],"eos")
        self.assertEqual(early.inputs,[])
        self.assertEqual(visible_prediction(FakeTokenizer(),result),"")
        long=FakeReader([10])
        result=free_generate(long,None,[],[9],[0],stop_ids={0},max_new_tokens=128)
        self.assertEqual(len(result["generated_ids"]),128)
        self.assertEqual(len(long.inputs),127)
        self.assertEqual(result["finish_reason"],"max_new_tokens")
        self.assertIsNone(result["eos_token_id"])

    def test_nonfinite_failure_preserves_partial_and_releases_state(self):
        reader=FakeReader([10,11,0])
        original=reader.decode_step
        def corrupt(token,state):
            return original(token,state)*float("nan")
        with patch.object(reader,"decode_step",side_effect=corrupt):
            try:
                free_generate(reader,None,[],[9],[0],stop_ids={0})
            except FloatingPointError as exc:
                self.assertEqual(exc.quality_partial["generated_ids"],[10])
            else:
                self.fail("Nonfinite logits passed")
        gc.collect()
        self.assertTrue(all(r() is None for r in reader.state_refs))

    def test_profile_own_state_and_no_decode_after_eos(self):
        for sequence in ([0],[10,0],[10,11,12,13,14,0]):
            with self.subTest(sequence=sequence),tempfile.TemporaryDirectory() as out:
                reader=FakeReader(sequence)
                result=free_generate(reader,None,[],[9],[0],stop_ids={0})
                before=len(reader.prefills)
                diagnostic=profile_generation(reader,None,[],[9],[0],result,out_dir=out)
                self.assertTrue(diagnostic["prefix_equal"])
                self.assertEqual(len(reader.prefills),before+1)
                self.assertNotIn(0,reader.inputs)
                self.assertLessEqual(diagnostic["profile"]["decode_steps_completed"],3)
                gc.collect()
                self.assertTrue(all(r() is None for r in reader.state_refs))

    def test_fixed_hash_order_smoke_full_and_invalid_count(self):
        rows=[{"id":f"id-{i}"} for i in range(99)]
        expected=sorted(rows,key=lambda row:hashlib.sha256(f"42:{row['id']}".encode()).hexdigest())
        self.assertEqual(select_rows(list(reversed(rows)),"smoke"),expected[:8])
        self.assertEqual(select_rows(rows,"full"),expected)
        with self.assertRaises(ValueError):
            select_rows(rows[:98],"full")
        with self.assertRaises(ValueError):
            select_rows(rows,"smoke",43)

    def test_existing_score_and_stop_protocol(self):
        tokenizer=FakeTokenizer()
        model=SimpleNamespace(config=SimpleNamespace(eos_token_id=[1]),
                              generation_config=SimpleNamespace(eos_token_id=[3]))
        self.assertEqual(eos_token_ids(tokenizer,model),{0,1,2,3})
        references=["wrong answer","The red fox."]
        self.assertEqual(max(token_f1("red fox",r) for r in references),1.)
        self.assertEqual(max(exact_match("red fox",r) for r in references),1.)
        with patch("evaluate_backend_quality.sys.platform","win32"):
            with self.assertRaises(RuntimeError):
                _validate_platform(SimpleNamespace(seed=42,max_new_tokens=128))

    def test_real_32q_8kv_bf16_lora_free_generation_and_profile(self):
        from test_same_math_diagnostic import fixture
        from backend_quality_results import aggregate_paired
        ref,new,sink,docs=fixture("bf16")
        initial=_input_identity(sink,docs)
        records=[]
        with torch.no_grad(),torch.autocast(ref.device.type,dtype=torch.bfloat16):
            for branch,reader in (("decode_v2",ref),("backend_v3",new)):
                generated=free_generate(reader,sink,docs,[21,22,23],[0,2],stop_ids={0},max_new_tokens=5)
                self.assertTrue(generated["generated_ids"])
                with tempfile.TemporaryDirectory() as out:
                    diagnostic=profile_generation(reader,sink,docs,[21,22,23],[0,2],generated,out_dir=out)
                self.assertTrue(diagnostic["prefix_equal"])
                self.assertTrue(diagnostic["profile"]["backend_operator_evidence"])
                self.assertEqual(_input_identity(sink,docs),initial)
                prediction=visible_prediction(FakeTokenizer(),generated)
                records.append({"schema":"backend-quality-record-v1","run_id":"cpu-integration",
                    "recipe_sha256":"a"*64,"source_sha256":{"test":"b"*64},"ordinal":0,"id":"tiny",
                    "document_id":"tiny-document","source":"tiny-generated-fixture","branch":branch,
                    "status":"complete","references":["43"],"prediction":prediction,
                    "token_f1":token_f1(prediction,"43"),"exact_match":exact_match(prediction,"43"),
                    "selected_chunk_indices":[0,1,2],"prompt_ids":[21,22,23],"probe_indices":[0,2],
                    "candidate_tokens":9,**generated})
        summary=aggregate_paired(records,["tiny"],run_id="cpu-integration",
                                 recipe_sha256="a"*64,source_sha256={"test":"b"*64})
        self.assertEqual(summary["status"],"complete",summary)


if __name__=="__main__":
    unittest.main(verbosity=2)
