"""CPU tiny-Qwen semantic/storage tests; no production timing or GPU claim.

The native causal graph is unchanged, but splitting document/query matmuls can
change floating-point rounding. Fixed tolerances are declared below; numerical
observations (including greedy changes) are retained, never rounded to equality.
"""
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import weakref

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from native_infra_readers import NativeCoMemReader
from native_prefix_reader import NativePrefixCoMemReader,_tensor_signature
from test_same_math_diagnostic import fixture


NUMERICAL_OBSERVATIONS=[]
TOLERANCES={"fp32":(3e-6,3e-5),"bf16":(.02,.02)}


@contextmanager
def precision_context(reader,precision="fp32"):
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH),torch.autocast(
            reader.comem.device.type,dtype=torch.bfloat16,enabled=precision=="bf16"):
        yield


def setup(precision="fp32"):
    previous,_,sink,docs=fixture(precision)
    if previous.device.type!="cpu":
        raise RuntimeError("This prototype suite is CPU-only; GPU execution is not authorized here")
    return NativeCoMemReader(previous.comem),NativePrefixCoMemReader(previous.comem),sink,docs


class NativePrefixTests(unittest.TestCase):
    def compare(self,left,right,precision,label,*,logits=False):
        error=right.detach().double()-left.detach().double()
        row={"precision":precision,"label":label,"shape":list(left.shape),
             "reference_dtype":str(left.dtype),"candidate_dtype":str(right.dtype),
             "max_abs":float(error.abs().max()),"rms":float(error.square().mean().sqrt()),
             "bitwise_equal":bool(torch.equal(left.view(torch.uint8),right.contiguous().view(torch.uint8)))}
        if logits:
            row["greedy_equal"]=bool(torch.equal(left.argmax(-1),right.argmax(-1)))
            row["reference_greedy"]=left.argmax(-1).tolist()
            row["candidate_greedy"]=right.argmax(-1).tolist()
        NUMERICAL_OBSERVATIONS.append(row)
        atol,rtol=TOLERANCES[precision]
        torch.testing.assert_close(left,right,atol=atol,rtol=rtol)

    def compare_state(self,reference,state,reader,precision,label):
        prefix=state.top_cache.prefix
        self.assertEqual((reference.query_position,reference.pack_position),
                         (state.query_position,state.pack_position))
        for i in range(reader.j):
            for field in ("keys","values"):
                self.compare(getattr(reference.bottom_cache.layers[i],field),
                             getattr(state.bottom_cache.layers[i],field),precision,f"{label}/bottom{i}/{field}")
        for i in range(reader.j,reader.L):
            own=state.top_cache.layers[i]
            for field,short in (("keys","k"),("values","v")):
                expected=getattr(reference.top_cache.layers[i],field)
                query=getattr(own,field)
                self.assertEqual(query.shape[1],8)
                self.assertEqual(query.shape[-2],state.query_position)
                self.assertEqual(query.untyped_storage().nbytes(),query.numel()*query.element_size())
                self.compare(expected[...,prefix.token_count:,:],query,precision,f"{label}/query{i}/{field}")
                if prefix.token_count:
                    document=getattr(prefix.pairs[i],short)
                    self.assertEqual(document.shape[1],8)
                    self.assertNotEqual(query.untyped_storage().data_ptr(),document.untyped_storage().data_ptr())
                    self.compare(expected[...,:prefix.token_count,:],document,precision,f"{label}/doc{i}/{field}")

    def test_two_questions_native_same_math_and_readonly_prefix(self):
        for precision in ("fp32","bf16"):
            with self.subTest(precision=precision):
                native,reader,sink,docs=setup(precision)
                with precision_context(reader,precision):
                    prefix=reader.build_prefix(sink,docs)
                    saved={i:(pair.k.clone(),pair.v.clone(),_tensor_signature(pair.k),_tensor_signature(pair.v))
                           for i,pair in prefix.pairs.items()}
                    requests=[]
                    for question,prompt in enumerate(([21,22,23],[24,25,26,27,28])):
                        left,reference=native.prefill(sink,docs,prompt)
                        right,state=reader.prefill(sink,docs,prompt)
                        self.compare(left,right,precision,f"q{question}/prefill/logits",logits=True)
                        self.compare_state(reference,state,reader,precision,f"q{question}/prefill")
                        self.assertIs(state.top_cache.prefix,prefix)
                        self.assertTrue(state.route_stats["prefix_cache_hit"])
                        self.assertEqual(state.route_stats["prefix_build_layer_calls_this_request"],0)
                        self.assertEqual(state.route_stats["query_online_document_layer_calls"],0)
                        requests.append((reference,state))
                    self.assertIsNot(requests[0][1].top_cache,requests[1][1].top_cache)
                    for step,token in enumerate((31,32,33),1):
                        for question,(reference,state) in enumerate(requests):
                            other=requests[1-question][1]
                            before=[_tensor_signature(layer.keys) for layer in other.top_cache.layers[reader.j:]]
                            left=native.decode_step(token,reference)
                            right=reader.decode_step(token,state)
                            self.compare(left,right,precision,f"q{question}/decode{step}/logits",logits=True)
                            self.compare_state(reference,state,reader,precision,f"q{question}/decode{step}")
                            self.assertEqual(before,[_tensor_signature(layer.keys) for layer in other.top_cache.layers[reader.j:]])
                    for i,pair in prefix.pairs.items():
                        k,v,ksig,vsig=saved[i]
                        self.assertTrue(torch.equal(k,pair.k));self.assertTrue(torch.equal(v,pair.v))
                        self.assertEqual(ksig,_tensor_signature(pair.k));self.assertEqual(vsig,_tensor_signature(pair.v))
                    self.assertEqual(reader.build_count,1)
                    self.assertEqual(reader.hit_count,2)

    def test_native_layer_call_lengths_and_actual_storage_accounting(self):
        _,reader,sink,docs=setup()
        calls=[]
        hooks=[layer.register_forward_pre_hook(lambda module,args,index=i:calls.append((index,args[0].shape[1])))
               for i,layer in enumerate(reader.comem.layers)]
        try:
            with precision_context(reader):
                prefix=reader.build_prefix(sink,docs)
                self.assertEqual(calls,[(i,10) for i in range(reader.j,reader.L)])
                self.assertEqual(prefix.build_layer_calls,reader.L-reader.j)
                pairs=[t for pair in prefix.pairs.values() for t in (pair.k,pair.v)]
                self.assertEqual(prefix.prefix_kv_bytes,sum(t.numel()*t.element_size() for t in pairs))
                self.assertEqual(prefix.prefix_storage_bytes,prefix.prefix_kv_bytes)
                self.assertEqual(prefix.hj_tensor_bytes,sum(t.numel()*t.element_size() for t in [sink]+docs))
                self.assertEqual(prefix.hj_storage_bytes,prefix.hj_tensor_bytes)
                calls.clear()
                _,state=reader.prefill(sink,docs,[21,22,23])
                self.assertEqual(calls,[(i,3) for i in range(reader.L)])
                calls.clear()
                reader.decode_step(31,state)
                self.assertEqual(calls,[(i,1) for i in range(reader.L)])
                expected=sum(t.numel()*t.element_size() for layer in state.top_cache.layers[reader.j:]
                             for t in (layer.keys,layer.values))
                self.assertEqual(state.route_stats["request_upper_query_kv_bytes"],expected)
        finally:
            for hook in hooks:hook.remove()

    def test_order_identity_source_and_adapter_changes_miss(self):
        _,reader,sink,docs=setup()
        with precision_context(reader):
            def request(expected_count):
                _,state=reader.prefill(sink,docs,[21,22,23])
                self.assertEqual(reader.build_count,expected_count)
                self.assertFalse(state.route_stats["prefix_cache_hit"])
                return state
            request(1)
            docs[:]=docs[::-1];request(2)
            docs[0]=docs[0].clone();request(3) # Equal content, intentionally conservative miss.
            docs[0].add_(.001);request(4)
            sink.add_(.001);request(5)
            lora=next(module for module in reader.model.modules() if hasattr(module,"B"))
            lora.B.add_(.001);request(6)
            lora.scale+=.01;request(7)
            parameter=next(reader.model.parameters());parameter.add_(.001);request(8)
            # This mutation deliberately bypasses versioning: explicit invalidate is required.
            docs[0].data.add_(.001);reader.invalidate();request(9)

    def test_corrupted_prefix_and_changed_active_request_rejected(self):
        _,reader,sink,docs=setup()
        with precision_context(reader):
            _,old=reader.prefill(sink,docs,[21,22,23])
            reader.prefix.pairs[reader.j].k.add_(.001)
            with self.assertRaises(RuntimeError):reader.decode_step(31,old)
            _,new=reader.prefill(sink,docs,[21,22,23])
            self.assertEqual(reader.build_count,2)
            self.assertIsNot(old.top_cache.prefix,new.top_cache.prefix)
            docs[0].add_(.001)
            positions=(new.query_position,new.pack_position)
            with self.assertRaises(RuntimeError):reader.decode_step(31,new)
            self.assertEqual(positions,(new.query_position,new.pack_position))
            _,current=reader.prefill(sink,docs,[21,22,23])
            reader.invalidate()
            with self.assertRaises(RuntimeError):reader.decode_step(31,current)

    def test_miss_releases_unowned_prefix_but_preserves_active_request(self):
        _,reader,sink,docs=setup()
        with precision_context(reader):
            old=weakref.ref(reader.build_prefix(sink,docs))
            original=reader.comem._run_layers
            observed=[]
            def check_before_build(*args,**kwargs):
                observed.append(old() is None)
                return original(*args,**kwargs)
            with patch.object(reader.comem,"_run_layers",side_effect=check_before_build):
                reader.build_prefix(sink,list(reversed(docs)))
            self.assertTrue(observed and all(observed))
            # A miss through prefill also drops both the local and reader references.
            old=weakref.ref(reader.prefix)
            observed.clear()
            with patch.object(reader.comem,"_run_layers",side_effect=check_before_build) as recorder:
                _,active=reader.prefill(sink,docs,[21,22,23])
            self.assertTrue(observed and all(observed))
            # MagicMock saves kwargs and thus owns this request's PrefixQueryCache.
            # Prove and remove that instrumentation reference before testing the
            # reader's ownership; no gc.collect() is used to hide a real cycle.
            self.assertTrue(any(call.kwargs.get("past_key_values") is active.top_cache
                                for call in recorder.call_args_list))
            recorder.reset_mock()
            self.assertEqual(recorder.call_args_list,[])
            self.assertIsNone(recorder.call_args)
            active_prefix=weakref.ref(active.top_cache.prefix)
            reader.build_prefix(sink,list(reversed(docs)))
            self.assertIsNotNone(active_prefix())
            self.assertIsNot(reader.prefix,active_prefix())
            reader.decode_step(31,active)
            active.top_cache.prefix.assert_intact()
            del active
            self.assertIsNone(active_prefix())

    def test_window_offset_eval_and_dynamic_rope_fail_before_layer_calls(self):
        _,reader,sink,docs=setup()
        with precision_context(reader):
            with self.assertRaises(ValueError):reader.prefill(sink,docs,[21],position_offset=1)
            with self.assertRaises(ValueError):reader.prefill(sink,docs,[])
            with self.assertRaises(ValueError):reader.prefill(sink,docs,[21]*247)
            _,state=reader.prefill(sink,docs,[21]*246)
            self.assertEqual(state.pack_position,256)
            before=[_tensor_signature(layer.keys) for layer in state.top_cache.layers[reader.j:]]
            with self.assertRaises(ValueError):reader.decode_step(31,state)
            self.assertEqual(before,[_tensor_signature(layer.keys) for layer in state.top_cache.layers[reader.j:]])
            reader.model.train()
            with self.assertRaises(ValueError):reader.prefill(sink,docs,[21])
            reader.model.eval()
            reader.comem.rotary_emb.rope_type="dynamic"
            with self.assertRaises(ValueError):reader.prefill(sink,docs,[21])

    def test_empty_prefix_has_no_document_calls_and_matches_native(self):
        native,reader,_,_=setup()
        with precision_context(reader):
            prefix=reader.build_prefix(None,[])
            self.assertEqual(prefix.token_count,0)
            self.assertEqual(prefix.prefix_storage_bytes,0)
            self.assertEqual(prefix.build_layer_calls,0)
            left,a=native.prefill(None,[],[21,22,23])
            right,b=reader.prefill(None,[],[21,22,23])
            self.compare(left,right,"fp32","empty/prefill/logits",logits=True)
            for token in (31,32):
                self.compare(native.decode_step(token,a),reader.decode_step(token,b),"fp32","empty/decode/logits",logits=True)
                self.compare_state(a,b,reader,"fp32","empty/decode")


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    unittest.main(verbosity=2)
