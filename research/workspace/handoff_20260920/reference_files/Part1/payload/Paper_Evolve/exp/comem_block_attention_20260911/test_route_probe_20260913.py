"""Remote-CPU causal and instrumentation validation; no benchmark claims."""
import os
import unittest
import torch
from test_sparse_reader import fixture
from route_probe_20260913 import reader_type, variants, compare_tensors


class ProbeSemantics(unittest.TestCase):
    def test_controls_separate_shape_and_content(self):
        v = dict(variants([20,21,22], [30,31,32,33], 101))
        self.assertEqual(v['prompt_1'], v['prompt_2'])
        self.assertEqual(v['teacher_original'], [20,21,22,30,31,32])
        self.assertEqual(len(v['teacher_original']),len(v['teacher_same_length_replaced']))
        self.assertNotEqual(v['teacher_original'][3:],v['teacher_same_length_replaced'][3:])
        self.assertLess(len(v['teacher_half_length']),len(v['teacher_original']))
        self.assertGreater(len(v['teacher_double_length']),len(v['teacher_original']))

    def test_instrumentation_full_route_and_causal_controls(self):
        for mode in ('block','dense'):
            with self.subTest(mode=mode), torch.no_grad():
                model, base, plain, sink, docs, _ = fixture(mode,.5)
                audited = reader_type()(base, fusion_layer=4, retain_ratio=.5, probe_mode=mode,
                    gradient_checkpointing=False).eval()
                prompt, answer, probes = [21,22,23], [31,32,33], [0,2]
                full = plain.forward_answer(sink,docs,prompt,answer,probe_indices=probes)
                result, saved = audited.probe(sink,docs,prompt+answer[:-1],len(prompt),probes)
                self.assertEqual(result['route']['block_scores'],full['route_stats']['block_scores'])
                self.assertEqual(result['route']['selected_indices'],full['route_stats']['selected_indices'])
                quiet,_ = audited.probe(sink,docs,prompt+answer[:-1],len(prompt),probes,False)
                self.assertEqual(quiet['route'],result['route'])
                replaced, other = audited.probe(sink,docs,prompt+[40,41],len(prompt),probes)
                self.assertEqual(replaced['route'],result['route'])
                self.assertTrue(all(x['exact'] for x in compare_tensors(saved,other).values()))
                shortest, shorter = audited.probe(sink,docs,prompt,len(prompt),probes)
                torch.testing.assert_close(torch.tensor(shortest['route']['block_scores']),
                    torch.tensor(result['route']['block_scores']),atol=2e-7,rtol=2e-6)
                scopes=[x for x in result['events'] if x['op']=='score_scope']
                self.assertEqual(len(scopes),2)
                self.assertTrue(all(x['own_key_scan_stop']==len(prompt) for x in scopes))
                masks=[x for x in result['events'] if x['op']=='masked_fill_']
                self.assertTrue(masks)
                self.assertEqual(masks[0]['mask'],[[[[[False,True,True],[False,False,False]]]]])
                matmuls=[x for x in result['events'] if x['op']=='matmul']
                self.assertTrue(matmuls)
                self.assertTrue(all(x['output_dtype']=='torch.float32' for x in matmuls))

    def test_probe_answer_rows_rejected(self):
        with torch.no_grad():
            _, base, _, sink, docs, _=fixture()
            net=reader_type()(base,fusion_layer=4,retain_ratio=.5,gradient_checkpointing=False).eval()
            with self.assertRaises(ValueError):
                net.probe(sink,docs,[21,22,23,31],3,[3])


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Validation requires explicitly hidden CUDA devices')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    unittest.main(verbosity=2)
