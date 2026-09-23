import unittest
from adaptive import nucleus,fixed_topk

class TestNucleus(unittest.TestCase):
    def test_minimal_mass_and_restore_document_order(self):
        r=nucleus([.1,.6,.3],p=.8,min_chunks=0,max_chunks=3)
        self.assertEqual(r['selected'],[1,2]);self.assertAlmostEqual(r['achieved_mass'],.9);self.assertTrue(r['target_met'])
    def test_flat_scores_cap_reports_unmet_target(self):
        r=nucleus([1]*64,p=.9,min_chunks=4,max_chunks=48)
        self.assertEqual(len(r['selected']),48);self.assertEqual(r['achieved_mass'],.75);self.assertTrue(r['cap_limited']);self.assertFalse(r['target_met'])
    def test_floor_deterministic_ties_and_scale_invariance(self):
        a=nucleus([9,1,0,0,0],p=.8,min_chunks=4,max_chunks=48)
        b=nucleus([900,100,0,0,0],p=.8,min_chunks=4,max_chunks=48)
        self.assertEqual(a,b);self.assertEqual(a['selected'],[0,1,2,3])
    def test_zero_evidence_is_explicit(self):
        r=nucleus([0]*8,p=.9,min_chunks=4,max_chunks=48)
        self.assertEqual(r['selected'],[4,5,6,7]);self.assertTrue(r['zero_mass']);self.assertFalse(r['target_met'])
    def test_p_one_and_candidate_ids(self):
        r=nucleus([2,3,0],p=1,min_chunks=0,max_chunks=48,ids=[10,4,20])
        self.assertEqual(r['selected'],[4,10]);self.assertAlmostEqual(r['achieved_mass'],1)
        self.assertEqual(fixed_topk([2,3,0],2,[10,4,20]),[4,10])
    def test_invalid_scores_do_not_silently_pass(self):
        for scores in [[-1,2],[float('nan')],[float('inf')]]:
            with self.assertRaises(AssertionError):nucleus(scores)
if __name__=='__main__':unittest.main()
