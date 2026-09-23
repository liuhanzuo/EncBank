"""Actual inference function on a CPU fake reader: EOS and output-budget accounting."""
import unittest
from unittest.mock import patch
import torch
import worker
class FakeReader:
    def __init__(self,stop):self.stop=stop;self.calls=0
    def write_prefill(self,q):return None,None,len(q)
    def read_prefill(self,s,h,q):return torch.tensor([[[1.,3.,9.]]]),None,7
    def decode_step(self,*args):
        self.calls+=1
        return torch.tensor([[[1.,3.,9.]]]) if self.stop else torch.tensor([[[1.,9.,3.]]])
class Tests(unittest.TestCase):
    def test_natural_eos_not_fixed_length(self):
        r=FakeReader(True)
        with patch.object(worker,'stamp',return_value=10.):out=worker.forward(r,None,[],[1,2],2,128)
        self.assertEqual(out[0],[1]);self.assertTrue(out[3]);self.assertEqual(r.calls,1)
    def test_cap_and_first_token_only(self):
        r=FakeReader(False)
        with patch.object(worker,'stamp',return_value=10.):out=worker.forward(r,None,[],[1],2,3)
        self.assertEqual(out[0],[1,1,1]);self.assertFalse(out[3]);self.assertEqual(r.calls,2)
        r=FakeReader(False)
        with patch.object(worker,'stamp',return_value=10.):out=worker.forward(r,None,[],[1],2,1)
        self.assertEqual(len(out[0]),1);self.assertEqual(r.calls,0)
if __name__=='__main__':unittest.main()
