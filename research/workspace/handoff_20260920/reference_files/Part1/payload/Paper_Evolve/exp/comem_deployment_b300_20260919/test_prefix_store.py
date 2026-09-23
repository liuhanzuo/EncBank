import unittest
from prefix_store import PrefixStore

class TestPrefix(unittest.TestCase):
    def test_no_cross_context_reuse(self):
        s=PrefixStore(20)
        for parts in [('sink',),('sink','a'),('sink','a','b')]:self.assertTrue(s.put(parts,lambda:object(),2,1))
        self.assertEqual(len(s.get_chain(['sink','a','b'])),3)
        self.assertEqual(len(s.get_chain(['sink','b','a'])),1)
        self.assertEqual(len(s.get_chain(['different','a','b'])),0)
    def test_leaf_eviction_keeps_reachable_prefixes(self):
        s=PrefixStore(6)
        for parts in [('s',),('s','a'),('s','a','b')]:s.put(parts,lambda:object(),2,1)
        s.put(('s','x'),lambda:object(),2,1)
        self.assertEqual(s.bytes,6);self.assertEqual(s.evictions,1)
        self.assertEqual(len(s.get_chain(['s','a','b'])),2)
        self.assertEqual(len(s.get_chain(['s','x'])),2)
        s.put(('s','x','y'),lambda:object(),2,1)
        self.assertEqual(len(s.get_chain(['s','x','y'])),3)
        self.assertEqual(len(s.get_chain(['s','a'])),1)
    def test_protect_current_ancestors_and_no_reallocate_hit(self):
        s=PrefixStore(4);calls=[]
        def create():calls.append(True);return object()
        self.assertTrue(s.put(('s',),create,2,1));self.assertTrue(s.put(('s','a'),create,2,1))
        self.assertFalse(s.put(('s','a','b'),create,2,1))
        self.assertTrue(s.put(('s','a'),create,2,1));self.assertEqual(len(calls),2)
        self.assertEqual(s.bytes,4)

if __name__=='__main__':unittest.main()
