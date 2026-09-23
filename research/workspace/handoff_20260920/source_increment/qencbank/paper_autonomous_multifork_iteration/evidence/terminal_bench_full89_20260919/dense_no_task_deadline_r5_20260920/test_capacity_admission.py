import unittest
from capacity_admission import admissible,reservation

class CapacityTests(unittest.TestCase):
    def fit(self,counts):
        active=used=0
        for n in counts:
            if admissible(active,used,n,32,2**21):active+=1;used+=n
        return active,used
    def test_long_context_cannot_overbook(self):
        self.assertEqual(self.fit([2**18]*9),(8,2**21))
    def test_variable_context_budget(self):
        self.assertEqual(self.fit([2**17]*17),(16,2**21))
        self.assertEqual(self.fit([2**16]*33),(32,2**21))
        self.assertEqual(self.fit([2**18]+[2**16]*31),(29,2**21))
    def test_generation_budget_included(self):
        self.assertEqual(reservation(32768,262144,32768),65536)
        self.assertEqual(reservation(250000,262144,32768),262144)
        self.assertEqual(reservation(262144,262144,32768),1)
    def test_empty_and_released_capacity(self):
        self.assertTrue(admissible(0,0,262144,32,2**21))
        self.assertFalse(admissible(8,2**21,1,32,2**21))
        self.assertTrue(admissible(7,2**21-262144,262144,32,2**21))
if __name__=='__main__':unittest.main()
