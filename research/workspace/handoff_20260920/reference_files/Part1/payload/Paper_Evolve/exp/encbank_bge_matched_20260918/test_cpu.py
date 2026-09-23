import ast,unittest
from pathlib import Path
from select_budget import choose
class Tests(unittest.TestCase):
    def test_largest_matching_budget(self):
        self.assertEqual(choose(100,{6:90,8:99,10:104,12:120})['quality_raw_ks'],[10])
    def test_bracketing_without_fake_match(self):
        r=choose(100,{6:80,8:91,10:109,12:130})
        self.assertFalse(r['strict_match_on_calibration']);self.assertEqual(r['quality_raw_ks'],[8,10])
    def test_all_above_no_fabricated_lower(self):
        self.assertEqual(choose(100,{4:120,6:140})['quality_raw_ks'],[4])
    def test_syntax(self):
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
if __name__=='__main__':unittest.main()
