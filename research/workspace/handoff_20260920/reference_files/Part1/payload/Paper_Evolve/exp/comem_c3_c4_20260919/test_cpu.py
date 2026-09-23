import ast,unittest
from pathlib import Path
from queue_metrics import closed_loop,summarize,percentile

class Clock:
    def __init__(self):self.t=100.
    def __call__(self):return self.t
    def service(self,ids):
        self.t+=1;first=self.t;self.t+=2
        return first,self.t,dict(ids=ids)

class Tests(unittest.TestCase):
    def test_actual_queue(self):
        clock=Clock();rows,span=closed_loop(clock.service,8,16,4,clock)
        self.assertEqual(span,12);self.assertEqual([r['queue_s'] for r in rows[:8]],[0]*4+[3]*4)
        self.assertEqual(len({r['id'] for r in rows}),16)
        stats=summarize(rows,span,32)
        self.assertAlmostEqual(stats['requests_per_s'],16/12)
        self.assertEqual(stats['queue_s']['p99'],3000)
        self.assertEqual(stats['e2e_s']['p99'],6000)
    def test_single(self):
        clock=Clock();rows,span=closed_loop(clock.service,1,5,4,clock)
        self.assertEqual(span,15);self.assertTrue(all(r['queue_s']==0 for r in rows))
        self.assertEqual(percentile([1,2,3,4],50),2.5)
    def test_tail_batch(self):
        clock=Clock();rows,span=closed_loop(clock.service,16,19,4,clock)
        self.assertEqual(len(rows),19);self.assertEqual(rows[-1]['batch_size'],3)
        self.assertLessEqual(max(r['batch_size'] for r in rows),4)
    def test_syntax(self):
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
