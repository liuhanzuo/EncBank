"""Recovery must never erase partial training or silently retry another failure."""
import unittest
from recover_slurm_training import check_failed_a


class RecoveryTests(unittest.TestCase):
    def state(self):
        return dict(phase='failed', complete=False, arm='A', step=0, cursor=0,
                    raw_tokens=0, training_seconds=0., target_steps=250,
                    error='Slurm controller heartbeat is stale')

    def test_observed_initial_eval_failure_is_eligible(self):
        check_failed_a(self.state(), ['status.json','eval_step0.records.jsonl'])

    def test_partial_training_or_other_failure_is_rejected(self):
        for key, value in [('step',1),('cursor',1),('raw_tokens',1),('training_seconds',1.),
                           ('arm','B'),('error','CUDA out of memory'),('target_steps',50)]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_failed_a({**self.state(),key:value}, [])

    def test_checkpoint_or_train_log_is_rejected(self):
        for name in ['last.pt','step50.pt','train.jsonl']:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                check_failed_a(self.state(),[name])


if __name__ == '__main__':
    unittest.main()
