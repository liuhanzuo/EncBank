"""No Torch/GPU: priority and external-process admission regressions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import remote_backend_quality_queue as queue
import remote_gpu_guard as guard


class QualityQueueChecks(unittest.TestCase):
    def state(self, phases):
        owner=dict(pid=17,start_ticks=99,argv=['queue'])
        return dict(controller=owner,
                    jobs={key:dict(phase=p,process=owner) for key,p in zip(sorted(queue.TRAIN_KEYS), phases)})

    def evaluate_priority(self, state, alive=True, valid=True):
        with patch.object(queue,'read_json',return_value=state), \
             patch.object(queue,'process_identity',return_value=state.get('controller') if alive else None), \
             patch.object(queue,'completed_job',return_value=valid):
            return queue.training_priority_clear()

    def test_unknown_failed_or_unassigned_training_blocks_quality(self):
        self.assertFalse(self.evaluate_priority({})[0])
        for phase in ('queued','launching','failed'):
            self.assertFalse(self.evaluate_priority(self.state(['running']*7+[phase]))[0])

    def test_dead_incomplete_controller_blocks_quality(self):
        self.assertFalse(self.evaluate_priority(self.state(['running']*8),alive=False)[0])

    def test_stale_worker_or_unknown_job_keys_blocks_quality(self):
        state=self.state(['running']*8)
        state['jobs']['smoke/D0']['process']=dict(pid=999)
        self.assertFalse(self.evaluate_priority(state)[0])
        state=self.state(['complete']*8)
        state['jobs']['other']=state['jobs'].pop('smoke/D0')
        self.assertFalse(self.evaluate_priority(state)[0])

    def test_failed_binding_cleans_up_new_unreaped_child(self):
        class Child:
            pid=53
            returncode=None
            def poll(self): return self.returncode
            def terminate(self): self.returncode=-15
            def wait(self,timeout): return self.returncode
        child=Child()
        with patch.object(guard,'process_identity',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'Cannot establish'):
                guard.bind_owned_process(child)
        self.assertEqual(child.returncode,-15)

    def test_assigned_training_or_valid_completed_training_permits_idle_gpu_check(self):
        self.assertTrue(self.evaluate_priority(self.state(['running']*4+['complete']*4))[0])
        self.assertTrue(self.evaluate_priority(self.state(['complete']*8),alive=False)[0])
        self.assertFalse(self.evaluate_priority(self.state(['complete']*8),alive=False,valid=False)[0])

    def test_graphics_python_and_unknown_pid_are_interference(self):
        display=dict(pid='3',type='G',process_name='/usr/lib/xorg/Xorg')
        own=dict(pid='8',type='C',process_name='python')
        other=dict(pid='9',type='G',process_name='python')
        row=dict(index=2,name='NVIDIA GeForce RTX 3090',used_mib=19000,eligible=False,
                 processes=[display,own,other])
        self.assertEqual(guard.unexpected_processes([row],2,{8}),[other])
        row['processes']=[display,own]
        self.assertEqual(guard.unexpected_processes([row],2,{8}),[])
        unknown=dict(pid='N/A',type='G',process_name='Xorg')
        row['processes']=[unknown]
        self.assertEqual(guard.unexpected_processes([row],2,{8}),[unknown])

    def test_unknown_inventory_never_becomes_idle(self):
        for row in (dict(index=2,name='RTX 3090',used_mib=None,processes=[],eligible=False),
                    dict(index=2,name='RTX 3090',used_mib=4,processes=[],eligible=False),
                    dict(index=2,name='RTX 3090',used_mib=19000,processes=[],eligible=False)):
            with self.assertRaises(RuntimeError):
                guard.unexpected_processes([row],2,{8})


if __name__ == '__main__':
    unittest.main(verbosity=2)
