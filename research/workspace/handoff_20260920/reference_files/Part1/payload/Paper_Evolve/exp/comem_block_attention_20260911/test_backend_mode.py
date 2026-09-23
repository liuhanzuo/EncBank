"""No-model checks for diagnostic isolation and the backend rollout boundary."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import infra_sparse as worker
import launch_sparse_infra as launcher
import run_backend_comparison as pipeline
from infra_protocol import GIB, GPU_NAME


class BackendModeChecks(unittest.TestCase):
    def test_diagnostic_and_formal_commands_are_separate(self):
        commands = dict(pipeline.make_commands(SimpleNamespace(python=Path('python'), out=Path('out'), run=False)))
        self.assertEqual(set(commands), {'diagnostic', 'decode_v2', 'backend_v3', 'native'})
        for stage, command in commands.items():
            self.assertIn('--validate-backend', command)
            self.assertNotIn('--run', command)
            self.assertEqual('--profile-reader-only' in command, stage == 'diagnostic')
            self.assertEqual('--backend-model-parity' in command, stage == 'diagnostic')
            self.assertEqual(command[command.index('--generation-tokens') + 1], '4' if stage == 'diagnostic' else '32')
            self.assertEqual(command[command.index('--repetitions') + 1], '1' if stage == 'diagnostic' else '3')

    def test_candidate_cannot_bypass_tiny_gpu_validation(self):
        with patch.object(launcher, 'read_json', side_effect=AssertionError('model read')):
            with self.assertRaisesRegex(ValueError, 'requires guarded tiny'):
                launcher.main(['--reader-implementation', 'backend_v3'])

    def test_model_diagnostic_is_not_permitted_in_formal_timing(self):
        with patch.object(launcher, 'read_json', side_effect=AssertionError('model read')):
            with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
                launcher.main(['--reader-implementation', 'backend_v3', '--validate-backend', '--backend-model-parity'])

    def test_missing_profiles_do_not_unlock_formal_run(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Three completed'):
                pipeline.check_profiles(Path(temp))

    def test_cpu_validation_requires_remote_hidden_cuda_and_matching_source(self):
        receipt = dict(passed=False, device='cpu', cuda_visible_devices='', tests_run=1,
                       same_backend_contract=dict(passed=True, tests_run=1, skipped=0), same_math_backend_bitwise_passed=True,
                       skipped=0, source_sha256={'backend_sparse_reader.py': 'expected'})
        with patch.object(pipeline, 'read_json', return_value=receipt), patch.object(pipeline, 'digest', return_value='expected'):
            self.assertEqual(pipeline.check_cpu_receipt(Path('stub')), receipt)
        for changed in (dict(device='cuda'), dict(same_backend_contract=dict(passed=True, skipped=1)),
                        dict(cuda_visible_devices='0'), dict(same_math_backend_bitwise_passed=False)):
            with self.subTest(changed=changed), patch.object(pipeline, 'read_json', return_value=receipt | changed):
                with self.assertRaises(ValueError):
                    pipeline.check_cpu_receipt(Path('stub'))
        with patch.object(pipeline, 'read_json', return_value=receipt), patch.object(pipeline, 'digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'source differs'):
                pipeline.check_cpu_receipt(Path('stub'))

    def test_parser_defaults_preserve_old_recipe(self):
        args = launcher.parser().parse_args([])
        self.assertEqual(args.reader_implementation, 'reference')
        self.assertFalse(args.validate_backend)
        self.assertFalse(args.backend_model_parity)

    def test_profile_backend_evidence_and_numerical_failure_block_formal_stage(self):
        cases = ('n4096_q64_g4_D0_cold_hj', 'n4096_q64_g4_B_cold_hj', 'n4096_q64_g4_B_block_hot')
        profile = dict(status='complete', source_sha256='profiler', summary_path='summary.stub',
                       cuda_timing_status='observed_device_kernel_timeline', decode_steps_completed=3,
                       generated_ids=[1, 2, 3, 4], trace_evidence=dict(cuda_kernel_timeline_observed=True),
                       backend_operator_evidence=[
                           dict(name='aten::_scaled_dot_product_efficient_attention', count=108, input_shapes=[[1, 32, 1, 128]]),
                           dict(name='aten::_scaled_dot_product_efficient_attention', count=172, input_shapes=[[1, 32, 64, 128]])])
        ident = dict(reader_implementation='backend_v3', profile_reader_only=True, backend_model_parity=True,
                     profiler_sha256='profiler', infra_launcher_sha256='launcher', monitor_policy_version='policy')
        result = dict(status='complete', identity=ident, profiler_receipt=profile, pid=77,
                      backend_gpu_validation=dict(passed=False, same_backend_contract=dict(passed=True),
                                                  same_math_backend_bitwise_passed=True), backend_model_parity=dict(passed=True),
                      hardware=dict(gpu=GPU_NAME), supervisor_lease=dict(worker='fixture'))
        monitor = dict(status='complete', exit_code=0, samples=10, peak_incremental_gpu_bytes=20*GIB,
                       baseline_gpu_used_bytes=2*GIB, worker_lease=result['supervisor_lease'], actual_worker_pid=77,
                       launcher_sha256='launcher', monitor_policy_version='policy')
        state = dict(status='complete', jobs={c:dict(status='complete', attempt=c) for c in cases})
        def read(path, default=None):
            return {'status.json':state, 'result.json':result, 'monitor.json':monitor, 'summary.stub':profile}[Path(path).name]
        with patch.object(pipeline, 'read_json', side_effect=read), patch.object(pipeline, 'save_json'):
            self.assertTrue(pipeline.check_profiles(Path('fixture'))['passed'])
            result['backend_model_parity']['passed'] = False
            with self.assertRaisesRegex(ValueError, 'failed numerical parity'):
                pipeline.check_profiles(Path('fixture'))
            result['backend_model_parity']['passed'] = True
            profile['backend_operator_evidence'].append(dict(name='aten::_scaled_dot_product_attention_math', count=1))
            with self.assertRaisesRegex(ValueError, 'fused backend was not established'):
                pipeline.check_profiles(Path('fixture'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
