"""CPU/stdlib-only checks for numerical diagnostic scope and source admission."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import infra_sparse as worker
import launch_sparse_infra as launcher
import same_math_protocol as protocol
from compare_decode_infra import collect


class SameMathModeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipt = self.root / 'cpu.json'
        self.data = dict(passed=True, device='cpu', cuda_visible_devices='', tests_run=2,
            failures=0, errors=0, skipped=0,
            source_sha256={n: 'stub:' + n for n in protocol.REQUIRED_CPU_SOURCES})
        self.write(self.receipt, self.data)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def argv(self):
        return ['--same-math-diagnostic-only', '--same-math-cpu-receipt', str(self.receipt),
                '--lengths', '4096', '--arms', 'D0', '--modes', 'cold_hj',
                '--generation-tokens', '4', '--reader-implementation', 'backend_v3', '--validate-backend']

    def test_one_bounded_diagnostic_and_remote_cpu_contract_required(self):
        args = launcher.parser().parse_args(self.argv())
        with patch.object(protocol, 'digest', side_effect=lambda p: 'stub:' + Path(p).name):
            self.assertEqual(protocol.validate_same_math_mode(args), self.data)
        for changed in (dict(passed=False), dict(device='cuda'), dict(cuda_visible_devices='0'),
                        dict(errors=1), dict(skipped=1), dict(source_sha256={})):
            self.write(self.receipt, self.data | changed)
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    protocol.validate_same_math_mode(args)

    def test_modified_source_does_not_pass_cpu_receipt(self):
        args = launcher.parser().parse_args(self.argv())
        with patch.object(protocol, 'digest', return_value='modified'):
            with self.assertRaisesRegex(ValueError, 'source differs'):
                protocol.validate_same_math_mode(args)

    def test_shared_state_mode_cannot_be_combined_with_fixed_math(self):
        args = launcher.parser().parse_args(self.argv() + ['--shared-state-diagnostic-only'])
        with self.assertRaisesRegex(ValueError, 'restricted'):
            protocol.validate_same_math_mode(args)

    def test_shape_mode_and_profile_mixing_are_rejected_before_model_gpu_io(self):
        for extra in (['--lengths','16384'], ['--arms','B'], ['--modes','block_hot'],
                      ['--profile-reader-only'], ['--backend-model-parity'], ['--repetitions','3'],
                      ['--generation-tokens','32'], ['--prompt-tokens','128'], ['--j','13']):
            with self.subTest(extra=extra), patch.object(launcher, 'read_json', side_effect=AssertionError('model IO')):
                with self.assertRaisesRegex(ValueError, 'restricted|diagnostic-only'):
                    launcher.main(self.argv() + extra)

    def test_receipt_does_not_enable_formal_measurement(self):
        args = launcher.parser().parse_args(['--same-math-cpu-receipt', str(self.receipt)])
        with self.assertRaisesRegex(ValueError, 'only to diagnostic'):
            protocol.validate_same_math_mode(args)
        folder = self.root / 'run'
        attempt = folder / 'case/attempts/0001'
        ident = dict(same_math_diagnostic_only=True)
        result = dict(status='complete', identity=ident, timing_eligible=False,
                      requests=[], summary={}, same_math_receipt=dict(status='complete', passed=True))
        self.write(attempt / 'result.json', result)
        self.write(attempt / 'monitor.json', dict(status='complete', exit_code=0))
        self.write(folder / 'status.json', dict(status='complete', jobs={'case':dict(status='complete',attempt=str(attempt))}))
        self.assertEqual(launcher.find_prior(folder / 'case', ident)['status'], 'complete')
        with self.assertRaisesRegex(ValueError, 'invalid measurement'):
            collect(folder)
        self.assertIsNone(launcher.find_prior(folder / 'case', dict(same_math_diagnostic_only=False)))
        result['same_math_receipt']['passed'] = False
        self.write(attempt / 'result.json', result)
        self.assertEqual(launcher.find_prior(folder / 'case', ident)['status'], 'invalid_incomplete_receipt')

    def test_dry_diagnostic_never_starts_worker_or_renders_infra_table(self):
        model, adapter, out = self.root / 'model', self.root / 'adapter.stub', self.root / 'dry'
        self.write(model / 'config.json', dict(model_type='qwen3', num_hidden_layers=36, max_position_embeddings=8192))
        adapter.write_bytes(b'not-a-model')
        argv = self.argv() + ['--model', str(model), '--adapter', str(adapter), '--out', str(out), '--python', sys.executable]
        stub = lambda p: 'stub:' + Path(p).name
        with patch.object(protocol, 'digest', side_effect=stub), patch.object(launcher, 'digest', side_effect=stub), \
             patch.object(launcher, 'snapshot', side_effect=AssertionError('GPU query')), \
             patch.object(launcher, 'render_report', side_effect=AssertionError('formal table')), \
             patch.object(launcher.subprocess, 'Popen', side_effect=AssertionError('worker')):
            self.assertEqual(launcher.main(argv), 0)
        state = json.loads((out / 'status.json').read_text())
        self.assertTrue(state['same_math_diagnostic_only'])
        self.assertEqual(len(state['jobs']), 1)
        self.assertEqual(next(iter(state['jobs'].values()))['status'], 'pending')
        self.assertFalse((out / 'INFRA_REPORT.md').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)


