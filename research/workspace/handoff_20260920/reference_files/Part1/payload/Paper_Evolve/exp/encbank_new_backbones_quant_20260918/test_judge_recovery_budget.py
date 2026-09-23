"""CPU-only test of the scoped extra-request budget; no API requests."""
import ast
import json
import tempfile
import time
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.entry=self.out/'cache'/'digest'
        self.calls=[]
        def write(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        self.write=write
        def call(prompt,target,timeout):
            self.calls.append(target)
            result=dict(ok=True,answer='CORRECT',returncode=0,errors=[])
            write(target/'result.json',result)
            return result
        module=ast.parse((ROOT/'judge_watch.py').read_text(encoding='utf-8'))
        functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('retryable','classify')]
        self.ns=dict(Path=Path,json=json,time=time,OUT=self.out,PROMPT='{question}',call=call,write_json=write)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual-classifier>','exec'),self.ns)
        self.failure=dict(ok=False,returncode=1,answer='',errors=[dict(message='unexpected status 503 Service Unavailable')])
        for i in range(3):write(self.entry/('attempt-'+str(i))/'result.json',self.failure)

    def grant(self):
        self.write(self.out/'transport_recovery.json',dict(id='verified-service',
            extra_attempts_by_stimulus={'digest':dict(previous_attempts=3,maximum_additional_attempts=1)}))

    def run_classify(self):
        return self.ns['classify'](dict(question='synthetic test'),self.entry/'attempt-next')

    def test_exhausted_without_grant_makes_no_call(self):
        self.assertFalse(self.run_classify()['ok']);self.assertEqual(self.calls,[])

    def test_grant_allows_only_one_actual_request(self):
        self.grant()
        self.assertTrue(self.run_classify()['ok'])
        self.assertTrue(self.run_classify()['ok'])
        self.assertEqual(len(self.calls),1)

    def test_uncertain_dispatch_is_not_repeated(self):
        self.grant()
        self.write(self.entry/'service-recovery-verified-service'/'dispatch.json',{})
        with self.assertRaises(AssertionError):self.run_classify()
        self.assertEqual(self.calls,[])

    def test_failed_recovery_is_not_repeated(self):
        self.grant()
        self.write(self.entry/'service-recovery-verified-service'/'result.json',self.failure)
        self.assertFalse(self.run_classify()['ok']);self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
