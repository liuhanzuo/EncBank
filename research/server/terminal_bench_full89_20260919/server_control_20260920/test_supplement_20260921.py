import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import supplement_20260921 as target


class OwnershipTests(unittest.TestCase):
    def fixture(self, root, active, completed):
        (root/'old').mkdir()
        (root/'runs'/'staged').mkdir(parents=True)
        (root/'old'/'plan.json').write_text(json.dumps({'tasks':active}))
        (root/'runs'/'staged'/'predecessor_partition.json').write_text(json.dumps({'retained_normal':[{'task':x} for x in completed]}))

    def audit(self, active, completed):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root, active, completed)
            with patch.multiple(target,B=root,S=root,ARMS=[('dense','staged','123','old')]):
                return target.audit_ownership()

    def test_disjoint_supplement(self):
        self.assertEqual(self.audit(['active-task'],['finished-task'])[0]['retained_normal'],['finished-task'])

    def test_active_task_is_rejected(self):
        with self.assertRaises(AssertionError):self.audit(['regex-chess'],[])

    def test_completed_task_is_rejected(self):
        with self.assertRaises(AssertionError):self.audit([],['vulnerable-secret'])


if __name__=='__main__':unittest.main()
