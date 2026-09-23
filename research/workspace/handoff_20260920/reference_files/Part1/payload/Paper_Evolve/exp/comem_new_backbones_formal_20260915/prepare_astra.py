"""Preserve the previous judge protocol and activate the user's exact new model."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'judge_protocol.json'
old=json.loads(path.read_text())
if old['model']!='gpt-6-astra':
    archive=root/'judge_gpt6'/'protocol_before_astra.json'
    assert not archive.exists()
    archive.write_text(json.dumps(old,indent=2)+'\n')
    old.update(protocol_id='midcache-locomo-gpt6-astra-v1',model='gpt-6-astra',
        categories_1_to_4='Same GPT-6-Astra model and prompt across every compared method and backbone',
        old_judgments='Retain separately; never mix different judge models in one comparison',
        calibration_passed=False,generation_verified=True,status='PENDING_CALIBRATION',
        limitation='Service alias gpt-6-astra returned a valid label via the official CLI; underlying dated snapshot unverified.')
    path.write_text(json.dumps(old,indent=2)+'\n')
status=dict(status='RESUMED_PENDING_CALIBRATION',model='gpt-6-astra',
    user_confirmed_role='Judge, not answer-generation baseline',
    scope_file='locomo_scope.json',generation_results='results_locomo',
    connection_verified=True,calibration_passed=False)
(root/'judge_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
