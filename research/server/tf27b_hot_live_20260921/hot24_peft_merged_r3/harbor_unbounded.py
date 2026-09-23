"""Run-scoped Harbor deadline policy, imported before this run constructs Trials."""
from harbor.trial.trial import Trial

def no_deadline(self, *args, **kwargs):
    return None

for method in ['_compute_agent_timeout_sec','_compute_verifier_timeout_sec',
               '_compute_agent_setup_timeout_sec','_compute_environment_build_timeout_sec',
               '_step_verifier_timeout_sec']:
    assert hasattr(Trial,method), method
    setattr(Trial,method,no_deadline)
