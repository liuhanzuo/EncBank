"""Known platform mapping only; never claims InfLLM GPU qualification."""
from protocol import local,read,sha
def validate_backend_binding(plan,*,allow_unbound=False):
    assert plan['launch_permitted'] is True
    q=plan['backend_qualification']
    assert q['runtime_versions']['torch']=='2.10.0+cu128' and q['runtime_versions']['transformers']=='5.5.4'
    p=local(q['registration']['path']);assert sha(p)==q['registration']['sha256']
    old=read(p)
    assert old['compute_capability']==q['compute_capability']
    assert old['verified_node_hostnames']==q['node_hostnames']
    assert q['qualified_nodes']==list(q['node_hostnames'])
    return True
def qualified_node_identity(plan,environ,hostname):
    q=plan['backend_qualification'];node=environ.get('SLURMD_NODENAME') or environ.get('SLURM_JOB_NODELIST')
    assert node in q['qualified_nodes'] and hostname in q['node_hostnames'][node]
    return {'slurm_node':node,'hostname':hostname}
