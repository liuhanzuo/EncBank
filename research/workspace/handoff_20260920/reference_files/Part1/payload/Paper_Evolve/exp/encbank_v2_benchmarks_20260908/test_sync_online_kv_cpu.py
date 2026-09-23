"""Small metadata-only completion-boundary tests; no model/Torch imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sync_remote_results import summarize_online_kv, scheduled_workloads_complete

ARMS = ['fix_all','pub','pub_sink','j0','pub_lora','cacheblend16']
PLAN = {'protocol':'online-kv-inventory-v1','methods':ARMS,'document_lengths':[32768,131072],
        'generation_lengths':[16,128],'boundaries':['prefill_complete','decode_complete'],
        'planned_generations':48,'planned_cache_boundaries':96,'timing_eligible':False}


def save(path,obj):
    path.write_text(json.dumps(obj),encoding='utf-8')


class OnlineKvProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.state={'status':'completed','protocol':'online-kv-inventory-v1',
            'required_gpu':'NVIDIA GeForce RTX 5090','remote_timing_allowed':False,
            'timing_eligible':False,'jobs':{},'active_job':None,'child_pid':None,'error':None}
        hardware={'platform':'Windows','device':'cuda:0','device_name':'NVIDIA GeForce RTX 5090','timing_eligible':True,
            'torch_cpu_threads':2,'torch_interop_threads':16,'omp_num_threads':'2','mkl_num_threads':'2','tokenizers_parallelism':'false',
            'gpu_admission':{'effective_idle_slack_gib':5.0,'comparison':'strictly_less_than','source':'nvidia-smi MiB / 1024',
                'initial_used_gib':4.0,'recheck_used_gib':4.1,'other_python_compute_processes':[]}}
        inventory={'populated_layer_entries':36,'tensor_count':72,'logical_tensor_bytes':100,'unique_backing_storage_bytes':100,
            'layers':[{'layer_index':i,'tensors':[{'device':'cuda:0'},{'device':'cuda:0'}]} for i in range(36)]}
        self.rows={}
        for arm in ARMS:
            folder=self.root/arm;folder.mkdir()
            cases=[{'arm':arm,'context_tokens':n,'G':g,'query_id':q,'case_id':f'{arm}_{n}_q{q}_g{g}'}
                   for n in (32768,131072) for g in (16,128) for q in (0,2)]
            rows=[dict(c, protocol=PLAN['protocol'],hardware=hardware,timing_eligible=False,
                persistent_store_metadata_unchanged=True,generated_tokens=c['G'],generated_ids=[1]*c['G'],
                actual_decode_forward_calls=c['G']-1,fixed_generation_length=True,document_or_query_capture_calls=0,
                state_reset_verified=True,num_layers=36,historical_generated_ids_equal=True,
                inventory={'prefill_complete':inventory,'decode_complete':inventory}) for c in cases]
            self.rows[arm]=copy.deepcopy(rows)
            marker={'status':'complete','protocol':PLAN['protocol'],'arm':arm,'diagnostic_generations':8,
                'observed_cache_boundaries':16,'timing_eligible':False,'hardware':hardware,'historical_output_disagreement_cases':0}
            save(folder/'cases.json',cases);save(folder/'COMPLETED.json',marker)
            self.write_rows(arm)
            self.state['jobs'][arm]={'status':'complete','result':str(folder),'diagnostic_generations':8,
                'observed_cache_boundaries':16,'timing_eligible':False,'historical_output_disagreement_cases':0}
    def tearDown(self): self.temp.cleanup()
    def write_rows(self,arm):
        (self.root/arm/'inventories.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows[arm])+'\n')
    def marker(self,arm,mutate):
        path=self.root/arm/'COMPLETED.json';m=json.loads(path.read_text());mutate(m);save(path,m)
    def summary(self): return summarize_online_kv(self.state,PLAN)
    def test_exact_48_generations_and_96_boundaries_complete(self):
        result=self.summary()
        self.assertTrue(result['all_complete']);self.assertEqual(result['completed_diagnostic_generations'],48)
        self.assertEqual(result['completed_cache_boundaries'],96)
    def test_one_missing_method_is_40_80_and_incomplete(self):
        del self.state['jobs']['cacheblend16'];r=self.summary()
        self.assertFalse(r['all_complete']);self.assertEqual(r['completed_diagnostic_generations'],40)
        self.assertEqual(r['completed_cache_boundaries'],80)
    def test_missing_boundary_and_bad_marker_do_not_count(self):
        self.rows['fix_all'][0]['inventory'].pop('decode_complete');self.write_rows('fix_all')
        self.marker('pub',lambda m:m.update(observed_cache_boundaries=15))
        r=self.summary();self.assertEqual(r['invalid_completed_jobs'],['fix_all','pub']);self.assertFalse(r['all_complete'])
    def test_duplicate_cases_or_empty_phase_cannot_count(self):
        self.rows['fix_all'][1]['case_id']=self.rows['fix_all'][0]['case_id'];self.write_rows('fix_all')
        self.rows['pub'][0]['inventory']['decode_complete']={};self.write_rows('pub')
        r=self.summary();self.assertEqual(r['invalid_completed_jobs'],['fix_all','pub'])
    def test_equal_five_gib_and_thread_drift_rejected(self):
        self.marker('fix_all',lambda m:m['hardware']['gpu_admission'].update(recheck_used_gib=5.0))
        self.marker('pub',lambda m:m['hardware'].update(torch_cpu_threads=16))
        self.assertEqual(self.summary()['invalid_completed_jobs'],['fix_all','pub'])
    def test_historical_disagreement_is_visible_not_dropped(self):
        self.rows['fix_all'][0]['historical_generated_ids_equal']=False;self.write_rows('fix_all')
        self.marker('fix_all',lambda m:m.update(historical_output_disagreement_cases=1))
        self.state['jobs']['fix_all']['historical_output_disagreement_cases']=1
        result=self.summary();self.assertTrue(result['all_complete']);self.assertEqual(result['historical_output_disagreement_cases'],1)
    def test_active_partial_is_observed_not_complete(self):
        del self.state['jobs']['cacheblend16'];self.state.update(status='running',active_job='cacheblend16',child_pid=123,
            active_attempt=str(self.root/'cacheblend16'))
        save(self.root/'cacheblend16/status.json',{'protocol':PLAN['protocol'],'arm':'cacheblend16','completed_diagnostic_generations':3})
        r=self.summary();self.assertEqual(r['completed_diagnostic_generations'],40);self.assertEqual(r['observed_diagnostic_generations'],43)
        self.assertEqual(r['observed_cache_boundaries'],86);self.assertFalse(r['all_complete'])
    def test_active_child_or_error_prevents_complete(self):
        for update in ({'child_pid':1},{'active_job':'pub'},{'error':'failed'}):
            state=copy.deepcopy(self.state);state.update(update)
            self.assertFalse(summarize_online_kv(state,PLAN)['all_complete'])
    def test_bad_or_absent_plan_cannot_finish(self):
        self.assertFalse(summarize_online_kv(self.state,{})['all_complete'])
        plan=copy.deepcopy(PLAN);plan['planned_cache_boundaries']=48
        self.assertFalse(summarize_online_kv(self.state,plan)['all_complete'])
    def test_global_completion_requires_online_diagnostic(self):
        self.assertFalse(scheduled_workloads_complete(True,'completed',True,True,False,True,[]))
        self.assertTrue(scheduled_workloads_complete(True,'completed',True,True,True,True,[]))
        self.assertFalse(scheduled_workloads_complete(True,'completed',True,True,True,True,['error']))


if __name__=='__main__': unittest.main(verbosity=2)
