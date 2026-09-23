"""CPU-only post-completion packet creation, no judge calls; one fresh output directory."""
import argparse,hashlib,json,random
from pathlib import Path
from protocol import read,save,sha,local

def make_candidates(plan,results,labels,seed):
    rows=[];crosswalk=[]
    rng=random.Random(seed)
    candidates=[(arm,row,variant) for arm in plan['arm_order'] for row in results[arm]['rows'] for variant in ('full64','posthoc_prefix32')]
    rng.shuffle(candidates)
    for index,(arm,row,variant) in enumerate(candidates):
        candidate_id='candidate_'+format(index+1,'04d')
        value=row if variant=='full64' else row['posthoc_prefix32']
        rows.append({'candidate_id':candidate_id,'question':row['question_for_blinded_assessment'],
                     'reference_answers':labels[row['id']]['references'],'candidate_answer':value['prediction']})
        crosswalk.append({'candidate_id':candidate_id,'arm':arm,'item_id':row['id'],'variant':variant})
    return rows,crosswalk

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--expected-plan-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert sha(a.plan)==a.expected_plan_sha256
    plan=read(a.plan);batch=read(local(plan['batch_output'])/'batch_status.json')
    assert batch['status']=='five_arms_and_analysis_complete_shell_and_slurm_exit_pending'
    assert batch['analysis_sha256']==sha(local(plan['batch_output'])/'completed_analysis.json')
    assert sha(local(plan['labels']['path']))==plan['labels']['sha256']
    results={arm:read(local(plan['outputs'][arm])/'result.json') for arm in plan['arm_order']}
    assert all(len(r['rows'])==32 and r['outer_closed'] for r in results.values())
    labels={x['id']:x for x in read(local(plan['labels']['path']))['items']}
    rows,crosswalk=make_candidates(plan,results,labels,2026091301)
    assert len(rows)==len(crosswalk)==320
    out=Path(a.output).resolve();out.relative_to(local(plan['assessment']['output_root']).parent)
    assert out==local(plan['assessment']['output_root']) and not out.exists();out.mkdir(parents=True,exist_ok=False)
    save(out/'assessor_packet.json',{'schema':'blinded_answer_content_packet_v1','candidates':rows})
    save(out/'PRIVATE_crosswalk_not_for_assessors.json',{'crosswalk':crosswalk})
    save(out/'preparation_receipt.json',{'status':'PACKET_ONLY_NO_JUDGMENTS','packet_sha256':sha(out/'assessor_packet.json'),'crosswalk_sha256':sha(out/'PRIVATE_crosswalk_not_for_assessors.json'),'plan_sha256':a.expected_plan_sha256,'actual_judge_model':None,'judgments_created':False,'seed':2026091301,'human_labels':False,'official_benchmark_judge':False})
if __name__=='__main__':main()
