"""Small fixed semantic/format checks, performed only after connectivity recovers."""
import json
from codex_judge_client import ROOT, MODEL, call
from locomo_judge_reference import PROMPT
from evaluation_scope import read_scope

CASES=[
    ('date_equivalent','When did Caroline go to the LGBTQ support group?','7 May 2023','May 7, 2023','CORRECT'),
    ('date_wrong','When did Caroline go to the LGBTQ support group?','7 May 2023','May 8, 2023','WRONG'),
    ('complete_fields','What fields would Caroline be likely to pursue in her education?','Psychology, counseling certification','Psychology and certification in counseling.','CORRECT'),
    ('missing_field','What fields would Caroline be likely to pursue in her education?','Psychology, counseling certification','Psychology.','WRONG'),
    ('paraphrase','What did Caroline research?','Adoption agencies','She researched agencies that arrange adoptions.','CORRECT'),
    ('wrong_entity','What did Caroline research?','Adoption agencies','Travel agencies.','WRONG'),
    ('exact_answer','What did the charity race raise awareness for?','mental health','Mental health.','CORRECT'),
    ('unjustified_refusal','What did the charity race raise awareness for?','mental health','I do not know.','WRONG'),
]

def main():
    assert json.loads((ROOT/'locomo_scope.json').read_text())['judge_model']==MODEL
    fixed=json.loads((ROOT/'judge_protocol.json').read_text());assert fixed['model']==MODEL
    out=ROOT/'judge_gpt6_astra'/'calibration';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ident,q,g,p,expected in CASES:
        result=call(PROMPT.format(question=q,gold=g,pred=p),out/ident)
        rows.append(dict(id=ident,question=q,gold=g,pred=p,expected=expected,
                         returned=result['answer'].strip(),transport_ok=result['ok']))
        if not result['ok']:break
    passed=len(rows)==len(CASES) and all(r['transport_ok'] and r['returned']==r['expected'] for r in rows)
    report=dict(model=MODEL,passed=passed,n=len(rows),planned=len(CASES),cases=rows,
                scope='Sanity/format calibration, not evidence of full human-judge agreement')
    (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    if passed:
        fixed.update(calibration_passed=True,generation_verified=True,status='READY',
                     calibration_file=str(out/'calibration.json'))
        (ROOT/'judge_protocol.json').write_text(json.dumps(fixed,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
