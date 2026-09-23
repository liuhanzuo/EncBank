"""Choose raw budget from calibration latency alone; never read QA scores."""
import json,statistics
import config

def choose(anchor,candidates):
    matching=[k for k,v in candidates.items() if abs(v/anchor-1)<=.05]
    if matching:return dict(strict_match_on_calibration=True,quality_raw_ks=[max(matching)])
    below=[k for k,v in candidates.items() if v<anchor];above=[k for k,v in candidates.items() if v>=anchor]
    ks=[]
    if below:ks.append(min(below,key=lambda k:abs(candidates[k]/anchor-1)))
    if above:ks.append(min(above,key=lambda k:abs(candidates[k]/anchor-1)))
    return dict(strict_match_on_calibration=False,quality_raw_ks=sorted(set(ks)))

def main():
    lat={};processes=[]
    for phase in ['calibration','extended']:
        for rep in range(3):
            p=config.ROOT/phase/f'process_{rep:02d}'
            if not (p/'complete.json').exists():
                if phase=='extended':continue
                raise ValueError('Missing primary calibration process')
            done=json.loads((p/'complete.json').read_text());assert done['complete']
            receipt=json.loads((p/'parent_exit.json').read_text());assert receipt['actual_wait'] and receipt['returncode']==0
            records=[json.loads(l) for l in (p/'records.jsonl').read_text().splitlines()]
            assert len(records)==done['records']
            for r in records:
                assert r['phase']==phase and r['score'] is None and r['generated_tokens']==1
                if r['warmup']:continue
                lat.setdefault((r['method'],r['k_max']),[]).append(r['ttft_ms'])
            processes.append(dict(phase=phase,rep=rep))
    anchor=statistics.median(lat[('comem',12)])
    raw={k:statistics.median(v) for (m,k),v in lat.items() if m=='raw'}
    outside=anchor<min(raw.values()) or anchor>max(raw.values())
    need_extended=outside and not all(k in raw for k in config.EXTENDED)
    value=dict(anchor_ttft_ms=anchor,raw_ttft_ms=raw,latency_ratios={k:v/anchor for k,v in raw.items()},
        calibration_processes=processes,needs_extension=need_extended,
        frozen_before_quality=not need_extended,selection_uses_scores=False,
        rule='Largest raw k within +/-5%; if primary grid does not bracket anchor, add 4/20/24. If no match, report neighboring points without artificial waiting or a strict-match claim.',
        **choose(anchor,raw))
    path=config.ROOT/('calibration_decision.json' if need_extended else 'budget_decision.json')
    assert not path.exists(),'Budget decision already frozen; no test-set retuning'
    path.write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))

if __name__=='__main__':main()
