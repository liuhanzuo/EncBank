"""Apply author-confirmed checkpoint identity to current text and its generators."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'Encbank/paper_iclr2027_rewrite_20260912'
for name in ('sections/tab_infra.tex','sections/tab_local_infra_processes.tex','README.md'):
    p=PAPER/name; s=p.read_text(encoding='utf-8')
    s=s.replace(r'the adapter differs from Table~\ref{tab:accuracy}',r'the adapter is shared with Table~\ref{tab:accuracy}; these cost inputs are unscored')
    s=s.replace('same newly trained adapter','same principal adapter')
    s=s.replace('shared newly trained suffix adapter','shared principal suffix adapter')
    s=s.replace("The two unresolved historical source issues are the training teacher's attention mask and the alignment between the LongBench clean-subset audit and headline prediction cohorts.","The teacher visibility and checkpoint identity have been resolved by inspecting code and incorporating the author's checkpoint confirmation; the LongBench clean-subset/headline-cohort alignment remains unresolved.")
    s=s.replace('its natural-task repair and additional accuracy experiments remain proposals.','the fixed-w32 LongEval/Qasper repair experiment is now running under `exp/encbank_overlap_20260913`; other accuracy experiments remain proposals.')
    s=s.replace('The current local teacher code was inspected after the initial rewrite and uses full causal packed replay by default; its correspondence to the historical ARR checkpoint still needs confirmation.','The teacher uses adapter-disabled full causal packed replay. The author confirms that the present adapter can be treated as the paper checkpoint. The original contextual-control script additionally exposes the query and lower-layer decode to the full source; the manuscript now states this and narrows the attribution claim. See `METHOD_VISIBILITY_zh.md`.')
    p.write_text(s,encoding='utf-8')
for name in ('encbank_infra_20260912/aggregate.py','encbank_infra_20260912/tab_infra_template.tex','encbank_dense5090_20260913/update_paper.py','encbank_b300_recheck_20260912/update_paper.py'):
    p=ROOT/'exp'/name;s=p.read_text(encoding='utf-8')
    s=s.replace(r'the adapter differs from Table~\ref{tab:accuracy}',r'the adapter is shared with Table~\ref{tab:accuracy}; these cost inputs are unscored')
    s=s.replace(r'The local cost workloads and newly trained adapter are separate from Table~\ref{tab:accuracy}.',r'The adapter is shared with Table~\ref{tab:accuracy}, but these cost inputs do not supply accuracy evidence.')
    s=s.replace(r'The shared rank-32 adapter is newly trained, distinct from Table~\ref{tab:accuracy}',r'The shared rank-32 adapter is the principal Table~\ref{tab:accuracy} checkpoint')
    s=s.replace('same newly trained unmerged adapter','same principal unmerged adapter').replace('shared newly trained suffix adapter','shared principal suffix adapter')
    p.write_text(s,encoding='utf-8')
