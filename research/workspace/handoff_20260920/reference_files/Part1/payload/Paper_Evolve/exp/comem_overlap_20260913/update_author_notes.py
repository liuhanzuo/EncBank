from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'COMem/paper_iclr2027_rewrite_20260912'
notice='> 2026-09-13 后续更新：作者已确认当前adapter可视为论文同一checkpoint；teacher全因果定义及实际训练参数已写回。原始contextual脚本也已找到，它同时扩大query和下层decode上下文，故已收窄归因。固定w32的500例LongEval/Qasper质量补测已完成并加入正文/附录：LongEval macro下降10.33分，Qasper无明确改善；5090含Write成本仍在运行。详见 `METHOD_VISIBILITY_zh.md` 和 `OVERLAP_CONFIRMATION_zh.md`。下文保留前次评审/修订时点的判断；未重评、未改独立评审原报告。\n\n'
for name in ('SCIENTIFIC_WRITING_REVIEW_zh.md','REVISION_NOTES_zh.md','SUPPLEMENTARY_EXPERIMENTS_zh.md'):
    p=PAPER/name;s=p.read_text(encoding='utf-8')
    if not s.startswith(notice):p.write_text(notice+s,encoding='utf-8')
p=PAPER/'README.md';s=p.read_text(encoding='utf-8')
s=s.replace('the fixed-w32 LongEval/Qasper repair experiment is now running under `exp/comem_overlap_20260913`; other accuracy experiments remain proposals.','the fixed-w32 LongEval/Qasper quality experiment has completed under `exp/comem_overlap_20260913` and is included in the manuscript. Local Write-inclusive cost measurements are still running; other proposed accuracy experiments remain unrun. See `OVERLAP_CONFIRMATION_zh.md`.')
p.write_text(s,encoding='utf-8')
p=PAPER/'CLAIM_EVIDENCE.md';s=p.read_text(encoding='utf-8')
s=s.replace('This document maps the rewritten claims to evidence; it does not certify a re-run of the experiments.','This document maps claims to their specific evidence; new experiments are distinguished below from transcribed historical results.')
s=s.replace('On paired 8k/16k multikey, document context reaches 100 under either Read-position convention; local Write is 92.5/88.0. A 32-token overlap reaches 98.5 with unchanged stored shape and online pack. Synthetic local repair only.','On paired 8k/16k multikey, full-document context reaches 100 under either Read-position convention; local Write is 92.5/88.0. The full-document arm also broadens query and decode context and does not isolate document states. A 32-token overlap reaches 98.5 with unchanged stored shape and online pack. Independent confirmation degrades LongEval from 72.67 to 62.33 and shows no clear Qasper gain (11.37 to 10.76).')
s=s.replace('At equal latency with BM25','Under latency-calibrated chunk budgets with BM25')
s=s.replace('New final4000 adapter and unscored fixed PG19 workloads','Author-confirmed principal adapter and unscored fixed PG19 workloads')
s=s.replace('- The rewrite supplies missing teacher-mask details or independently reproduces the research artifact.','- The full-document control isolates document-cache fidelity, or the independent confirmation exactly repeats historical sample/sink settings.\n\nTeacher-mask details and checkpoint identity are now resolved by code inspection and author confirmation; teacher top-64 retained mass remains unreported. The principal training recipe is aligned to the executed configuration. See `METHOD_VISIBILITY_zh.md`.')
p.write_text(s,encoding='utf-8')
p=ROOT/'exp/comem_overlap_20260913/PROTOCOL_zh.md';s=p.read_text(encoding='utf-8')
cap_note='\n\n运行参数澄清：run_cost将28e9/1024³传给gpu_gate的cap_gb，而该参数按十进制GB计算。三个进程因此实际使用约26.077GB allocator上限，严格小于28GB预算。保持三个正式进程参数一致，不在中途更换；最终cost_summary与结果报告披露实际上限并覆盖metadata中仅写名义cap_bytes的歧义。\n'
if '运行参数澄清' not in s:p.write_text(s+cap_note,encoding='utf-8')
