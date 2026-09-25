# GPU7 concurrency recovery, 2026-09-25

The score-only main benchmark uses registry v7, SHA-256 `90c769dffec91d8ae4a811187c249b45a83a076cdef855ac7035b0fe296f8035`. Dense remains closed. The latest pre-recovery verified coverage was Dense 89/89, k12 38/89, and k48 58/89. Five original score-only jobs remain active on gpu3/gpu5/gpu6: 141570, 141572, 141574, 141575, and 141577. The four original gpu7 jobs 141571, 141573, 141576, and 141578 were `CANCELLED by 0` after about three hours with zero formal requests and scores. The canceling actor is unknown; they must not be restarted or counted as scored failures.

The gpu7 host admission ledger was exactly full at 204800 MiB despite no owned task launching. Under its original lock, `reconcile_gpu7_host_admission_20260925.py` archived the original ledger, removed only 39 entries whose owners were confirmed dead and whose Slurm jobs had ended, and retained four entries (18432 MiB) with matching live Harbor processes. The original SHA-256 was `9e0e53c804f21bffc198470b6ab7cfd4807f9a6c1157ad055fb34ffd46a72d3c`; the backup and applied receipt are under `/cluster/home/liuhanzuo/qcomem_runtime_20260911/server_control_20260920/unbounded_host_admission/l20-instan-001/`. A real 8192 MiB acquire/release probe passed after reconciliation. This directly explains the host-admission stall, but the Slurm cancellation cause remains unproven.

The four zero-call source shards were split into eight disjoint, immutable children covering the same 44 previously unstarted tasks. `gpu7_score_recovery_r2_20260925.py` prepared and submitted them once. Twenty sustained storage rounds and 200 metadata cycles passed on gpu7 before submission. Current identities are each root's `submission.json`:

| Arm | Job | Tasks | GPU7 eligible |
| --- | ---: | ---: | --- |
| k12 | 143158 | 4 | Yes |
| k12 | 143159 | 5 | No |
| k12 | 143160 | 4 | Yes |
| k12 | 143161 | 6 | No |
| k48 | 143162 | 6 | Yes |
| k48 | 143163 | 6 | No |
| k48 | 143164 | 7 | Yes |
| k48 | 143165 | 6 | No |

At 10:21 +08, the four eligible jobs were RUNNING on gpu7 and 143159 was RUNNING on gpu1. The four gpu7 jobs had loaded their own model workers, launched 4/4/6/6 real trials, and first formal requests appeared. At 10:23, 143158 had its first complete 207-token response, with one request-complete event; 143164 and 143159 also had request-start events. The remaining three non-gpu7 children were queued by the scheduler. Keep the gpu7 cap at four EncBank GPUs; independent Reader GPUs are excluded by the user's instruction and must not be touched. No per-GPU task ceiling was increased: k12 remains four and k48 six because earlier eight-way k12 SDPA expansion OOMed and several surviving workers already approach physical VRAM limits. The added capacity comes from separate one-GPU jobs. Queueing, model-ready, trial launch, request, reply, and scored closure must be reported as distinct states.

Do not duplicate these IDs, hot-patch used sources, or run the obsolete global-four-card `score_resume_release_gate_20260925.py --release`. Continue read-only monitoring of real Slurm state, unique task ownership, request/reply hashes, verifier, parent exit, container/session release, and the gpu7 four-card cap. Infrastructure interruption is not a normal zero score; native context exhaustion remains a separately recorded zero.
