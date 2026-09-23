"""
S17 -- Re-select the sparse lower-layer set S under the DEPLOYABLE write, on several
DISJOINT selection splits, with runner-up margins.  Qwen3-8B, j = 12, inference only.

WHY
---
The |S|=3 sets used so far ({1,9,11} wikitext, {1,4,5} PG19) were greedy-selected in S13
under the slot-aware write, on one 12-sample split each, and the S15 RULER runs then used
{1,9,11} everywhere.  Per the retracted layer-9 anchor (memory: encbank-layer9-anchor-
retracted), a layer set is only reportable when it recurs across >= 3 disjoint selection
sets and the greedy step's runner-up margin is stated.  S15d also showed that
variable_tracking wants ~6 lower layers, so the sweep goes to |S| = 6.

DESIGN
------
Selection splits (each 12 windows of 12x512-token chunks + 32-token query, k=4 scattered):
  pgA, pgB, pgC   three disjoint, later regions of the clean PG19 stream (docs 2+), placed
                  after the S13/S14 selection (windows 0-11) and eval (windows 12-35) regions
  wikiA           the wikitext 'alt' half (the S13 selection text)
Greedy forward selection at j=12 up to |S| = --smax under the deployable local write
(s14.prepare / s14.arm_FL), objective = mean frac over the split.  At every step the full
candidate table is kept, plus the runner-up margin (best - second best, mean frac) and the
number of selection samples on which best < second best.

Consensus sets: layer frequency over the splits' |S|=3 sets and |S|=6 sets; S3* = the three
most frequent layers, S6* = the six most frequent (ties by mean greedy rank).

Evaluation (never on a selection split): every split's S3/S6, the consensus S3*/S6*, the old
{1,9,11} and {1,4,5}, FL_empty and FL_all, on the S14 eval splits (PG19 windows 12-35, n=24;
wikitext 'main', n=15), reported as frac with paired CIs against FL_all.

Writes exp/results/s17_reselect.json and exp/results/s17_sets.json (for the RULER step).
"""

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))

from s13_query_fix import kl, score, samples_from, pg19_text, tiny_model  # noqa: E402
from s14_deployable_fix import prepare, arm_FL                          # noqa: E402


def boot_ci(v, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


@torch.no_grad()
def greedy_split(model, sink, samples, j, smax, log):
    """Greedy forward selection on one split.  Returns the trace (one entry per step)."""
    preps = [prepare(model, sink, chunks, qi, [j], j) for chunks, qi in samples]
    S, trace = [], []
    for step in range(smax):
        cands = [l for l in range(j) if l not in S]
        per = {l: [] for l in cands}
        for P in preps:
            for l in cands:
                lg = arm_FL(model, P, j, S + [l])
                per[l].append(score(lg, P["ref"], P["base"], P["targets"], P["ref_nll"])["frac"])
        means = {l: st.mean(v) for l, v in per.items()}
        order = sorted(cands, key=lambda l: means[l])
        best, second = order[0], (order[1] if len(order) > 1 else None)
        entry = {"size": step + 1, "added": best, "layers": sorted(S + [best]),
                 "sel_frac": means[best], "cands": {str(l): round(means[l], 4) for l in order}}
        if second is not None:
            diffs = [a - b for a, b in zip(per[best], per[second])]
            entry.update({"runner_up": second, "margin": means[second] - means[best],
                          "best_wins": sum(d < 0 for d in diffs), "n": len(diffs)})
        S.append(best)
        trace.append(entry)
        log(f"    |S|={step+1}: +{best} -> {sorted(S)}  frac {means[best]:.3f}  "
            f"runner-up {second} margin {entry.get('margin', float('nan')):.4f}  "
            f"best wins {entry.get('best_wins', '-')}/{len(preps)}")
    return trace


@torch.no_grad()
def eval_sets(model, sink, samples, j, sets, L):
    rows = []
    for si, (chunks, qi) in enumerate(samples):
        P = prepare(model, sink, chunks, qi, [j], j)
        row = {"sample": si, "kl_no_mem": P["base"]}
        for name, S in sets.items():
            lg = arm_FL(model, P, j, S)
            r = score(lg, P["ref"], P["base"], P["targets"], P["ref_nll"])
            row[f"{name}_frac"] = r["frac"]
            row[f"{name}_gap"] = r["gap"]
        rows.append(row)
    return rows


def consensus(traces, size):
    """Layers ranked by how many splits include them in their |S|=size set, then by mean
    greedy rank (earlier is better)."""
    freq, rank = {}, {}
    for tr in traces.values():
        layers = tr[size - 1]["layers"]
        for e in tr[:size]:
            l = e["added"]
            freq[l] = freq.get(l, 0) + 1
            rank.setdefault(l, []).append(e["size"])
    order = sorted(freq, key=lambda l: (-freq[l], st.mean(rank[l]), l))
    return sorted(order[:size]), {str(l): freq[l] for l in order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--pg19", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--pg19-skip", type=int, default=2)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--smax", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-sel", type=int, default=12)
    ap.add_argument("--n-eval-pg", type=int, default=24)
    ap.add_argument("--out", default="exp/results/s17_reselect.json")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)

    if args.cpu_smoke:
        model, L, j, smax = tiny_model(), 4, 3, 2
        torch.manual_seed(1)
        ids = torch.randint(5, 512, (6000,))
        sink = torch.tensor([[3]])
        c, nq, k, n_doc, n_sel = 8, 4, 2, 3, 2
        win = n_doc * c + nq
        splits = {"pgA": samples_from(ids, c, n_doc, nq, k, n_sel, start=36 * win),
                  "pgB": samples_from(ids, c, n_doc, nq, k, n_sel, start=48 * win),
                  "pgC": samples_from(ids, c, n_doc, nq, k, n_sel, start=60 * win),
                  "wikiA": samples_from(ids[3000:], c, n_doc, nq, k, n_sel)}
        evals = {"pg19": samples_from(ids, c, n_doc, nq, k, 3, start=12 * win),
                 "wikitext": samples_from(ids[4000:], c, n_doc, nq, k, 2)}
        out = Path("exp/results/s17_smoke.json")
        old = {"old_g3w": [0, 2], "old_g3p": [0, 1]}
    else:
        from gpu_gate import acquire_gpu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from s1_binding_curve import fetch_text
        acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb, idle_slack_gb=args.idle_slack_gb,
                    tag="s17_reselect")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        L, j, smax = model.config.num_hidden_layers, args.j, args.smax
        sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device="cuda")
        c, nq, k, n_doc, n_sel = args.chunk, args.query, args.k, args.n_doc, args.n_sel
        win = n_doc * c + nq
        need = win * (36 + 3 * n_sel + 2) * 5
        pg = tok(pg19_text(args.pg19, need, args.pg19_skip), return_tensors="pt").input_ids[0].to("cuda")
        alt = tok(fetch_text("alt", Path("exp/data")), return_tensors="pt").input_ids[0].to("cuda")
        main_ids = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to("cuda")
        splits = {"pgA": samples_from(pg, c, n_doc, nq, k, n_sel, start=36 * win),
                  "pgB": samples_from(pg, c, n_doc, nq, k, n_sel, start=(36 + n_sel) * win),
                  "pgC": samples_from(pg, c, n_doc, nq, k, n_sel, start=(36 + 2 * n_sel) * win),
                  "wikiA": samples_from(alt, c, n_doc, nq, k, n_sel)}
        evals = {"pg19": samples_from(pg, c, n_doc, nq, k, args.n_eval_pg, start=12 * win),
                 "wikitext": samples_from(main_ids, c, n_doc, nq, k, 24)}
        out = Path(args.out)
        old = {"old_g3w": [1, 9, 11], "old_g3p": [1, 4, 5]}
    log(f"model L={L} j={j} smax={smax}; splits " + ", ".join(f"{n}={len(s)}" for n, s in splits.items())
        + "; evals " + ", ".join(f"{n}={len(s)}" for n, s in evals.items()))
    if any(len(s) < n_sel for s in splits.values()):
        log("WARNING: a selection split is short of samples (corpus exhausted)")

    traces = {}
    for name, samples in splits.items():
        log(f"greedy on {name} (n={len(samples)})")
        traces[name] = greedy_split(model, sink, samples, j, smax, log)

    s3, f3 = consensus(traces, min(3, smax))
    s6, f6 = consensus(traces, min(6, smax))
    log(f"consensus S3*={s3} (freq {f3});  S6*={s6} (freq {f6})")

    sets = {"FL_empty": [], "FL_all": list(range(j)), "S3_star": s3, "S6_star": s6, **old}
    for name, tr in traces.items():
        sets[f"S3_{name}"] = tr[min(3, smax) - 1]["layers"]
        if smax >= 6:
            sets[f"S6_{name}"] = tr[5]["layers"]
    ev = {name: eval_sets(model, sink, samples, j, sets, L) for name, samples in evals.items()}

    summary = {}
    for ename, rows in ev.items():
        summary[ename] = {}
        for sname in sets:
            v = [r[f"{sname}_frac"] for r in rows]
            d = [r[f"{sname}_frac"] - r["FL_all_frac"] for r in rows]
            lo, hi = boot_ci(d)
            summary[ename][sname] = {"frac": st.mean(v), "gap": st.mean(r[f"{sname}_gap"] for r in rows),
                                     "vs_all": st.mean(d), "ci": [lo, hi], "n": len(rows)}
    log("\n### eval (frac; lower is better; paired vs FL_all with 95% CI)")
    for ename in summary:
        log(f"-- {ename} n={summary[ename]['FL_all']['n']}")
        for sname, r in summary[ename].items():
            log(f"   {sname:10s} S={str(sets[sname]):22s} frac {r['frac']:.3f}  gap {r['gap']:.2f}  "
                f"vs FL_all {r['vs_all']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "L": L, "j": j, "traces": traces,
                               "consensus": {"S3": s3, "S6": s6, "freq3": f3, "freq6": f6},
                               "sets": sets, "eval_rows": ev, "summary": summary}, indent=1))
    Path("exp/results/s17_sets.json").write_text(json.dumps({"S3": s3, "S6": s6}))
    if not args.cpu_smoke:
        log(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
