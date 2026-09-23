"""
S16 -- Estate-wide screen: is a scoring arm actually reading the option? (0 GPU)

WHY
---
A36 audited an arm it called the iLLaDA confidence protocol and reported that on 260/260
equal-length items all four options received EXACTLY the same score at numerical
precision.  That is impossible for any scorer that conditions on the option text: two
distinct token strings do not produce bit-identical floating-point sums by accident, 260
times in a row.  The only explanation is that the arm never reads the candidate, so its
score is a function of (question, option length) alone.  Everything downstream of that --
tie-break to index 0, position collapse, accuracy at chance -- follows by arithmetic, not
by any property of the model.

The same failure can hide in any manuscript in the estate, because "GT injection" was used
for two different things (the candidate's own tokens in A14; nothing at all in A36's
control arm).  This script is the cheap screen.  It reads frozen per-question records and
needs no GPU and no model.

THE DIAGNOSTIC
--------------
Group every score by (question, option token length).  If an arm reads the option text,
two DIFFERENT options of the SAME length in the SAME question must get different scores.
So:

    blind_share  = fraction of such groups whose score spread (max-min) is EXACTLY 0.0.
                   Primary, and unambiguous: bit-identical floats on distinct strings.
    spread_ratio = median over those groups of
                       (spread within the (question, length) group)
                     / (spread across ALL options of that same question)
                   i.e. how much of a question's option-to-option score variation
                   survives once length is held fixed.  Content-blind -> exactly 0.

An earlier draft normalised by TOTAL score variance instead.  That was wrong and the
self-test caught it: scores vary far more between questions than between options, so
(question, length) "explained" >0.99 of the variance even for an arm with a healthy
sigma=0.8 content signal.  The denominator must be within-question, not global.

VERDICT BANDS (stated before looking at any real record)
    blind_share >= 0.99                      -> CONTENT-BLIND
    blind_share >= 0.50 or spread_ratio<0.01 -> SUSPECT, inspect by hand
    otherwise                                -> reads the option

WHAT THIS CANNOT SHOW
    It cannot tell you an arm is CORRECT -- only that it is not content-blind.  An arm can
    read the option and still be a bad scorer.  It says nothing about reveal-order
    variance, calibration, or accuracy.  It needs per-option scores; an arm that only
    persisted its argmax choice cannot be screened this way.

RECORD SCHEMA (JSONL, one object per question per arm)
    {"question_id": "q17", "arm": "conf_noinject",
     "options": [{"text": "42", "n_tokens": 2, "score": -8.13}, ...]}
  `n_tokens` may be omitted; then it is not used and only exact whole-question ties are
  reported.  Run with --self-test first to see the diagnostic separate known arms.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def screen(records):
    """records: list of {arm, question_id, options:[{text, n_tokens, score}]}"""
    groups = defaultdict(list)          # (arm, qid, length) -> [(text, score)]
    per_q = defaultdict(list)           # (arm, qid)         -> [score]
    for r in records:
        for o in r["options"]:
            groups[(r["arm"], r["question_id"], o.get("n_tokens"))].append(
                (o["text"], float(o["score"])))
            per_q[(r["arm"], r["question_id"])].append(float(o["score"]))

    def med(v):
        v = sorted(v)
        n = len(v)
        return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))

    out = {}
    for arm in sorted({k[0] for k in per_q}):
        gs = [(k, v) for k, v in groups.items() if k[0] == arm
              and len({t for t, _ in v}) >= 2]      # >=2 DISTINCT option texts
        if not gs:
            out[arm] = dict(groups=0, verdict="no comparable group")
            continue
        exact, spreads, ratios = 0, [], []
        for (a, qid, _), v in gs:
            sp = max(s for _, s in v) - min(s for _, s in v)
            exact += sp == 0.0
            spreads.append(sp)
            qs = per_q[(a, qid)]
            denom = max(qs) - min(qs)
            # the group is a subset of the question's options, so sp <= denom; when the
            # question itself has zero spread both are 0 and the ratio is 0, not undefined
            ratios.append(sp / denom if denom > 0 else 0.0)
        bs = exact / len(gs)
        sr = med(ratios)
        verdict = ("CONTENT-BLIND" if bs >= 0.99
                   else "SUSPECT" if bs >= 0.50 or (sr is not None and sr < 0.01)
                   else "reads the option")
        out[arm] = dict(groups=len(gs), exact_zero_spread=exact, blind_share=bs,
                        median_spread=med(spreads), spread_ratio=sr, verdict=verdict)
    return out


def self_test(seed=20260905):
    """Synthetic positive and negative controls, so we know the screen works."""
    rng = random.Random(seed)
    recs = []
    for q in range(400):
        base = rng.uniform(-30, -5)
        # 260 questions with 4 equal-length options, 140 with mixed lengths -- A36's split
        lens = [4, 4, 4, 4] if q < 260 else [3, 4, 5, 6]
        texts = [f"q{q}opt{i}" for i in range(4)]          # always distinct strings
        for arm, fn in [
            # never reads the option: score depends only on (question, length)
            ("blind", lambda i, L: base - 1.7 * L),
            # reads the option
            ("reads", lambda i, L: base - 1.7 * L + rng.gauss(0, 0.8)),
            # reads it, but only barely -- the band that should come out SUSPECT
            ("weak", lambda i, L: base - 1.7 * L + rng.gauss(0, 0.02)),
        ]:
            recs.append(dict(arm=arm, question_id=f"q{q}", options=[
                dict(text=t, n_tokens=L, score=fn(i, L))
                for i, (t, L) in enumerate(zip(texts, lens))]))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", help="JSONL of per-question per-arm score records")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="exp/results/s16_content_blind.json")
    args = ap.parse_args()

    if args.self_test:
        recs, src = self_test(), "SELF-TEST (synthetic controls)"
    elif args.records:
        recs = [json.loads(l) for l in Path(args.records).read_text().splitlines() if l.strip()]
        src = args.records
    else:
        ap.error("give --records or --self-test")

    res = screen(recs)
    print(f"source: {src}\n{len(recs)} records\n")
    print("arm              | groups | exact-0 | blind_share | med spread | ratio  | verdict")
    print("-----------------+--------+---------+-------------+------------+--------+--------")
    for arm, r in res.items():
        if r["groups"] == 0:
            print(f"{arm:<16s} |      0 |       - |           - |          - |      - | {r['verdict']}")
            continue
        print(f"{arm:<16s} | {r['groups']:6d} | {r['exact_zero_spread']:7d} | "
              f"{r['blind_share']:11.4f} | {r['median_spread']:10.4f} | "
              f"{r['spread_ratio']:6.4f} | {r['verdict']}")

    if args.self_test:
        ok = (res["blind"]["verdict"] == "CONTENT-BLIND"
              and res["reads"]["verdict"] == "reads the option")
        print(f"\nself-test {'PASSED' if ok else 'FAILED'}: the screen separates a "
              f"content-blind arm from one that reads the option.")
        print("LIMITATION the self-test exposes: when every option in a question has the "
              "SAME\nlength -- exactly A36's 260-item subset -- the (question, length) "
              "group IS the whole\nquestion, so spread_ratio is 1.0 for any arm that reads "
              "the option at all and carries\nno information.  On such a bed only "
              "blind_share and median_spread are diagnostic.\n'weak' (sigma=0.02) shows "
              "this: ratio 1.0, but median spread 0.039 vs 1.578 for a\nhealthy arm -- "
              "read the spread, not the ratio, and compare arms on the same bed.")
        print("\nTo screen a real manuscript, export its per-question records to the JSONL\n"
              "schema in this file's docstring and rerun with --records.  A36's arm should\n"
              "come out CONTENT-BLIND with blind_share ~ 1.0 on its 260 equal-length items.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(source=src, n_records=len(recs), result=res), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
