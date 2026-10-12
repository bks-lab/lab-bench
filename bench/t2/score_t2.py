"""T2 and T9 scorer (plan/t2.md, plan/t9.md), on the Mac after fetching the runs.

  python bench/t2/score_t2.py t2 arm=<out dir> ... [--md results/t2/score.md]
  python bench/t2/score_t2.py t9 arm=<out dir> ... [--md results/t9/score.md]

Gold: cases/route1/massive-de.jsonl (scenario, intent). Unanswered rows count
as wrong and as a flip. Bootstrap unit: the utterance (both questions of a
request move together; in T9 all four variants of an utterance move
together), 1,000 resamples, numpy default_rng(20261009); 95 % percentile
intervals.

t2, per arm: flip rate of the plain pass against the committed route1 run 1
(same prompt, a later day: run-to-run noise), flip rate of the shuffled pass
against the plain pass (option order), accuracy of both passes and the exact
McNemar test between them.
t9, per arm and variant: accuracy, accuracy drop against clean (paired) and
flip rate against clean, on all sampled utterances and on those the variant
actually changed.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SEED = 20261009
B = 1000


def gold():
    g = {}
    for line in open(os.path.join(ROOT, "cases", "route1", "massive-de.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        g[f"massive-{r['id']}-de"] = {"scenario": r["scenario"], "intent": r["intent"]}
    return g


def rows_of(d):
    """result.jsonl of a queue out dir, or a run file (Jev: give run 1 and run 2 joined with a comma)."""
    paths = d.split(",") if not os.path.isdir(d) else [os.path.join(d, "result.jsonl")]
    return [json.loads(x) for p in paths for x in open(p, encoding="utf-8")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(units, fn):
    rng = np.random.default_rng(SEED)
    keys = sorted(units)
    vals = []
    for _ in range(B):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals.append(fn(pick))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rate(pairs):
    return sum(pairs) / len(pairs) if pairs else float("nan")


def t2(arms, g):
    out, lines = {}, ["# T2 score", "", "| arm | flip plain vs run 1 | flip shuffled vs plain | acc plain | acc shuffled | p (McNemar) |", "|---|---|---|---|---|---|"]
    for arm, d in arms.items():
        rows = rows_of(d)
        mp = os.path.join(d, "meta.json")
        meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        ref_path = os.path.join(ROOT, "results", "route1", arm, "2026-10-09-run1.jsonl")
        ref = {(r["line"], r["question"]): r.get("choice") for r in map(json.loads, open(ref_path, encoding="utf-8"))} if os.path.exists(ref_path) else {}
        by = {}
        for r in rows:
            ps = r.get("pass") or ("shuffled" if r.get("shuffled") else "plain")  # Jev rows: run 2 shuffled, run 1 plain
            by.setdefault(r["line"], {}).setdefault(ps, {})[r["question"]] = r.get("choice")
        units = {}
        for line, p in by.items():
            u = []
            for q in ("scenario", "intent"):
                a, s, r1 = p.get("plain", {}).get(q), p.get("shuffled", {}).get(q), ref.get((line, q))
                gq = g[line][q]
                u.append({"rep": (a != r1) if (line, q) in ref else None, "shuf": a != s or a is None or s is None,
                          "acc_p": a == gq, "acc_s": s == gq})
            units[line] = u

        def agg(keys, k):
            v = [x[k] for key in keys for x in units[key] if x[k] is not None]
            return rate(v)
        keys = list(units)
        res = {k: agg(keys, k) for k in ("rep", "shuf", "acc_p", "acc_s")}
        res["ci"] = {k: boot(units, lambda ks, k=k: agg(ks, k)) for k in ("rep", "shuf")}
        b = sum(1 for key in keys for x in units[key] if x["acc_p"] and not x["acc_s"])
        c = sum(1 for key in keys for x in units[key] if x["acc_s"] and not x["acc_p"])
        res["mcnemar"] = {"plain_only": b, "shuffled_only": c, "p": mcnemar(b, c)}
        res["questions"] = sum(len(u) for u in units.values())
        res["runtime"] = meta.get("runtime")
        out[arm] = res
        lines.append(f"| {arm} | {100 * res['rep']:.1f} % [{100 * res['ci']['rep'][0]:.1f}, {100 * res['ci']['rep'][1]:.1f}] | "
                     f"{100 * res['shuf']:.1f} % [{100 * res['ci']['shuf'][0]:.1f}, {100 * res['ci']['shuf'][1]:.1f}] | "
                     f"{100 * res['acc_p']:.1f} % | {100 * res['acc_s']:.1f} % | {res['mcnemar']['p']:.3f} |")
    return out, lines


def t9(arms, g):
    out, lines = {}, ["# T9 score", "", "| arm | variant | acc | drop vs clean | 95 % CI | flip vs clean | changed only: drop |", "|---|---|---|---|---|---|---|"]
    for arm, d in arms.items():
        rows = rows_of(d)
        units = {}
        for r in rows:
            if r.get("pass", "plain") != "plain":
                continue
            base, v = r["line"].split("~")
            units.setdefault(base, {}).setdefault(v, {})[r["question"]] = r.get("choice")
        texts = {}
        for x in open(os.path.join(ROOT, "cases", "t9", "requests.jsonl"), encoding="utf-8"):
            r = json.loads(x)
            texts[(r["base_line"], r["variant"])] = r["text"]
        for key, u in units.items():
            for v in ("typo", "translit", "colloquial"):
                u["_same_" + v] = texts.get((key, v)) == texts.get((key, "clean"))
        res = {}
        for v in ("typo", "translit", "colloquial"):
            def stats(keys, only_changed=False, v=v):
                acc_c = acc_v = flips = n = 0
                for key in keys:
                    u = units[key]
                    if only_changed and u.get("_same_" + v):
                        continue
                    for q in ("scenario", "intent"):
                        gq = g[key][q]
                        c, x = u["clean"].get(q), u[v].get(q)
                        acc_c += c == gq
                        acc_v += x == gq
                        flips += c != x or x is None
                        n += 1
                return (acc_v / n, (acc_c - acc_v) / n, flips / n) if n else (float("nan"),) * 3
            acc, drop, flip = stats(list(units))
            ci = boot(units, lambda ks: stats(ks)[1])
            _, drop_ch, _ = stats(list(units), True)
            res[v] = {"acc": acc, "drop": drop, "drop_ci": ci, "flip": flip, "drop_changed_only": drop_ch,
                      "changed": sum(1 for u in units.values() if not u["_same_" + v])}
            lines.append(f"| {arm} | {v} | {100 * acc:.1f} % | {100 * drop:.1f} pt | [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
                         f"{100 * flip:.1f} % | {100 * drop_ch:.1f} pt ({res[v]['changed']} utt.) |")
        out[arm] = res
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["t2", "t9"])
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)
    out, lines = (t2 if a.test == "t2" else t9)(arms, gold())
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text)
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
