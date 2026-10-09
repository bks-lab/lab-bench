#!/usr/bin/env python3
"""Independent recompute of five T12 numbers (plan/t12.md), written without
the bench's JavaScript libraries: reads the raw result rows, the gold files
and the requests directly and prints each number next to the report value.

    python3 -I bench/check-t12.py
"""
import json
import glob
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows(path_glob):
    out = []
    for f in sorted(glob.glob(str(ROOT / path_glob))):
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def run1(pv, arm):
    # the plain run-1 file of the newest date (split files of that date merged), no tags
    files = sorted(p for p in glob.glob(str(ROOT / f"results/{pv}/{arm}/*-run1*.jsonl")) if re.search(r"-run1(-(de|en))?\.jsonl$", p))
    latest = Path(files[-1]).name[:10]
    return [r for f in files if Path(f).name.startswith(latest) for r in rows(Path(f).relative_to(ROOT))]


gold = {}
for lang in ("de", "en"):
    for g in rows(f"reference/route1/gold.{lang}.jsonl"):
        gold[(g["line"], g["question"])] = g
nopts = {}
for r in rows("requests/route1/requests.jsonl"):
    for qid, q in r["request"]["questions"].items():
        nopts[(r["line"], qid)] = len(q["criteria"])


def route_items(arm, keep):
    got = {(r["line"], r["question"]): r.get("choice") for r in run1("route1", arm)}
    return {k: got.get(k) is not None and got.get(k) == g["choice"] for k, g in gold.items() if keep(k, g) and k in got}


def acc(items):
    return round(100 * sum(items.values()) / len(items), 1), len(items)


def norm(s):
    s = re.sub(r"\s+", " ", str(s).lower()).strip()
    return re.sub(r"[\s.,;:!?¿¡'\"`)\]]+$", "", s).strip()


def slot_f1(arm, lang):
    g = {x["line"]: x["slots"] for x in rows(f"reference/ex1/gold.{lang}.jsonl")}
    tp = fp = fn = 0
    for r in run1("ex1", arm):
        if r["lang"] != lang or r["line"] not in g:
            continue
        G, P = {}, {}
        for s in g[r["line"]]:
            k = (s["type"], norm(s["text"])); G[k] = G.get(k, 0) + 1
        for s in r.get("slots") or []:
            k = (s["type"], norm(s["text"])); P[k] = P.get(k, 0) + 1
        for k, c in G.items():
            h = min(c, P.get(k, 0)); tp += h; fn += c - h
        for k, c in P.items():
            fp += c - min(c, G.get(k, 0))
    p, rc = tp / (tp + fp), tp / (tp + fn)
    return round(100 * 2 * p * rc / (p + rc), 1)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(md, section, arm, col):
    sec = md.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    head = None
    for line in sec.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if head is None:
            head = cells
        elif cells[0] == arm:
            return cells[head.index(col)]


route_md = (ROOT / "results/route1/score.md").read_text(encoding="utf-8")
ex_md = (ROOT / "results/ex1/score.md").read_text(encoding="utf-8")
massive = lambda q, lang: (lambda k, g: k[0].startswith("massive-") and g["lang"] == lang and k[1] == q)
checks = []

a, n = acc(route_items("qwen3-4b", massive("scenario", "de")))
checks.append(("qwen3-4b MASSIVE scenario de, acc", f"{a} % (n {n})", report(route_md, "massive scenario, de", "qwen3-4b", "acc")))
a, n = acc(route_items("qwen3-32b", lambda k, g: massive("intent", "en")(k, g) and nopts[k] > 1))
checks.append(("qwen3-32b MASSIVE intent 2+ en, acc", f"{a} % (n {n})", report(route_md, "massive intent, 2+ options, en", "qwen3-32b", "acc")))
a, n = acc(route_items("qwen3-14b", lambda k, g: k[0].startswith("fastdec-") and nopts[k] <= 26))
checks.append(("qwen3-14b fast-decisions <= 26 labels en, acc", f"{a} % (n {n})", report(route_md, "fastdec, tasks with 26 options or fewer, en", "qwen3-14b", "acc")))
checks.append(("qwen3-30b-a3b ex1 slot F1 de", f"{slot_f1('qwen3-30b-a3b', 'de')} %", report(ex_md, "de", "qwen3-30b-a3b", "F1")))
L = route_items("qwen3-32b", massive("scenario", "de")); S = route_items("qwen3-4b", massive("scenario", "de"))
b = sum(1 for k in L if k in S and L[k] and not S[k]); c = sum(1 for k in L if k in S and S[k] and not L[k])
tm = json.loads((ROOT / "results/toolmap.json").read_text(encoding="utf-8"))
row = next(r for r in tm["rows"] if r["id"] == "route1-scenario-de")
checks.append(("McNemar qwen3-32b vs qwen3-4b, scenario de", f"b {b}, c {c}, p {mcnemar_p(b, c):.3g}", str(row.get("local_size_contrast"))))

ok = True
for name, mine, rep in checks:
    print(f"{name}: recomputed {mine} | report {rep}")
print("all checks printed; compare by eye (the McNemar line compares b, c and p with the map's local_size_contrast)")
