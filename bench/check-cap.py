#!/usr/bin/env python3
"""Independent recompute of the C1, C2 and C3 headline numbers, written
without the bench's scoring code (no bench/score-cap.py, no pytrec_eval, no
jiwer, no bench/lib/norm_de.py, no bench/c3/fields.py): it reads the raw
result rows and the gold directly, recomputes each number with its own
implementation and prints it next to the value in results/<cN>/summary.json.

    uv run --with pyarrow python -I bench/check-cap.py [--qrels .cache/c1data]

C1 needs the qrels parquet files of both sets (not committed; MIRACL is
CC BY-SA). cases/c1/fetch.py downloads them; their SHA-256 is in
cases/c1/manifest.json and is checked here. Without them the C1 checks are
skipped with a note. Exit 1 when any number differs beyond its tolerance.
"""
import argparse
import glob
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rows(test, armdir, tag=""):
    files = sorted(glob.glob(str(ROOT / f"results/{test}/{armdir}/*-run1{tag}.jsonl")))
    return jl(files[-1])


def summary(test):
    return json.loads((ROOT / f"results/{test}/summary.json").read_text(encoding="utf-8"))


checks = []


def check(name, mine, rep, tol):
    ok = rep is not None and abs(mine - rep) <= tol
    checks.append(ok)
    print(f"{'ok  ' if ok else 'DIFF'} {name}: recomputed {mine:.4f} | report {rep}")


# ---------------------------------------------------------------- C1

def ndcg10(ranked, rel):
    # linear gain (pytrec_eval / MTEB), log2(rank + 1) discount, ideal from the qrels
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def c1(qdir):
    import pyarrow.parquet as pq
    man = json.loads((ROOT / "cases/c1/manifest.json").read_text(encoding="utf-8"))
    qrels = {}
    for s in ("miracl", "germandpr"):
        p = Path(qdir) / s / "qrels.parquet"
        if not p.exists():
            print(f"skip C1: {p} missing (cases/c1/fetch.py)")
            return
        want = man["sets"][s]["files"]["qrels"]["sha256"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            print(f"note: {p} sha256 {got[:12]} differs from the manifest {want[:12]}")
        q = {}
        for r in pq.read_table(p).to_pylist():
            q.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]
        qrels[s] = q
    sm = {a["arm"]: a for a in summary("c1")["arms"]}
    for arm in ("qwen3-emb-8b", "bge-m3", "e5-large", "bm25"):
        rs = rows("c1", arm)
        for s in ("miracl", "germandpr"):
            vals, rec, mrr = [], [], []
            for r in rs:
                if r["set"] != s:
                    continue
                ranked = [d for d, _ in r["top100"]]
                rel = qrels[s].get(r["qid"], {})
                vals.append(ndcg10(ranked, rel))
                pos = {d for d, g in rel.items() if g > 0}
                rec.append(len(pos & set(ranked[:100])) / len(pos) if pos else 0.0)
                first = next((i for i, d in enumerate(ranked[:10]) if d in pos), None)
                mrr.append(0.0 if first is None else 1 / (first + 1))
            nd = sum(vals) / len(vals)
            if s == "miracl":
                check(f"C1 {arm} MIRACL nDCG@10 (n {len(vals)})", nd, sm[arm]["quality"]["value"], 0.002)
                check(f"C1 {arm} MIRACL Recall@100", sum(rec) / len(rec), sm[arm]["quality"]["recall100"], 0.002)
                check(f"C1 {arm} MIRACL MRR@10", sum(mrr) / len(mrr), sm[arm]["quality"]["mrr10"], 0.002)
            else:
                check(f"C1 {arm} GermanDPR nDCG@10 (n {len(vals)})", nd, sm[arm]["quality"]["germandpr_ndcg10"], 0.002)
    g = summary("c1")["gates"]["bge-m3"]
    bge = sm["bge-m3"]["quality"]["value"]
    print(f"     C1 gate: bge-m3 {bge} against 0.5759 +- 0.015 -> {'passed' if abs(bge - 0.5759) <= 0.015 else 'FAILED'} (runner: {g['passed']})")


# ---------------------------------------------------------------- C2

def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return s.split()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def c2():
    sm = {a["arm"]: a for a in summary("c2")["arms"]}
    for arm in sm:
        e = n = 0
        for r in rows("c2", arm):
            ref, hyp = norm_text(r["ref"]), norm_text(r["hyp"])
            e += lev(ref, hyp); n += len(ref)
        check(f"C2 {arm} WER % (from raw ref and hyp)", 100 * e / n, sm[arm]["quality"]["value"], 0.02)
    w = sm["canary-1b-v2"]["quality"]["value"]
    print(f"     C2 gate: canary-1b-v2 {w} against 4.40 +- 1.5 -> {'passed' if abs(w - 4.40) <= 1.5 else 'FAILED'}")


# ---------------------------------------------------------------- C3

def money(v):
    s = re.sub(r"[^0-9,.\-]", "", str(v))
    if not s:
        return None
    if "," in s and (s.rfind(",") > s.rfind(".")):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    if s.count(".") > 1:
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + "." + tail
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return None


def iso(v):
    s = str(v).strip()
    m = re.fullmatch(r"(\d{4})-?(\d\d)-?(\d\d)", s)
    if m:
        return "-".join(m.groups())
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}" if m else s


def cell(field, v):
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null")):
        return None
    v = str(v)
    if field in ("net_total", "vat_total", "gross_total", "amount_due"):
        return money(v)
    if field in ("issue_date", "due_date"):
        return iso(v)
    if field in ("seller_name", "buyer_name"):
        return " ".join(unicodedata.normalize("NFC", v).split()).casefold()
    if field in ("seller_vat_id", "iban", "currency"):
        return "".join(v.split()).upper()
    return v.strip()


def c3():
    gold = {g["id"]: g for g in jl(ROOT / "reference/c3/gold.jsonl")}
    fields = [k for k in next(iter(gold.values())) if k != "id"]
    sm = {a["arm"]: a for a in summary("c3")["arms"]}
    for arm, a in sm.items():
        if not a["finished"]:
            continue
        right = total = 0
        for r in rows("c3", arm.replace("+", "-")):
            p = r["pred"] if isinstance(r["pred"], dict) else {}
            for f in fields:
                total += 1
                right += cell(f, p.get(f)) == cell(f, gold[r["id"]][f])
        check(f"C3 {arm} field accuracy ({right} of {total})", right / total, a["quality"]["value"], 0.0005)


ap = argparse.ArgumentParser()
ap.add_argument("--qrels", default=str(ROOT / ".cache" / "c1data"))
a = ap.parse_args()
c1(a.qrels)
c2()
c3()
print(f"{sum(checks)} of {len(checks)} checks agree")
sys.exit(0 if all(checks) else 1)
