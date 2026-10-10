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
            failed = not isinstance(r["pred"], dict)  # failed call: all 12 cells wrong (results/c3/method.md)
            p = r["pred"] if not failed else {}
            for f in fields:
                total += 1
                right += (not failed) and cell(f, p.get(f)) == cell(f, gold[r["id"]][f])
        check(f"C3 {arm} field accuracy ({right} of {total})", right / total, a["quality"]["value"], 0.0005)


# ---------------------------------------------------------------- C4 to C8
# Each block runs only when results/<cN>/summary.json exists.


def have(test):
    return (ROOT / f"results/{test}/summary.json").exists()


def c4():
    """Part A: the totals of the per-item verdicts of BFCL's checker (the check itself
    needs the gorilla checkout). Part B: parse and validate every answer here with
    jsonschema directly, without bench/c4/check_ast.py."""
    if not have("c4"):
        return
    import jsonschema
    from jsonschema import validators
    schemas = {json.loads(l)["id"]: json.loads(json.loads(l)["schema"]) for l in open(ROOT / "cases/c4/schemas.jsonl", encoding="utf-8")}

    def first_json(t):
        t = (t or "").strip()
        cands = [t]
        m = re.search(r"```(?:json|JSON)?\s*\n(.*?)\n?```", t, re.S)
        if m:
            cands.append(m.group(1))
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            cands.append(t[i:j + 1])
        for c in cands:
            try:
                return json.loads(c), True
            except Exception:
                pass
        return None, False

    for a in summary("c4")["arms"]:
        if not a["finished"]:
            continue
        ver = rows("c4", a["arm"], "-verdicts")
        fc = [v["valid"] for v in ver if v["mode"] == "fc"]
        check(f"C4 {a['arm']} fc AST accuracy (verdicts)", sum(fc) / len(fc), a["quality"]["value"], 0.0005)
        raw = rows("c4", a["arm"])
        for mode, key in (("free", "schema_validity_free"), ("constrained", "schema_validity_constrained")):
            ok = n = 0
            for r in raw:
                if r["mode"] != mode:
                    continue
                if mode == "constrained" and r.get("error") and re.search(r"format|schema|grammar", r["error"], re.I):
                    continue  # not constrainable, scored in free mode only
                n += 1
                if r.get("error"):
                    continue
                val, parsed = first_json(r.get("content"))
                if not parsed:
                    continue
                sch = schemas[r["id"]]
                cls = validators.validator_for(sch, default=jsonschema.Draft202012Validator)
                ok += cls(sch).is_valid(val)
            check(f"C4 {a['arm']} {mode} schema validity ({ok} of {n})", ok / n, a["quality"][key], 0.0005)


def squad_norm(s, arts=r"\b(a|an|the)\b"):
    s = "".join(ch for ch in s.lower() if ch not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return " ".join(re.sub(arts, " ", s).split())


def f1_one(p, g):
    from collections import Counter
    pt, gt = squad_norm(p).split(), squad_norm(g).split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if not same:
        return 0.0
    pr, rc = same / len(pt), same / len(gt)
    return 2 * pr * rc / (pr + rc)


def c6():
    if not have("c6"):
        return
    sample = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "cases/c6/sample.jsonl", encoding="utf-8")}
    for a in summary("c6")["arms"]:
        if not a["finished"]:
            continue
        rr = {r["id"]: r for r in rows("c6", a["arm"])}
        f = [max(f1_one(rr[i]["answer"] or "", g) for g in sample[i]["answers"]) for i in sample]
        check(f"C6 {a['arm']} F1 on {len(f)} questions", sum(f) / len(f), a["quality"]["value"], 0.0005)


def c7():
    if not have("c7"):
        return
    trials = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "cases/c7/trials.jsonl", encoding="utf-8")}
    for a in summary("c7")["arms"]:
        if not a.get("per_length"):
            continue
        rr = rows("c7", a["arm"].replace("+", "-"))
        for L, p in a["per_length"].items():
            got = []
            for r in rr:
                if str(r["length"]) != L:
                    continue
                codes = set(re.findall(r"(?<!\d)\d{7}(?!\d)", r.get("answer") or ""))
                t = trials[r["id"]]
                got.append((not r["error"]) and set(t["asked_codes"]) <= codes and not codes & set(t["other_codes"]))
            check(f"C7 {a['arm']} accuracy at {L}", sum(got) / len(got), p["accuracy"], 0.0005)


def c8():
    """Aggregate rates from the rows and the section wall times: generated tokens of
    the timed requests per configuration, divided by the summed wall time."""
    if not have("c8"):
        return
    for a in summary("c8")["arms"]:
        if not a.get("configurations"):
            continue
        meta = json.loads((ROOT / f"results/c8/{a['arm']}/meta.json").read_text(encoding="utf-8"))
        rr = rows("c8", a["arm"])
        for key in ("chat-short@1", "chat-short@16", "decide@16"):
            if key not in a["configurations"]:
                continue
            wl, lv = key.split("@")
            vals = []
            for s in meta["sections"]:
                if s["workload"] != wl or s["level"] != int(lv):
                    continue
                sel = [r for r in rr if r["workload"] == wl and r["level"] == int(lv) and r["rep"] == s["rep"] and not r["warmup"]]
                if wl == "decide":
                    vals.append(len(sel) / s["wall_s"] * 3600)
                else:
                    vals.append(sum(r["eval_count"] or 0 for r in sel if not r["error"]) / s["wall_s"])
            rep = a["configurations"][key]["decisions_per_h_mean" if wl == "decide" else "agg_gen_tok_s_mean"]
            check(f"C8 {a['arm']} {key} mean of 3", sum(vals) / len(vals), rep, 0.01 if wl != "decide" else 1.0)


ap = argparse.ArgumentParser()
ap.add_argument("--qrels", default=str(ROOT / ".cache" / "c1data"))
a = ap.parse_args()
c1(a.qrels)
c2()
c3()
c4()
c6()
c7()
c8()
print(f"{sum(checks)} of {len(checks)} checks agree")
sys.exit(0 if all(checks) else 1)
