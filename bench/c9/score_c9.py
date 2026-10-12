"""C9 scorer (plan/c9.md, Metrics; notes of 2026-10-12).

  python bench/c9/score_c9.py regex=<out dir> gliner-multi-pii=<out dir> ... [--audit private/c9/gold-audit.json]
        [--md results/c9/score.md] [--json results/c9/score.json]

Per arm, over the 476 documents (spans after c9lib.norm, the seven types of
reference/c9/labels.json):
- primary: type-agnostic span F1 with exact boundaries; recall next to it
- typed strict F1 per type (descriptive); character masking recall (share of
  gold characters covered by any predicted span)
- 95 % bootstrap intervals, unit document, 1,000 resamples,
  numpy default_rng(20261009); paired bootstrap of the F1 difference between
  every arm and the best arm on the same resamples
- the same numbers on the 40 audited documents against the audited gold, and
  the audited gold's own precision and recall, once the audit file exists
- machine: median ms per document, characters per second, and (when the out
  dir holds the queue's gpu.csv) peak VRAM over idle and GPU energy per 1,000
  documents over the timed section
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import c9lib  # noqa: E402

SEED = 20261009
B = 1000


def counts(gold, pred):
    """tp, fp, fn for sets of spans."""
    return len(gold & pred), len(pred - gold), len(gold - pred)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def read_rows(d):
    rows = {}
    for line in open(os.path.join(d, "result.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def pred_spans(doc, row):
    out = set()
    for p in (row or {}).get("pred") or []:
        b = c9lib.norm(doc["text"], p["start"], p["end"])
        if b:
            out.add((b[0], b[1], p["type"]))
    return out


def per_doc(docs, golds, rows, types):
    """Per document: [tp, fp, fn] untyped, per type, and gold/covered characters."""
    m = []
    for d in docs:
        g = golds[d["id"]]
        p = pred_spans(d, rows.get(d["id"]))
        gu, pu = {(s, e) for s, e, _ in g}, {(s, e) for s, e, _ in p}
        v = list(counts(gu, pu))
        for t in types:
            v += counts({x for x in g if x[2] == t}, {x for x in p if x[2] == t})
        gch = set()
        for s, e in gu:
            gch.update(range(s, e))
        pch = set()
        for s, e in pu:
            pch.update(range(s, e))
        v += [len(gch), len(gch & pch)]
        m.append(v)
    return np.array(m, dtype=float)


def metrics(arr, types):
    tp, fp, fn = arr[:3]
    p, r, f = f1(tp, fp, fn)
    out = {"precision": p, "recall": r, "f1": f, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    out["typed_f1"] = {t: f1(*arr[3 + 3 * i: 6 + 3 * i])[2] for i, t in enumerate(types)}
    out["typed_gold"] = {t: int(arr[3 + 3 * i] + arr[5 + 3 * i]) for i, t in enumerate(types)}
    out["mask_recall"] = arr[-1] / arr[-2] if arr[-2] else 0.0
    return out


def boot(mats, types, idx):
    """Bootstrap intervals of f1, recall, precision, mask recall per arm and the paired F1 difference to the best arm."""
    stats = {a: {"f1": [], "recall": [], "precision": [], "mask_recall": []} for a in mats}
    for ix in idx:
        for a, m in mats.items():
            s = metrics(m[ix].sum(0), types)
            for k in stats[a]:
                stats[a][k].append(s[k])
    ci = {a: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in st.items()} for a, st in stats.items()}
    return ci, stats


def gpu_machine(d, meta):
    """Peak VRAM over the idle value before load and GPU energy in the timed section, from the queue's gpu.csv."""
    p = os.path.join(d, "gpu.csv")
    if not os.path.exists(p):
        return {}
    timed = next((s for s in meta.get("sections", []) if s["name"] == "timed"), None)
    load = next((s for s in meta.get("sections", []) if s["name"] == "load"), None)
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for r in csv.reader(f):
            if not r or not r[0][:4].isdigit():
                continue
            try:
                ts = dt.datetime.strptime(r[0].strip()[:19], "%Y/%m/%d %H:%M:%S")
                rows.append((ts, float(r[1].split()[0]), float(r[2].split()[0])))
            except (ValueError, IndexError):
                continue
    if not rows:
        return {}

    def local(s):
        return dt.datetime.fromisoformat(s).replace(tzinfo=None)

    out = {}
    if load:
        idle = [m for ts, m, _ in rows if ts <= local(load["start"])]
        out["idle_mib"] = idle[-1] if idle else rows[0][1]
    peak = max(m for _, m, _ in rows)
    out["peak_mib"] = peak
    if "idle_mib" in out:
        out["peak_over_idle_mib"] = peak - out["idle_mib"]
    if timed:
        t0, t1 = local(timed["start"]), local(timed["end"])
        w = [pw for ts, _, pw in rows if t0 <= ts <= t1]
        if w:
            out["energy_j"] = sum(w)
            out["mean_power_w"] = sum(w) / len(w)
            out["energy_j_per_1000_docs"] = sum(w) / timed["items"] * 1000
    return out


def audited_gold(docs_by_id, labels, audit):
    """Audited gold of the audited documents, and the original gold's precision and recall against it."""
    out, kept, orig_n, missed_n = {}, 0, 0, 0
    for did, a in audit["docs"].items():
        d = docs_by_id[did]
        g = c9lib.gold(d, labels)
        verdicts = a.get("spans", {})
        spans = set()
        for i, (s, e, t) in enumerate(g):
            v = verdicts.get(str(i), {"verdict": "ok"})
            orig_n += 1
            if v["verdict"] == "ok":
                spans.add((s, e, t))
                kept += 1
            elif v["verdict"] == "type" and v.get("type") in labels["types"]:
                spans.add((s, e, v["type"]))
                kept += 1
            elif v["verdict"] == "boundary" and v.get("text"):
                hits = [h for h in c9lib.find_all(d["text"], v["text"]) if h[0] < e and s < h[1]]
                for hs, he in hits[:1]:
                    b = c9lib.norm(d["text"], hs, he)
                    if b:
                        spans.add((b[0], b[1], t))
        for m in a.get("missed", []):
            for hs, he in c9lib.find_all(d["text"], m["text"]):
                b = c9lib.norm(d["text"], hs, he)
                if b:
                    spans.add((b[0], b[1], m["type"]))
                    missed_n += 1
        out[did] = sorted(spans)
    prec = kept / orig_n if orig_n else None
    rec = kept / (kept + missed_n) if kept + missed_n else None
    return out, {"documents": len(out), "gold_spans": orig_n, "kept": kept, "missed_added": missed_n,
                 "gold_precision": prec, "gold_recall": rec,
                 "gold_noisy": (prec is not None and prec < 0.9) or (rec is not None and rec < 0.9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="arm=out_dir")
    ap.add_argument("--audit", default=os.path.join(ROOT, "private", "c9", "gold-audit.json"))
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    man, docs, labels = c9lib.load(ROOT)
    types = labels["types"]
    golds = {d["id"]: set(c9lib.gold(d, labels)) for d in docs}
    dirs = dict(x.split("=", 1) for x in a.arms)
    rows = {arm: read_rows(d) for arm, d in dirs.items()}
    mats = {arm: per_doc(docs, golds, rows[arm], types) for arm in dirs}
    res = {arm: metrics(m.sum(0), types) for arm, m in mats.items()}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(docs), len(docs)) for _ in range(B)]
    ci, st = boot(mats, types, idx)
    best = max(res, key=lambda k: res[k]["f1"])
    for arm in res:
        res[arm]["ci"] = ci[arm]
        diff = np.array(st[best]["f1"]) - np.array(st[arm]["f1"])
        res[arm]["paired_vs_best"] = {"best": best, "diff": res[best]["f1"] - res[arm]["f1"],
                                      "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                      "p": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))) if arm != best else None}
        meta = json.load(open(os.path.join(dirs[arm], "meta.json"), encoding="utf-8"))
        secs = [r["seconds"] for r in rows[arm].values()]
        chars = sum(r["chars"] for r in rows[arm].values())
        res[arm]["machine"] = {"median_ms_per_doc": float(np.median(secs)) * 1000 if secs else None,
                               "chars_per_s": chars / sum(secs) if sum(secs) else None,
                               "errors": sum(1 for r in rows[arm].values() if r.get("error")),
                               "documents": len(rows[arm]), **gpu_machine(dirs[arm], meta)}
        if meta.get("gate"):
            res[arm]["gate"] = meta["gate"]
    out = {"design": "plan/c9.md", "documents": len(docs), "bootstrap": {"unit": "document", "B": B, "seed": SEED},
           "arms": res}
    if os.path.exists(a.audit):
        audit = json.load(open(a.audit, encoding="utf-8"))
        ag, gq = audited_gold({d["id"]: d for d in docs}, labels, audit)
        sub = [d for d in docs if d["id"] in ag]
        out["audit"] = gq
        agset = {k: set(v) for k, v in ag.items()}
        amats = {arm: per_doc(sub, agset, rows[arm], types) for arm in dirs}
        out["audited"] = {arm: metrics(m.sum(0), types) for arm, m in amats.items()}
        aidx = [rng.integers(0, len(sub), len(sub)) for _ in range(B)]
        aci, ast = boot(amats, types, aidx)
        abest = max(out["audited"], key=lambda k: out["audited"][k]["f1"])
        for arm, r in out["audited"].items():
            r["ci"] = aci[arm]
            dd = np.array(ast[abest]["f1"]) - np.array(ast[arm]["f1"])
            r["paired_vs_best"] = {"best": abest, "diff": out["audited"][abest]["f1"] - r["f1"],
                                   "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
    lines = ["# C9 score", "", f"{len(docs)} documents, spans after the boundary rule of `bench/c9/c9lib.py`. "
             "95 % bootstrap intervals over documents (1,000 resamples, seed 20261009).", "",
             "| arm | span F1 | recall | precision | mask recall | paired diff to best | ms/doc | errors |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, r in sorted(res.items(), key=lambda kv: -kv[1]["f1"]):
        c = r["ci"]
        pv = r["paired_vs_best"]
        diff = "best" if arm == best else f"{pv['diff']:.3f} [{pv['ci'][0]:.3f}, {pv['ci'][1]:.3f}]"
        ms = r["machine"]["median_ms_per_doc"]
        lines.append(f"| {arm} | {r['f1']:.3f} [{c['f1'][0]:.3f}, {c['f1'][1]:.3f}] | {r['recall']:.3f} [{c['recall'][0]:.3f}, {c['recall'][1]:.3f}] "
                     f"| {r['precision']:.3f} | {r['mask_recall']:.3f} | {diff} | {ms:.1f} | {r['machine']['errors']} |")
    lines += ["", "Typed strict F1 (descriptive):", "", "| arm | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for arm, r in res.items():
        lines.append(f"| {arm} | " + " | ".join(f"{r['typed_f1'][t]:.2f}" for t in types) + " |")
    if "audit" in out:
        gq = out["audit"]
        lines += ["", f"Gold audit: {gq['documents']} documents, gold precision {gq['gold_precision']:.3f}, "
                  f"gold recall {gq['gold_recall']:.3f}" + (" (gold noisy: numbers descriptive only)" if gq["gold_noisy"] else ""), ""]
        lines += ["Audited gold, 40 documents (95 % bootstrap over documents, B 1,000, seed 20261009; paired difference to the best arm on these documents):", "",
                  "| arm | span F1 | recall | precision | paired diff to best |", "|---|---|---|---|---|"]
        for arm, r in sorted(out["audited"].items(), key=lambda kv: -kv[1]["f1"]):
            c, pv = r["ci"], r["paired_vs_best"]
            diff = "best" if pv["best"] == arm else f"{pv['diff']:.3f} [{pv['ci'][0]:.3f}, {pv['ci'][1]:.3f}]"
            lines.append(f"| {arm} | {r['f1']:.3f} [{c['f1'][0]:.3f}, {c['f1'][1]:.3f}] | {r['recall']:.3f} [{c['recall'][0]:.3f}, {c['recall'][1]:.3f}] | {r['precision']:.3f} | {diff} |")
    text = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
