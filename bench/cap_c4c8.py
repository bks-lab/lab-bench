"""Scoring of the capability tests C4, C6, C7 and C8 as preregistered in
plan/c4.md, plan/c6.md, plan/c7.md and plan/c8.md (with their dated notes),
called by bench/score-cap.py. Same conventions as C1 to C3: everything is
recomputed from the committed rows (C4 from the per-item verdicts of
bench/c4/check_ast.py, which runs BFCL's checker), 95 % bootstrap intervals
with 1,000 resamples of numpy default_rng(20261009), a paired bootstrap
against the best arm, GPU energy and peak VRAM from the 1 s GPU log of each
job.

Each score_cN() returns (summary dict, score.md text, capability row) or
None when results/cN/ has no scored arm yet.
"""
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("score_cap", ROOT / "bench" / "score-cap.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)
sys.path.insert(0, str(ROOT / "bench" / "c6"))
import squad  # noqa: E402

r3, ci_mean, ci_median, paired_mean, overlap = sc.r3, sc.ci_mean, sc.ci_median, sc.paired_mean, sc.overlap
CARD_MIB = 24564
LIC = {"licence": "Apache-2.0", "licence_class": "open source"}

C4_ARMS = {
    "qwen3-8b": {"model": "qwen3:8b (Qwen/Qwen3-8B)", **LIC},
    "qwen3-14b": {"model": "qwen3:14b (Qwen/Qwen3-14B)", **LIC},
    "qwen3.8-27b": {"model": "qwen3.8:27b (Qwen/Qwen3.8-27B)", **LIC},
    "gemma4-12b": {"model": "gemma4:12b (google/gemma-4-12B-it)", **LIC},
    "gemma4-26b": {"model": "gemma4:26b (google/gemma-4-26B-A4B-it)", **LIC},
    "mistral-small-3.2": {"model": "mistral-small3.2:24b (mistralai/Mistral-Small-3.2-24B-Instruct-2506)", **LIC},
    "gpt-oss-20b": {"model": "gpt-oss:20b (openai/gpt-oss-20b)", **LIC},
    "granite4-3b": {"model": "granite4:micro, same weights as granite4:3b (ibm-granite/granite-4.0-micro)", **LIC},
}
C6_ARMS = {
    "gelectra-large": {"model": "deepset/gelectra-large-germanquad (trained reader)", "licence": "MIT", "licence_class": "open source", "params_b": 0.335, "params_source": "model card (ELECTRA large)", "reader": True},
    "qwen3-8b": C4_ARMS["qwen3-8b"],
    "qwen3.8-27b": C4_ARMS["qwen3.8-27b"],
    "gemma4-12b": C4_ARMS["gemma4-12b"],
    "gemma4-26b": C4_ARMS["gemma4-26b"],
    "mistral-small-3.2": C4_ARMS["mistral-small-3.2"],
    "winnow-12b": {"model": "hf.co/EldanRing/Winnow-12B:Q8_0 (Gemma 4 12B fine-tune)", **LIC},
}
C7_BASE = {k: C4_ARMS[k] for k in ("qwen3-8b", "qwen3-14b", "gemma4-12b", "gemma4-26b", "mistral-small-3.2", "qwen3.8-27b", "gpt-oss-20b")}
C7_ARMS = {**C7_BASE, **{f"{k}-kvq8": {**v, "kv": "q8_0", "base": k} for k, v in C7_BASE.items()}}
C8_BASE = {"granite4-3b": C4_ARMS["granite4-3b"], "qwen3-8b": C4_ARMS["qwen3-8b"], "gemma4-12b": C4_ARMS["gemma4-12b"],
           "winnow-12b": C6_ARMS["winnow-12b"], "qwen3-14b": C4_ARMS["qwen3-14b"], "gpt-oss-20b": C4_ARMS["gpt-oss-20b"],
           "mistral-small-3.2": C4_ARMS["mistral-small-3.2"], "gemma4-26b": C4_ARMS["gemma4-26b"], "qwen3.8-27b": C4_ARMS["qwen3.8-27b"]}
C8_ARMS = {**C8_BASE,
           "vllm-qwen3-14b": {"model": "vLLM, Qwen/Qwen3-14B-AWQ (AWQ INT4, official)", **LIC, "engine": "vllm"},
           "vllm-gemma4-12b": {"model": "vLLM, cyankiwi/gemma-4-12B-it-AWQ-INT4 (AWQ INT4, community build)", **LIC, "engine": "vllm"}}
ARMS = {"c4": C4_ARMS, "c6": C6_ARMS, "c7": C7_ARMS, "c8": C8_ARMS}


# ---------------------------------------------------------------- shared

def present(test, arms):
    out = []
    for a in arms:
        p = ROOT / "results" / test / sc.arm_dir(a)
        if (p / "queue-status.json").exists():
            out.append(a)
    return out


def load(test, arm, tag=""):
    p = ROOT / "results" / test / sc.arm_dir(arm)
    meta = json.load(open(p / "meta.json", encoding="utf-8")) if (p / "meta.json").exists() else None
    status = json.load(open(p / "queue-status.json", encoding="utf-8"))
    gpu = sc.gpu_samples(ROOT / "results" / test / "gpu" / f"{sc.arm_dir(arm)}.csv")
    rows = sc.latest_rows(test, arm, tag)
    return meta, status, gpu, rows


def model_cols(meta):
    m = (meta or {}).get("model") or {}
    det = m.get("details") or {}
    ps = None
    try:
        ps = float(det.get("parameter_size", "").rstrip("B"))
    except ValueError:
        pass
    return {"ollama_digest": (m.get("digest") or "")[:12] or None, "quant": det.get("quantization_level"), "params_b": ps}


def offloaded_ps(ps_list):
    """True when Ollama keeps part of the model off the card (size_vram below size)."""
    for m in ps_list or []:
        if m.get("size") and m.get("size_vram") is not None and m["size_vram"] < m["size"] * 0.99:
            return True
    return False


def vram_job(status):
    g = status["gpu"]
    return round(g["peak_over_idle_mib"] / 1024, 2), round(g["peak_mem_mib"] / 1024, 2), g["idle_mem_mib"]


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired right/wrong lists."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p": p}


def vs_best(best_items, items, best_ci, ci):
    diff, dci, p = paired_mean(best_items, items)
    return {"diff": diff, "diff_ci95": dci, "p": p, "intervals_overlap": overlap(ci, best_ci),
            "difference_shown": p < 0.05 or not overlap(ci, best_ci)}


def pick(arms, value, vram, sep):
    """Roadmap rule (plan/ROADMAP.md): highest value; if the next is not separated, the one with less peak VRAM."""
    order = sorted(arms, key=lambda a: -value(a))
    top, nxt = order[0], (order[1] if len(order) > 1 else None)
    shown, tie = top, None
    if nxt and not sep(nxt):
        if vram(nxt) < vram(top):
            shown = nxt
        tie = {"top_by_value": top, "next": nxt, "note": "intervals overlap and the paired test does not separate the two; the one with less peak VRAM is shown"}
    small = [a for a in arms if a == top or not sep(a)]
    smallest = min(small, key=vram)
    return shown, top, tie, smallest, order


def f3(x):
    return "" if x is None else f"{x:.3f}"


def fx(x, k=1):
    return "" if x is None else f"{x:.{k}f}"


def cis(c, k=3):
    return f"[{c[0]:.{k}f}, {c[1]:.{k}f}]"


# ---------------------------------------------------------------- C4

def score_c4():
    arms = present("c4", C4_ARMS)
    if not arms:
        return None
    gold_ids = [json.loads(l)["id"] for l in open(ROOT / "cases" / "c4" / "bfcl.jsonl", encoding="utf-8")]
    cat = {json.loads(l)["id"]: json.loads(l)["category"] for l in open(ROOT / "cases" / "c4" / "bfcl.jsonl", encoding="utf-8")}
    sch_ids = [json.loads(l)["id"] for l in open(ROOT / "cases" / "c4" / "schemas.jsonl", encoding="utf-8")]
    subset = {i: i.split(":")[0] for i in sch_ids}
    res = {}
    for a in arms:
        meta, status, gpu, rows = load("c4", a)
        ver = sc.latest_rows("c4", a, "-verdicts")
        if meta is None or rows is None or ver is None:
            res[a] = {"finished": False, "status": status}
            continue
        V = {(v["mode"], v["id"]): v for v in ver}
        R = {(r["mode"], r["id"]): r for r in rows}
        assert len(V) == len(R), a
        secs = {s["name"]: s for s in meta["sections"]}
        modes = {}
        for m, ids in (("fc", gold_ids), ("prompt", gold_ids), ("free", sch_ids), ("constrained", sch_ids)):
            if not all((m, i) in V for i in ids):
                continue
            rr = [R[(m, i)] for i in ids]
            ms = [r["seconds"] * 1000 for r in rr if not r["error"]]
            ev = sum(r["eval_count"] or 0 for r in rr)
            evs = sum(r["eval_s"] or 0 for r in rr)
            sec = secs.get(m)
            e = sc.energy_j(gpu, sec["start"], sec["end"]) if sec else None
            d = {"items": len(ids), "ms_median": float(np.median(ms)) if ms else None,
                 "ms_median_ci95": ci_median(ms) if ms else None, "out_tok_s": ev / evs if evs else None,
                 "energy_j_per_1000": (e / len(ids) * 1000) if e is not None else None,
                 "failed_calls": sum(1 for r in rr if r["error"]),
                 "capped": sum(1 for r in rr if r.get("done_reason") == "length"),
                 "truncated_prompts": sum(1 for r in rr if r.get("truncated_prompt")),
                 "offloaded": offloaded_ps((sec or {}).get("ps_during"))}
            if m in ("fc", "prompt"):
                ok = [float(V[(m, i)]["valid"]) for i in ids]
                d.update({"acc": float(np.mean(ok)), "acc_ci95": ci_mean(ok), "items_ok": ok,
                          "per_category": {c: float(np.mean([V[(m, i)]["valid"] for i in ids if cat[i] == c]))
                                           for c in ("simple_python", "multiple", "parallel", "parallel_multiple")},
                          "unparseable": float(np.mean([str(V[(m, i)].get("error_type", "")).startswith(("ast_decoder", "call_failed")) for i in ids]))})
            else:
                usable = [i for i in ids if V[(m, i)]["valid"] is not None]
                ok = [float(V[(m, i)]["valid"]) for i in usable]
                d.update({"usable": len(usable), "not_constrainable": sum(1 for i in ids if V[(m, i)].get("not_constrainable")),
                          "validity": float(np.mean(ok)) if ok else None, "validity_ci95": ci_mean(ok) if ok else None,
                          "ok_by_id": {i: V[(m, i)]["valid"] for i in usable},
                          "per_subset": {s: float(np.mean([V[(m, i)]["valid"] for i in usable if subset[i] == s]))
                                         for s in sorted(set(subset.values())) if any(subset[i] == s for i in usable)},
                          "unparseable": float(np.mean([V[(m, i)].get("parsed") is False for i in usable])) if usable else None})
            modes[m] = d
        res[a] = {"finished": True, "meta": meta, "status": status, "modes": modes}
        if "fc" in modes and "prompt" in modes:
            res[a]["fc_vs_prompt"] = mcnemar([bool(x) for x in modes["fc"]["items_ok"]], [bool(x) for x in modes["prompt"]["items_ok"]])
        if "free" in modes and "constrained" in modes:
            ids = [i for i in modes["constrained"]["ok_by_id"] if i in modes["free"]["ok_by_id"]]
            fr = [bool(modes["free"]["ok_by_id"][i]) for i in ids]
            co = [bool(modes["constrained"]["ok_by_id"][i]) for i in ids]
            res[a]["constrained_vs_free"] = {**mcnemar(co, fr), "schemas": len(ids), "free_on_same": float(np.mean(fr)) if fr else None,
                                             "constrained": float(np.mean(co)) if co else None,
                                             "coverage": len(modes["constrained"]["ok_by_id"]) / len(sch_ids)}
            fm, cm = modes["free"]["ms_median"], modes["constrained"]["ms_median"]
            res[a]["slowdown_constrained"] = cm / fm if fm and cm else None
    done = [a for a in res if res[a]["finished"] and "fc" in res[a]["modes"]]
    if not done:
        return None
    best = max(done, key=lambda a: res[a]["modes"]["fc"]["acc"])
    for a in done:
        if a != best:
            res[a]["vs_best"] = vs_best(res[best]["modes"]["fc"]["items_ok"], res[a]["modes"]["fc"]["items_ok"],
                                        res[best]["modes"]["fc"]["acc_ci95"], res[a]["modes"]["fc"]["acc_ci95"])
    vram = lambda a: res[a]["status"]["gpu"]["peak_over_idle_mib"]  # noqa: E731
    shown, top, tie, smallest, order = pick(done, lambda a: res[a]["modes"]["fc"]["acc"], vram,
                                            lambda a: a == best or res[a]["vs_best"]["difference_shown"])
    gate = None
    if "qwen3-14b" in done:
        pc = res["qwen3-14b"]["modes"]["fc"]["per_category"]
        gate = {"arm": "qwen3-14b", "mode": "fc", "simple_python": 100 * pc["simple_python"], "multiple": 100 * pc["multiple"],
                "reference": {"simple_python": 95.25, "multiple": 93.00}, "tolerance_points": 5,
                "passed": abs(100 * pc["simple_python"] - 95.25) <= 5 and abs(100 * pc["multiple"] - 93.00) <= 5,
                "source": "https://gorilla.cs.berkeley.edu/data_non_live.csv, Qwen3-14B (FC), read 2026-10-09"}
    arms_out = []
    for a in order + [x for x in res if x not in order]:
        r = res[a]
        info = C4_ARMS[a]
        d = {"arm": a, "model": info["model"], "licence": info["licence"], "licence_class": info["licence_class"], "finished": r["finished"]}
        if r["finished"]:
            vp, vt, _ = vram_job(r["status"])
            md = r["modes"]
            d.update(model_cols(r["meta"]))
            d["quality"] = {
                "metric": "AST accuracy, BFCL v3 non-live (1,000 items), fc mode (native tool calling)",
                "value": r3(md["fc"]["acc"]), "ci95": [r3(x) for x in md["fc"]["acc_ci95"]],
                "prompt_mode": r3(md["prompt"]["acc"]) if "prompt" in md else None,
                "prompt_mode_ci95": [r3(x) for x in md["prompt"]["acc_ci95"]] if "prompt" in md else None,
                "per_category_fc": {k: r3(v) for k, v in md["fc"]["per_category"].items()},
                "per_category_prompt": {k: r3(v) for k, v in md["prompt"]["per_category"].items()} if "prompt" in md else None,
                "unparseable_fc": r3(md["fc"]["unparseable"]), "unparseable_prompt": r3(md["prompt"]["unparseable"]) if "prompt" in md else None,
                "fc_vs_prompt": {**r["fc_vs_prompt"], "p": round(r["fc_vs_prompt"]["p"], 4)} if "fc_vs_prompt" in r else None,
                "schema_validity_free": r3(md["free"]["validity"]) if "free" in md else None,
                "schema_validity_free_ci95": [r3(x) for x in md["free"]["validity_ci95"]] if "free" in md else None,
                "schema_validity_constrained": r3(md["constrained"]["validity"]) if "constrained" in md else None,
                "schema_validity_constrained_ci95": [r3(x) for x in md["constrained"]["validity_ci95"]] if "constrained" in md else None,
                "constrained_coverage": r3(r["constrained_vs_free"]["coverage"]) if "constrained_vs_free" in r else None,
                "constrained_vs_free": ({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["constrained_vs_free"].items()}
                                        if "constrained_vs_free" in r else None),
                "per_subset_free": {k: r3(v) for k, v in md["free"]["per_subset"].items()} if "free" in md else None,
                "per_subset_constrained": {k: r3(v) for k, v in md["constrained"]["per_subset"].items()} if "constrained" in md else None,
                **({"vs_best": {"best": best, "diff": r3(r["vs_best"]["diff"]), "diff_ci95": [r3(x) for x in r["vs_best"]["diff_ci95"]],
                                "p": round(r["vs_best"]["p"], 3), "difference_shown": r["vs_best"]["difference_shown"]}} if "vs_best" in r else {}),
            }
            d["speed"] = {m: {"ms_median": r3(md[m]["ms_median"]), "ms_median_ci95": [r3(x) for x in md[m]["ms_median_ci95"]] if md[m]["ms_median_ci95"] else None,
                              "out_tok_s": r3(md[m]["out_tok_s"]), "failed_calls": md[m]["failed_calls"], "capped": md[m]["capped"],
                              "truncated_prompts": md[m]["truncated_prompts"]} for m in md}
            d["speed"]["slowdown_constrained_over_free"] = r3(r.get("slowdown_constrained"))
            d["energy"] = {m: {"j_per_1000_items": round(md[m]["energy_j_per_1000"]) if md[m]["energy_j_per_1000"] is not None else None} for m in md}
            d["vram_peak_gb"], d["vram_peak_total_gb"] = vp, vt
            d["fits_24gb"] = "offloaded" if (any(md[m]["offloaded"] for m in md) or r["status"]["gpu"]["peak_mem_mib"] >= CARD_MIB - 512) else True
        else:
            d["unfinished"] = {"reason": r["status"].get("reason")}
        arms_out.append(d)
    by = {x["arm"]: x for x in arms_out}
    summary = {"test": "C4", "design": "plan/c4.md", "report": "results/c4/findings.md", "best_by_fc_ast_accuracy": top,
               "gate": gate, "arms": arms_out}
    S = by[shown]
    row = {"id": "C4", "capability": "Structured output and tool calling", "design": "plan/c4.md", "report": "results/c4/findings.md",
           "best_local_model": shown, "best_by_value": top, "tie": tie, "licence_class": C4_ARMS[shown]["licence_class"],
           "quality": {"metric": "AST accuracy, BFCL v3 non-live, 1,000 items, native tool calling", "value": S["quality"]["value"], "ci95": S["quality"]["ci95"],
                       "schema_validity_constrained": S["quality"]["schema_validity_constrained"],
                       "schema_validity_free": S["quality"]["schema_validity_free"],
                       "baseline": "prompt mode of the same model", "baseline_value": S["quality"]["prompt_mode"],
                       "reference": "BFCL leaderboard Qwen3-14B (FC) simple 95.25 % (sanity gate)",
                       "reference_ours": r3(gate["simple_python"] / 100) if gate else None},
           "speed": {"unit": "ms per tool-calling item, median (one call at a time)", "value": S["speed"]["fc"]["ms_median"],
                     "out_tok_s": S["speed"]["fc"]["out_tok_s"]},
           "vram_peak_gb": S["vram_peak_gb"],
           "energy": {"unit": "GPU joules per 1,000 tool-calling items", "value": S["energy"]["fc"]["j_per_1000_items"]},
           "fits_24gb": S["fits_24gb"],
           "smallest_within_interval": {"arm": smallest, "value": by[smallest]["quality"]["value"], "vram_peak_gb": by[smallest]["vram_peak_gb"],
                                        "rule": "smallest peak VRAM among the arms the paired test does not separate from the best"},
           "unfinished": [x["arm"] for x in arms_out if not x["finished"]]}
    return summary, c4_md(summary), row


def c4_md(s):
    L = ["# C4 score: structured output and tool calling", "",
         "Generated by `bench/score-cap.py score` from `results/c4/<arm>/*-run1.jsonl` and the per-item verdicts of `bench/c4/check_ast.py` (BFCL's own checker, gorilla `6ea57973c7a6`). Design: `plan/c4.md`, method `results/c4/method.md`.",
         "Intervals: 95 % bootstrap over items (1,000 resamples, seed 20261009). Paired test against the best arm: bootstrap of the per-item difference in fc mode. Mode against mode of one model: exact McNemar test.", ""]
    g = s["gate"]
    if g:
        L += ["## Sanity gate", "", f"qwen3-14b in fc mode: simple_python {g['simple_python']:.2f} % (leaderboard 95.25), multiple {g['multiple']:.2f} % (leaderboard 93.00), tolerance 5 points: {'passed' if g['passed'] else 'failed'}.", ""]
    L += ["## Part A: BFCL non-live, AST accuracy (1,000 items)", "",
          "| arm | params (B) | fc | 95 % CI | prompt | 95 % CI | fc vs prompt p | diff to best (fc) | p | difference shown | unparseable fc | unparseable prompt |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        if not x["finished"]:
            L.append(f"| {x['arm']} | | not finished | | | | | | | | | |")
            continue
        q = x["quality"]; v = q.get("vs_best")
        L.append(f"| {x['arm']} | {x['params_b']} | {q['value']:.3f} | {cis(q['ci95'])} | {f3(q['prompt_mode'])} | {cis(q['prompt_mode_ci95']) if q['prompt_mode_ci95'] else ''} | "
                 f"{q['fc_vs_prompt']['p']:.3f} | " + (f"{-v['diff']:+.3f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} |" if v else "best | | |")
                 + f" {f3(q['unparseable_fc'])} | {f3(q['unparseable_prompt'])} |")
    L += ["", "## Part A per category (fc / prompt)", "", "| arm | simple_python | multiple | parallel | parallel_multiple |", "|---|---|---|---|---|"]
    for x in s["arms"]:
        if x["finished"]:
            q = x["quality"]
            L.append(f"| {x['arm']} | " + " | ".join(f"{q['per_category_fc'][c]:.3f} / {q['per_category_prompt'][c]:.3f}" for c in ("simple_python", "multiple", "parallel", "parallel_multiple")) + " |")
    L += ["", "## Part B: JSONSchemaBench, schema validity (600 schemas)", "",
          "| arm | free | 95 % CI | constrained | 95 % CI | coverage of constrained | free on the same schemas | constrained vs free p | slowdown constrained/free |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        if x["finished"]:
            q = x["quality"]; cv = q["constrained_vs_free"] or {}
            L.append(f"| {x['arm']} | {f3(q['schema_validity_free'])} | {cis(q['schema_validity_free_ci95']) if q['schema_validity_free_ci95'] else ''} | {f3(q['schema_validity_constrained'])} | "
                     f"{cis(q['schema_validity_constrained_ci95']) if q['schema_validity_constrained_ci95'] else ''} | {f3(q['constrained_coverage'])} | {f3(cv.get('free_on_same'))} | "
                     f"{cv.get('p', '')} | {f3(x['speed'].get('slowdown_constrained_over_free'))} |")
    L += ["", "## Part B per subset (free / constrained)", ""]
    subs = None
    for x in s["arms"]:
        if x["finished"] and x["quality"]["per_subset_free"]:
            subs = list(x["quality"]["per_subset_free"])
            break
    if subs:
        L += ["| arm | " + " | ".join(subs) + " |", "|---|" + "---|" * len(subs)]
        for x in s["arms"]:
            if x["finished"]:
                q = x["quality"]
                L.append(f"| {x['arm']} | " + " | ".join(f"{q['per_subset_free'].get(k, 0):.2f} / {(q['per_subset_constrained'] or {}).get(k, 0):.2f}" for k in subs) + " |")
    L += ["", "## Machine", "",
          "| arm | ms per item fc (median) | 95 % CI | ms prompt | ms free | ms constrained | output tok/s fc | J per 1,000 items fc | peak VRAM GB (over idle) | peak VRAM GB (total) | capped answers (A/B) | truncated prompts (B) | fits on 24 GB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        if x["finished"]:
            sp = x["speed"]
            cap_a = sum(sp[m]["capped"] for m in ("fc", "prompt") if m in sp)
            cap_b = sum(sp[m]["capped"] for m in ("free", "constrained") if m in sp)
            tr = sum(sp[m]["truncated_prompts"] for m in ("free", "constrained") if m in sp)
            L.append(f"| {x['arm']} | {fx(sp['fc']['ms_median'], 0)} | {cis(sp['fc']['ms_median_ci95'], 0) if sp['fc']['ms_median_ci95'] else ''} | {fx(sp['prompt']['ms_median'], 0)} | {fx(sp['free']['ms_median'], 0)} | {fx(sp['constrained']['ms_median'], 0)} | "
                     f"{fx(sp['fc']['out_tok_s'])} | {x['energy']['fc']['j_per_1000_items']} | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} | {cap_a}/{cap_b} | {tr} | {'yes' if x['fits_24gb'] is True else x['fits_24gb']} |")
    L += ["", "Peak VRAM from nvidia-smi over the whole job; energy is GPU only, the timed section of a mode divided by its items.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- C6

def score_c6():
    arms = present("c6", C6_ARMS)
    if not arms:
        return None
    sample = [json.loads(l) for l in open(ROOT / "cases" / "c6" / "sample.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in sample]
    G = {r["id"]: r for r in sample}
    base = {r["id"]: r["answer"] for r in sc.rd_jsonl(ROOT / "results" / "c6" / "baseline" / "result.jsonl")}
    bf1 = [squad.best(squad.f1, base[i], G[i]["answers"]) for i in ids]
    res = {}
    for a in arms:
        meta, status, gpu, rows = load("c6", a)
        if meta is None or rows is None:
            res[a] = {"finished": False, "status": status}
            continue
        R = {r["id"]: r for r in rows}
        assert all(i in R for i in ids), a
        ans = [R[i]["answer"] or "" for i in ids]
        f1 = [squad.best(squad.f1, p, G[i]["answers"]) for p, i in zip(ans, ids)]
        em = [squad.best(squad.em, p, G[i]["answers"]) for p, i in zip(ans, ids)]
        f1de = [squad.best(squad.f1, p, G[i]["answers"], "de") for p, i in zip(ans, ids)]
        faith = [float(bool(p.strip()) and p.strip().lower() != "keine antwort" and p.strip() in G[i]["context"]) for p, i in zip(ans, ids)]
        kA = [float(p.strip().lower().strip(".") == "keine antwort") for p in ans]
        secs = [R[i]["seconds"] for i in ids]
        timed = [s for s in meta["sections"] if s["name"] == "timed"][0]
        e = sc.energy_j(gpu, timed["start"], timed["end"])
        n_timed = timed["items"]
        res[a] = {"finished": True, "meta": meta, "status": status, "f1_items": f1,
                  "f1": float(np.mean(f1)), "f1_ci95": ci_mean(f1), "em": float(np.mean(em)), "em_ci95": ci_mean(em),
                  "f1_de": float(np.mean(f1de)), "faith": float(np.mean(faith)), "keine": int(sum(kA)),
                  "errors": sum(1 for i in ids if R[i]["error"]),
                  "ms_median": 1000 * float(np.median(secs)), "ms_ci95": [1000 * x for x in ci_median(secs)],
                  "energy_j_per_1000": e / n_timed * 1000, "offloaded": offloaded_ps(timed.get("ps_during"))}
    done = [a for a in res if res[a]["finished"]]
    if not done:
        return None
    best = max(done, key=lambda a: res[a]["f1"])
    for a in done:
        if a != best:
            res[a]["vs_best"] = vs_best(res[best]["f1_items"], res[a]["f1_items"], res[best]["f1_ci95"], res[a]["f1_ci95"])
    vram = lambda a: res[a]["status"]["gpu"]["peak_over_idle_mib"]  # noqa: E731
    shown, top, tie, smallest, order = pick(done, lambda a: res[a]["f1"], vram, lambda a: a == best or res[a]["vs_best"]["difference_shown"])
    gate = (res.get("gelectra-large") or {}).get("meta", {}).get("gate") if "gelectra-large" in done else None
    arms_out = []
    for a in order + [x for x in res if x not in order]:
        r = res[a]; info = C6_ARMS[a]
        d = {"arm": a, "model": info["model"], "licence": info["licence"], "licence_class": info["licence_class"], "finished": r["finished"],
             "trained_reader": bool(info.get("reader"))}
        if r["finished"]:
            vp, vt, _ = vram_job(r["status"])
            d.update(model_cols(r["meta"]) if not info.get("reader") else {"params_b": info["params_b"], "params_source": info["params_source"]})
            d["quality"] = {"metric": "F1 (SQuAD v1.1), GermanQuAD test sample, 1,000 questions", "value": r3(r["f1"]), "ci95": [r3(x) for x in r["f1_ci95"]],
                            "em": r3(r["em"]), "em_ci95": [r3(x) for x in r["em_ci95"]], "f1_german_articles": r3(r["f1_de"]),
                            "span_faithfulness": r3(r["faith"]), "keine_antwort": r["keine"], "failed_or_unparsed": r["errors"],
                            "skill_vs_baseline": r3((r["f1"] - float(np.mean(bf1))) / (1 - float(np.mean(bf1)))),
                            **({"vs_best": {"best": best, "diff": r3(r["vs_best"]["diff"]), "diff_ci95": [r3(x) for x in r["vs_best"]["diff_ci95"]],
                                            "p": round(r["vs_best"]["p"], 3), "difference_shown": r["vs_best"]["difference_shown"]}} if "vs_best" in r else {})}
            d["speed"] = {"ms_per_question_median": r3(r["ms_median"]), "ms_ci95": [r3(x) for x in r["ms_ci95"]],
                          "note": "one question per call; the reader answers all 2,204 test questions, the median is over the 1,000 of the sample"}
            d["energy"] = {"j_per_1000_questions": round(r["energy_j_per_1000"]), "note": "GPU only, timed section divided by its questions"}
            d["vram_peak_gb"], d["vram_peak_total_gb"] = vp, vt
            d["fits_24gb"] = "offloaded" if (r["offloaded"] or r["status"]["gpu"]["peak_mem_mib"] >= CARD_MIB - 512) else True
        else:
            d["unfinished"] = {"reason": r["status"].get("reason")}
        arms_out.append(d)
    by = {x["arm"]: x for x in arms_out}
    summary = {"test": "C6", "design": "plan/c6.md", "report": "results/c6/findings.md", "best_by_f1": top, "gate": gate,
               "baseline": {"f1": r3(float(np.mean(bf1))), "f1_ci95": [r3(x) for x in ci_mean(bf1)], "rule": "results/c6/method.md, lexical baseline"},
               "arms": arms_out}
    S = by[shown]
    row = {"id": "C6", "capability": "German reading comprehension (answer from a passage)", "design": "plan/c6.md", "report": "results/c6/findings.md",
           "best_local_model": shown, "best_by_value": top, "tie": tie, "licence_class": C6_ARMS[shown]["licence_class"],
           "quality": {"metric": "F1 on 1,000 GermanQuAD test questions", "value": S["quality"]["value"], "ci95": S["quality"]["ci95"],
                       "baseline": "best-overlap sentence of the passage", "baseline_value": summary["baseline"]["f1"],
                       "reference": "GELECTRA-large card F1 88.1 on the full test set (sanity gate)",
                       "reference_ours": r3(gate["value"] / 100) if gate else None},
           "speed": {"unit": "ms per question, median", "value": S["speed"]["ms_per_question_median"]},
           "vram_peak_gb": S["vram_peak_gb"],
           "energy": {"unit": "GPU joules per 1,000 questions", "value": S["energy"]["j_per_1000_questions"]},
           "fits_24gb": S["fits_24gb"],
           "smallest_within_interval": {"arm": smallest, "value": by[smallest]["quality"]["value"], "vram_peak_gb": by[smallest]["vram_peak_gb"],
                                        "rule": "smallest peak VRAM among the arms the paired test does not separate from the best"},
           "unfinished": [x["arm"] for x in arms_out if not x["finished"]]}
    return summary, c6_md(summary), row


def c6_md(s):
    L = ["# C6 score: German reading comprehension", "",
         "Generated by `bench/score-cap.py score` from `results/c6/<arm>/*-run1.jsonl`, `cases/c6/sample.jsonl` and the GPU logs. Design: `plan/c6.md`, method `results/c6/method.md`.",
         "F1 and EM: SQuAD v1.1 definitions (`bench/c6/squad.py`), maximum over the three answers, recomputed from the answers. Intervals: 95 % bootstrap over the 1,000 questions (1,000 resamples, seed 20261009). Paired test: bootstrap of the per-question F1 difference to the best arm.", ""]
    g = s["gate"]
    if g:
        L += ["## Sanity gate", "", f"gelectra-large on all {g['questions']} test questions: F1 {g['value']:.2f} against the card's 88.1, tolerance 2 points: {'passed' if g['passed'] else 'failed'}.", ""]
    L += [f"Lexical baseline (best-overlap sentence): F1 {s['baseline']['f1']:.3f} {cis(s['baseline']['f1_ci95'])}.", "",
          "## Quality", "",
          "| arm | params (B) | F1 | 95 % CI | EM | F1 (German articles) | span faithfulness | keine Antwort | diff to best | p | difference shown |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        if not x["finished"]:
            L.append(f"| {x['arm']} | | not finished | | | | | | | | |")
            continue
        q = x["quality"]; v = q.get("vs_best")
        L.append(f"| {x['arm']}{' (trained reader)' if x['trained_reader'] else ''} | {x.get('params_b')} | {q['value']:.3f} | {cis(q['ci95'])} | {q['em']:.3f} | {q['f1_german_articles']:.3f} | {q['span_faithfulness']:.3f} | {q['keine_antwort']} | "
                 + (f"{-v['diff']:+.3f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} |" if v else "best | | |"))
    L += ["", "## Machine", "", "| arm | ms per question (median) | 95 % CI | J per 1,000 questions | peak VRAM GB (over idle) | peak VRAM GB (total) | fits on 24 GB |", "|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        if x["finished"]:
            sp = x["speed"]
            L.append(f"| {x['arm']} | {sp['ms_per_question_median']:.0f} | {cis(sp['ms_ci95'], 0)} | {x['energy']['j_per_1000_questions']} | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} | {'yes' if x['fits_24gb'] is True else x['fits_24gb']} |")
    L += ["", "Peak VRAM from nvidia-smi over the whole job; energy is GPU only.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- C7

def score_c7():
    arms = present("c7", C7_ARMS)
    if not arms:
        return None
    trials = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "cases" / "c7" / "trials.jsonl", encoding="utf-8")}
    spec = importlib.util.spec_from_file_location("gen_c7", ROOT / "bench" / "gen-c7.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    res = {}
    for a in arms:
        meta, status, gpu, rows = load("c7", a)
        if meta is None or rows is None:
            res[a] = {"finished": False, "status": status}
            continue
        idle = status["gpu"]["idle_mem_mib"]
        per = {}
        for L in meta["lengths"]:
            rr = [r for r in rows if r["length"] == L]
            if not rr:
                continue
            for r in rr:  # rejudge from the answer
                ok = (not r["error"]) and gen.judge(r["answer"], trials[r["id"]])
                assert ok == r["right"], (a, r["id"])
            secs = [s for s in meta["sections"] if s["length"] == L]
            right = {r["id"]: float(r["right"]) for r in rr}
            e = sum(sc.energy_j(gpu, s["start"], s["end"]) for s in secs)
            pk = max((sc.peak_mib(gpu, s["start"], s["end"]) or 0) for s in secs)
            ttft = [r["prompt_eval_s"] for r in rr if r["prompt_eval_s"]]
            per[L] = {"n": len(rr), "acc": float(np.mean(list(right.values()))), "acc_ci95": ci_mean(list(right.values())), "items": right,
                      "per_task": {t: float(np.mean([r["right"] for r in rr if r["task"] == t])) for t in ("single", "multikey", "multivalue") if any(r["task"] == t for r in rr)},
                      "per_depth": {str(d): float(np.mean([r["right"] for r in rr if r["depth"] == d])) for d in (0.1, 0.5, 0.9) if any(r["depth"] == d for r in rr)},
                      "ttft_s_median": float(np.median(ttft)) if ttft else None, "ttft_ci95": ci_median(ttft) if ttft else None,
                      "prompt_tokens_median": float(np.median([r["prompt_eval_count"] for r in rr if r["prompt_eval_count"]])),
                      "length_off": sum(1 for r in rr if r["length_off"]), "errors": sum(1 for r in rr if r["error"]),
                      "vram_peak_gb": round((pk - idle) / 1024, 2), "vram_peak_total_gb": round(pk / 1024, 2),
                      "offloaded": any(offloaded_ps(s.get("ps")) for s in secs) or pk >= CARD_MIB - 512,
                      "energy_j_per_prompt": e / len(rr)}
        res[a] = {"finished": bool(per) and (meta.get("gate") or {}).get("passed", True) and len(per) == len(meta["lengths"]),
                  "partial": bool(per), "meta": meta, "status": status, "per": per, "gate": meta.get("gate")}
        ok_lengths = [L for L in sorted(per) if per[L]["acc"] >= 0.9]
        res[a]["longest_90"] = max(ok_lengths) if ok_lengths else None
    base_arms = [a for a in res if res[a]["finished"] and not C7_ARMS[a].get("kv") and 32768 in res[a]["per"]]
    if not base_arms:
        return None
    best = max(base_arms, key=lambda a: res[a]["per"][32768]["acc"])
    for a in base_arms:
        if a != best:
            ids = sorted(res[best]["per"][32768]["items"])
            b = [res[best]["per"][32768]["items"][i] for i in ids]
            x = [res[a]["per"][32768]["items"][i] for i in ids]
            res[a]["vs_best"] = vs_best(b, x, res[best]["per"][32768]["acc_ci95"], res[a]["per"][32768]["acc_ci95"])
    vram = lambda a: res[a]["per"][32768]["vram_peak_gb"]  # noqa: E731
    shown, top, tie, smallest, order = pick(base_arms, lambda a: res[a]["per"][32768]["acc"], vram,
                                            lambda a: a == best or res[a]["vs_best"]["difference_shown"])
    arms_out = []
    for a in order + [x for x in res if x not in order]:
        r = res[a]; info = C7_ARMS[a]
        d = {"arm": a, "model": info["model"], "licence": info["licence"], "licence_class": info["licence_class"],
             "kv_cache": "q8_0" if info.get("kv") else "f16", "finished": r["finished"], "gate": r.get("gate")}
        if r.get("partial"):
            d.update(model_cols(r["meta"]))
            d["per_length"] = {str(L): {"n": p["n"], "accuracy": r3(p["acc"]), "ci95": [r3(x) for x in p["acc_ci95"]],
                                        "per_task": {k: r3(v) for k, v in p["per_task"].items()}, "per_depth": {k: r3(v) for k, v in p["per_depth"].items()},
                                        "ttft_s_median": r3(p["ttft_s_median"]), "ttft_ci95": [r3(x) for x in p["ttft_ci95"]] if p["ttft_ci95"] else None,
                                        "prompt_tokens_median": p["prompt_tokens_median"], "length_off": p["length_off"], "errors": p["errors"],
                                        "vram_peak_gb": p["vram_peak_gb"], "vram_peak_total_gb": p["vram_peak_total_gb"],
                                        "offloaded": p["offloaded"], "energy_j_per_prompt": round(p["energy_j_per_prompt"])}
                                for L, p in sorted(r["per"].items())}
            d["longest_length_at_90"] = r["longest_90"]
            if 32768 in r["per"]:
                d["quality"] = {"metric": "accuracy at 32k tokens, 180 prompts (3 tasks x 3 depths x 20)", "value": r3(r["per"][32768]["acc"]),
                                "ci95": [r3(x) for x in r["per"][32768]["acc_ci95"]],
                                **({"vs_best": {"best": best, "diff": r3(r["vs_best"]["diff"]), "diff_ci95": [r3(x) for x in r["vs_best"]["diff_ci95"]],
                                                "p": round(r["vs_best"]["p"], 3), "difference_shown": r["vs_best"]["difference_shown"]}} if "vs_best" in r else {})}
        else:
            d["unfinished"] = {"reason": r["status"].get("reason")}
        arms_out.append(d)
    by = {x["arm"]: x for x in arms_out}
    summary = {"test": "C7", "design": "plan/c7.md", "report": "results/c7/findings.md", "best_by_accuracy_32k": top, "arms": arms_out}
    S = by[shown]
    p32 = S["per_length"]["32768"]
    row = {"id": "C7", "capability": "Long context, 4k to 32k tokens", "design": "plan/c7.md", "report": "results/c7/findings.md",
           "best_local_model": shown, "best_by_value": top, "tie": tie, "licence_class": C7_ARMS[shown]["licence_class"],
           "quality": {"metric": "needle accuracy at 32k tokens (180 prompts)", "value": p32["accuracy"], "ci95": p32["ci95"],
                       "at_4k": S["per_length"]["4096"]["accuracy"], "at_16k": S["per_length"]["16384"]["accuracy"],
                       "baseline": "the same model at 4k", "baseline_value": S["per_length"]["4096"]["accuracy"],
                       "reference": "none published; internal check: 95 % on single at 4k (sanity gate)"},
           "speed": {"unit": "seconds of prompt processing at 32k tokens, median", "value": p32["ttft_s_median"]},
           "vram_peak_gb": p32["vram_peak_gb"],
           "energy": {"unit": "GPU joules per 32k-token prompt", "value": p32["energy_j_per_prompt"]},
           "fits_24gb": "offloaded" if p32["offloaded"] else True,
           "smallest_within_interval": {"arm": smallest, "value": by[smallest]["per_length"]["32768"]["accuracy"],
                                        "vram_peak_gb": by[smallest]["per_length"]["32768"]["vram_peak_gb"],
                                        "rule": "smallest peak VRAM at 32k among the arms the paired test does not separate from the best"},
           "unfinished": [x["arm"] for x in arms_out if not x["finished"]]}
    return summary, c7_md(summary), row


def c7_md(s):
    L = ["# C7 score: long context on one card", "",
         "Generated by `bench/score-cap.py score` from `results/c7/<arm>/*-run1.jsonl`, `cases/c7/trials.jsonl` and the GPU logs. Design: `plan/c7.md`, generator `bench/gen-c7.py`.",
         "Every answer is judged again with the generator's rule. Intervals: 95 % bootstrap over prompts (1,000 resamples, seed 20261009). Paired test at 32k: bootstrap of the per-prompt difference to the best arm (same trials for every model).", "",
         "## Sanity gate (single at 4k, 60 prompts, at least 95 %)", "", "| arm | right | passed |", "|---|---|---|"]
    for x in s["arms"]:
        g = x.get("gate")
        if g:
            L.append(f"| {x['arm']} | {g['right']} of {g['prompts']} | {'yes' if g['passed'] else 'no'} |")
    L += ["", "## Accuracy per length (180 prompts each)", "",
          "| arm | KV cache | 4k | 16k | 32k | 95 % CI at 32k | diff to best at 32k | p | difference shown | longest length >= 90 % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        pl = x.get("per_length") or {}
        q = x.get("quality") or {}; v = q.get("vs_best")
        cell = lambda L_: f"{pl[L_]['accuracy']:.3f}" if L_ in pl else ""  # noqa: E731
        L.append(f"| {x['arm']} | {x['kv_cache']} | {cell('4096')} | {cell('16384')} | {cell('32768')} | {cis(pl['32768']['ci95']) if '32768' in pl else ''} | "
                 + ((f"{-v['diff']:+.3f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} |" if v else ("best | | |" if '32768' in pl and x['kv_cache'] == 'f16' else "| | |")))
                 + f" {x.get('longest_length_at_90') or ''} |")
    L += ["", "## Per task and depth at 32k (descriptive)", "", "| arm | single | multikey | multivalue | depth 10 % | depth 50 % | depth 90 % |", "|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        p = (x.get("per_length") or {}).get("32768")
        if p:
            L.append(f"| {x['arm']} | " + " | ".join(f"{p['per_task'].get(t, 0):.2f}" for t in ("single", "multikey", "multivalue")) + " | "
                     + " | ".join(f"{p['per_depth'].get(d, 0):.2f}" for d in ("0.1", "0.5", "0.9")) + " |")
    L += ["", "## Machine per length", "",
          "| arm | length | prompt tokens (median) | prompt processing s (median) | peak VRAM GB (over idle) | peak VRAM GB (total) | offloaded | J per prompt | prompts off target |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        for Lk, p in (x.get("per_length") or {}).items():
            L.append(f"| {x['arm']} | {Lk} | {p['prompt_tokens_median']:.0f} | {f3(p['ttft_s_median'])} | {p['vram_peak_gb']:.2f} | {p['vram_peak_total_gb']:.2f} | {'yes' if p['offloaded'] else 'no'} | {p['energy_j_per_prompt']} | {p['length_off']} |")
    L += ["", "Prompt processing time is Ollama's prompt evaluation time per call. Offloaded: Ollama kept part of the model off the card (/api/ps) or nvidia-smi reached the card. Energy is GPU only.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- C8

def score_c8():
    arms = present("c8", C8_ARMS)
    if not arms:
        return None
    res = {}
    for a in arms:
        meta, status, gpu, rows = load("c8", a)
        if meta is None or rows is None or not meta.get("sections"):
            res[a] = {"finished": False, "status": status}
            continue
        idle = status["gpu"]["idle_mem_mib"]
        conf = {}
        for s in meta["sections"]:
            key = (s["workload"], s["level"])
            e = sc.energy_j(gpu, s["start"], s["end"])
            w = sc.window(gpu, s["start"], s["end"])
            rr = [r for r in rows if r["workload"] == s["workload"] and r["level"] == s["level"] and r["rep"] == s["rep"] and not r["warmup"]]
            c = conf.setdefault(key, {"reps": [], "rows": []})
            c["rows"] += rr
            c["reps"].append({"agg_gen_tok_s": s["agg_gen_tok_s"], "per_h": s["requests_per_h"], "wall_s": s["wall_s"], "energy_j": e,
                              "generated": s["generated_tokens"], "requests": s["requests"], "errors": s["errors"],
                              "mean_w": float(np.mean([x[2] for x in w])) if w else None, "peak_w": max((x[2] for x in w), default=None),
                              "peak_mib": max((x[1] for x in w), default=None)})
        per = {}
        for (wl, lv), c in conf.items():
            reps = c["reps"]
            rr = [r for r in c["rows"] if not r["error"]]
            agg = [x["agg_gen_tok_s"] for x in reps]
            gr = [r["eval_count"] / r["eval_s"] for r in rr if r.get("eval_s") and r.get("eval_count") and r["eval_count"] > 1]
            pr = [r["prompt_eval_count"] / r["prompt_eval_s"] for r in rr if r.get("prompt_eval_s")]
            tt = [r["ttft_s"] for r in rr if r.get("ttft_s") is not None]
            d = {"reps": len(reps), "errors": sum(x["errors"] for x in reps),
                 "agg_gen_tok_s_mean": float(np.mean(agg)), "agg_gen_tok_s_range": [min(agg), max(agg)],
                 "per_request_gen_tok_s_median": float(np.median(gr)) if gr else None, "per_request_gen_ci95": ci_median(gr) if gr else None,
                 "prompt_tok_s_median": float(np.median(pr)) if pr else None,
                 "ttft_p50_s": float(np.percentile(tt, 50)) if tt else None, "ttft_p95_s": float(np.percentile(tt, 95)) if tt else None,
                 "mean_power_w": float(np.mean([x["mean_w"] for x in reps if x["mean_w"]])), "peak_power_w": max(x["peak_w"] or 0 for x in reps),
                 "peak_vram_gb": round((max(x["peak_mib"] or 0 for x in reps) - idle) / 1024, 2),
                 "peak_vram_total_gb": round(max(x["peak_mib"] or 0 for x in reps) / 1024, 2),
                 "gen_tokens_per_request_mean": float(np.mean([r["eval_count"] or 0 for r in rr])) if rr else None,
                 "prompt_tokens_median": float(np.median([r["prompt_eval_count"] or 0 for r in rr])) if rr else None}
            if wl == "decide":
                ph = [x["per_h"] for x in reps]
                d.update({"decisions_per_h_mean": float(np.mean(ph)), "decisions_per_h_range": [min(ph), max(ph)],
                          "j_per_1000_decisions": float(np.mean([x["energy_j"] / x["requests"] * 1000 for x in reps]))})
            else:
                d["j_per_1000_gen_tokens"] = float(np.mean([x["energy_j"] / x["generated"] * 1000 for x in reps if x["generated"]]))
            per[f"{wl}@{lv}"] = d
        lv_off = {lv["level"]: offloaded_ps(lv.get("ps")) for lv in meta.get("levels_detail", [])}
        for k, d in per.items():
            lvl = int(k.split("@")[1])
            d["offloaded"] = bool(lv_off.get(lvl)) or d["peak_vram_total_gb"] * 1024 >= CARD_MIB - 512
            base = per.get(f"{k.split('@')[0]}@1")
            d["scaling_vs_1"] = d["agg_gen_tok_s_mean"] / base["agg_gen_tok_s_mean"] if base and base["agg_gen_tok_s_mean"] and k.split("@")[0] != "decide" else (
                d["decisions_per_h_mean"] / base["decisions_per_h_mean"] if base and k.startswith("decide") else None)
        complete = all(f"{w}@{lv}" in per for w in meta["workloads"] for lv in meta["levels"])
        res[a] = {"finished": complete and (meta.get("gate") or {}).get("passed", False), "partial": True, "meta": meta, "status": status,
                  "per": per, "gate": meta.get("gate"), "stopped": meta.get("stopped")}
    done = [a for a in res if res[a].get("finished")]
    arms_out = []
    order = sorted(res, key=lambda a: list(C8_ARMS).index(a))
    for a in order:
        r = res[a]; info = C8_ARMS[a]
        d = {"arm": a, "model": info["model"], "engine": info.get("engine", "ollama"), "licence": info["licence"], "licence_class": info["licence_class"],
             "finished": r.get("finished", False), "gate": r.get("gate")}
        if r.get("partial"):
            d.update(model_cols(r["meta"]) if info.get("engine") != "vllm" else {"quant": r["meta"].get("quantisation"), "hf_model": r["meta"].get("hf_model"), "revision": r["meta"].get("revision")})
            d["configurations"] = {k: {kk: (r3(vv) if isinstance(vv, float) else ([r3(x) for x in vv] if isinstance(vv, list) else vv)) for kk, vv in v.items()} for k, v in r["per"].items()}
            if r.get("stopped"):
                d["stopped"] = r["stopped"]
        else:
            d["unfinished"] = {"reason": r["status"].get("reason")}
        arms_out.append(d)
    summary = {"test": "C8", "design": "plan/c8.md", "report": "results/c8/findings.md", "arms": arms_out}
    row = None
    if done:
        ollama_done = [a for a in done if not C8_ARMS[a].get("engine")]
        el = [a for a in ollama_done if "decide@16" in res[a]["per"] and not res[a]["per"]["decide@16"]["offloaded"]] or ollama_done
        top = max(el, key=lambda a: res[a]["per"].get("decide@16", {}).get("decisions_per_h_mean", 0))
        P = res[top]["per"]
        series = []
        for a in ollama_done + [x for x in done if x not in ollama_done]:
            p = res[a]["per"]
            series.append({"arm": a, "engine": C8_ARMS[a].get("engine", "ollama"), "params_b": model_cols(res[a]["meta"])["params_b"] if not C8_ARMS[a].get("engine") else None,
                           "chat_short_tok_s_at_1": r3(p["chat-short@1"]["agg_gen_tok_s_mean"]), "chat_short_tok_s_at_16": r3(p.get("chat-short@16", {}).get("agg_gen_tok_s_mean")),
                           "decisions_per_h_at_1": round(p["decide@1"]["decisions_per_h_mean"]), "decisions_per_h_at_16": round(p.get("decide@16", {}).get("decisions_per_h_mean", 0)),
                           "j_per_1000_tokens_at_16": round(p.get("chat-short@16", {}).get("j_per_1000_gen_tokens", 0)),
                           "peak_vram_gb": max(v["peak_vram_gb"] for v in p.values()), "offloaded_at_16": any(v["offloaded"] for k, v in p.items() if k.endswith("@16"))})
        row = {"id": "C8", "capability": "Throughput and energy per model size", "design": "plan/c8.md", "report": "results/c8/findings.md",
               "best_local_model": top, "best_by_value": top, "tie": None, "licence_class": C8_ARMS[top]["licence_class"],
               "quality": {"metric": "none (throughput test); headline: most one-token decisions per hour at 16 parallel clients among arms that stay on the card",
                           "value": None, "ci95": None},
               "speed": {"unit": "decisions per hour at 16 parallel clients (route1 one-token method)", "value": round(P["decide@16"]["decisions_per_h_mean"]),
                         "chat_short_tok_s_at_1": r3(P["chat-short@1"]["agg_gen_tok_s_mean"]), "chat_short_tok_s_at_16": r3(P["chat-short@16"]["agg_gen_tok_s_mean"])},
               "vram_peak_gb": max(v["peak_vram_gb"] for v in P.values()),
               "energy": {"unit": "GPU joules per 1,000 decisions at 16 parallel clients", "value": round(P["decide@16"]["j_per_1000_decisions"])},
               "fits_24gb": not P["decide@16"]["offloaded"], "smallest_within_interval": None, "series": series,
               "unfinished": [x["arm"] for x in arms_out if not x["finished"]]}
    return summary, c8_md(summary), row


def c8_md(s):
    L = ["# C8 score: throughput and energy per model size", "",
         "Generated by `bench/score-cap.py score` from `results/c8/<arm>/*-run1.jsonl`, the timed sections in each `meta.json` and the GPU logs. Design: `plan/c8.md`, generator `bench/gen-c8.py`.",
         "Aggregate rates are the mean of three repetitions with their range; per-request medians carry a 95 % bootstrap interval over requests (1,000 resamples, seed 20261009). Time to first token is measured by the client and includes waiting for a free slot above concurrency 1.", "",
         "## Sanity gate (chat-short at concurrency 1, three repetitions, CV at most 5 %)", "", "| arm | tokens/s per repetition | CV | passed |", "|---|---|---|---|"]
    for x in s["arms"]:
        g = x.get("gate")
        if g:
            L.append(f"| {x['arm']} | {', '.join(f'{v:.1f}' for v in g['values'])} | {100 * g['cv']:.2f} % | {'yes' if g['passed'] else 'no'} |")
    for wl, unit in (("chat-short", "aggregate generation tokens/s"), ("chat-long", "aggregate generation tokens/s"), ("decide", "decisions per hour")):
        L += ["", f"## {wl}: {unit} by concurrency (mean of 3, range)", "", "| arm | engine | quant | 1 | 4 | 8 | 16 | scaling 16 vs 1 | " + ("J per 1,000 decisions at 16" if wl == "decide" else "J per 1,000 tokens at 1 | J per 1,000 tokens at 16") + " | peak VRAM GB at 16 | offloaded at 16 |",
              "|---|---|---|---|---|---|---|---|" + ("---|" if wl == "decide" else "---|---|") + "---|---|"]
        for x in s["arms"]:
            c = x.get("configurations") or {}
            if f"{wl}@1" not in c:
                continue
            key = "decisions_per_h_mean" if wl == "decide" else "agg_gen_tok_s_mean"
            rng_k = "decisions_per_h_range" if wl == "decide" else "agg_gen_tok_s_range"
            cell = lambda lv: (f"{c[f'{wl}@{lv}'][key]:.0f} ({c[f'{wl}@{lv}'][rng_k][0]:.0f} to {c[f'{wl}@{lv}'][rng_k][1]:.0f})" if wl == "decide"  # noqa: E731
                               else f"{c[f'{wl}@{lv}'][key]:.1f} ({c[f'{wl}@{lv}'][rng_k][0]:.1f} to {c[f'{wl}@{lv}'][rng_k][1]:.1f})") if f"{wl}@{lv}" in c else ""
            c16 = c.get(f"{wl}@16", {})
            en = (f"{c16.get('j_per_1000_decisions', 0):.0f}" if wl == "decide" else f"{c[f'{wl}@1']['j_per_1000_gen_tokens']:.0f} | {c16.get('j_per_1000_gen_tokens', 0):.0f}")
            L.append(f"| {x['arm']} | {x['engine']} | {x.get('quant') or ''} | {cell(1)} | {cell(4)} | {cell(8)} | {cell(16)} | {f3(c16.get('scaling_vs_1'))} | {en} | {c16.get('peak_vram_gb', '')} | {'yes' if c16.get('offloaded') else 'no'} |")
    L += ["", "## Per request at concurrency 1 (medians)", "", "| arm | chat-short gen tok/s | 95 % CI | chat-long gen tok/s | chat-long prompt tok/s | TTFT p50 chat-short s | TTFT p50 chat-long s | TTFT p95 chat-long s at 16 | tokens per answer (chat-short) |", "|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        c = x.get("configurations") or {}
        if "chat-short@1" not in c:
            continue
        a1, l1, l16 = c["chat-short@1"], c.get("chat-long@1", {}), c.get("chat-long@16", {})
        L.append(f"| {x['arm']} | {f3(a1['per_request_gen_tok_s_median'])} | {cis(a1['per_request_gen_ci95'], 1) if a1.get('per_request_gen_ci95') else ''} | {f3(l1.get('per_request_gen_tok_s_median'))} | {f3(l1.get('prompt_tok_s_median'))} | "
                 f"{f3(a1.get('ttft_p50_s'))} | {f3(l1.get('ttft_p50_s'))} | {f3(l16.get('ttft_p95_s'))} | {f3(a1.get('gen_tokens_per_request_mean'))} |")
    L += ["", "Energy is GPU only (sum of 1 s power samples in each timed section). Peak VRAM is nvidia-smi memory.used minus the idle value before the job. vLLM rows run a different 4-bit build (AWQ) and a different engine; they compare engine plus quantisation, never engine alone (plan/c8.md).", ""]
    return "\n".join(L)
