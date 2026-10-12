"""C9 (personal data detection) for the capability map, called by bench/score-cap.py.

The result rows of C9 carry the predicted texts of invented documents, and the
documents themselves stay out of git (plan/c9.md, notes of 2026-10-12). So the
scores are made on the Mac by bench/c9/score_c9.py and only they are committed
(results/c9/score.json, score.md), together with the gold audit
(results/c9/gold-audit.json) and the meta and queue status of every arm. This
module imports meta and status (no rows) and builds the summary, the report
table and the capability row from score.json.

    python bench/score-cap.py import      (calls import_c9)
    python bench/score-cap.py score       (calls score_c9)

The gold audit was done by a language model, not a human (plan/c9.md, note of
2026-10-12): audited gold precision is below 0.90, so the test is "gold noisy"
and every number is descriptive. The capability row takes the numbers on the
audited gold of the 40 audited documents (the gold with the machine audit
applied); the original-gold numbers over all 476 documents sit beside them in
the summary. An arm is separated from the best when the 95 % paired bootstrap
interval of the F1 difference (on the 40 documents) excludes 0.
"""
import glob
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C9_ARMS = {
    "regex": {"model": "regular expressions (e-mail, IBAN with checksum, German phone patterns)", "licence": "own code", "licence_class": "open source", "params_b": 0.0, "cpu": True},
    "presidio": {"model": "Microsoft Presidio 2.2.364 with spaCy de_core_news_lg", "licence": "MIT", "licence_class": "open source", "params_b": 0.0, "cpu": True},
    "gliner-multi-pii": {"model": "urchade/gliner_multi_pii-v1", "licence": "Apache-2.0", "licence_class": "open source", "params_b": 0.5},
    "gliner-pii-domains": {"model": "E3-JSI/gliner-multi-pii-domains-v1", "licence": "Apache-2.0", "licence_class": "open source", "params_b": 0.5},
    "gliner2.5-multi": {"model": "fastino/gliner2.5-multi-v1", "licence": "Apache-2.0", "licence_class": "open source", "params_b": 0.3},
    "gemma4-12b": {"model": "gemma4:12b (google/gemma-4-12B-it)", "licence": "Apache-2.0", "licence_class": "open source", "params_b": 12.0},
    "qwen3.8-27b": {"model": "qwen3.8:27b (Qwen/Qwen3.8-27B)", "licence": "Apache-2.0", "licence_class": "open source", "params_b": 27.0},
}
CARD_MIB = 24564
CAVEAT = ("Gold noisy: the 40-document gold audit was done by a language model (Claude Haiku 5.5, spot check of all non-ok verdicts by Claude Opus 5.5), "
          "not by a human as plan/c9.md requires; audited gold precision 0.685 is below 0.90, so all numbers are descriptive and make no ranking claim.")


def _sc():
    spec = importlib.util.spec_from_file_location("score_cap", ROOT / "bench" / "score-cap.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def import_c9(queue):
    """Meta and status of the C9 jobs, never the result rows."""
    sc = _sc()
    n = 0
    for d in sorted(glob.glob(os.path.join(queue, "*-c9-*"))):
        st_files = glob.glob(os.path.join(d, "*.status.json"))
        if not st_files:
            continue
        status = json.load(open(st_files[0], encoding="utf-8"))
        out = ROOT / "results" / "c9" / status["arm"]
        out.mkdir(parents=True, exist_ok=True)
        if os.path.exists(os.path.join(d, "meta.json")):
            sc.wr_json(out / "meta.json", sc.scrub(json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))))
        sc.wr_json(out / "queue-status.json", sc.scrub(status))
        n += 1
        print(f"imported {os.path.basename(d)} -> results/c9/{status['arm']}/")
    return n


def score_c9():
    sj = ROOT / "results" / "c9" / "score.json"
    if not sj.exists():
        return None
    S = json.load(open(sj, encoding="utf-8"))
    arms = [a for a in C9_ARMS if a in S["arms"] and (ROOT / "results" / "c9" / a / "queue-status.json").exists()]
    status = {a: json.load(open(ROOT / "results" / "c9" / a / "queue-status.json", encoding="utf-8")) for a in arms}
    A, O = S["audited"], S["arms"]
    best = max(arms, key=lambda a: A[a]["f1"])
    vram = {a: status[a]["gpu"]["peak_over_idle_mib"] / 1024 for a in arms}
    # separated: p < 0.05 of the paired bootstrap on the 40 audited documents
    sep = lambda a: a != best and (1.0 if A[a]["paired_vs_best"]["p"] is None else A[a]["paired_vs_best"]["p"]) < 0.05  # noqa: E731
    unsep = [a for a in arms if not sep(a)]
    order = sorted(arms, key=lambda a: -A[a]["f1"])
    nxt = order[1]
    shown, tie = best, None
    if not sep(nxt):
        if vram[nxt] < vram[best]:
            shown = nxt
        tie = {"top_by_value": best, "next": nxt, "note": "the paired bootstrap on the 40 audited documents does not separate the two; the one with less peak VRAM is shown. On all 476 documents (original gold) the order is the other way round and equally not separated"}
    smallest = min(unsep, key=lambda a: vram[a])
    arms_out = []
    for a in order:
        info, g, m = C9_ARMS[a], status[a]["gpu"], O[a]["machine"]
        arms_out.append({
            "arm": a, "baseline": a == "regex", "model": info["model"], "licence": info["licence"], "licence_class": info["licence_class"], "params_b": info["params_b"],
            "runs_on": "CPU" if info.get("cpu") else "GPU", "finished": True,
            "quality": {"metric": "span F1 (exact boundaries, type-agnostic) on the audited gold of 40 documents", "value": round(A[a]["f1"], 6),
                        "ci95": [round(x, 6) for x in A[a]["ci"]["f1"]], "recall": round(A[a]["recall"], 6), "precision": round(A[a]["precision"], 6),
                        "vs_best": ({"best": best, "diff": round(A[a]["paired_vs_best"]["diff"], 6), "ci95": [round(x, 6) for x in A[a]["paired_vs_best"]["ci"]],
                                    "p": round(A[a]["paired_vs_best"]["p"], 6), "difference_shown": sep(a)} if a != best else None),
                        "original_gold_476_docs": {"f1": round(O[a]["f1"], 6), "ci95": [round(x, 6) for x in O[a]["ci"]["f1"]], "recall": round(O[a]["recall"], 6),
                                                   "precision": round(O[a]["precision"], 6), "mask_recall": round(O[a]["mask_recall"], 6),
                                                   "p_vs_best": round(O[a]["paired_vs_best"]["p"], 6) if O[a]["paired_vs_best"]["p"] is not None else None}},
            "speed": {"unit": "median ms per document (476 documents)", "value": round(m["median_ms_per_doc"], 1)},
            "energy": {"j_per_1000_docs": round(m["energy_j_per_1000_docs"]) if m.get("energy_j_per_1000_docs") is not None else None,
                       "note": "GPU only, timed section" + ("; the job used the CPU, the figure is the idle GPU" if info.get("cpu") else "")},
            "vram_peak_gb": round(vram[a], 2), "vram_peak_total_gb": round(g["peak_mem_mib"] / 1024, 2),
            "fits_24gb": True if g["peak_mem_mib"] < CARD_MIB - 512 else "full"})
    by = {x["arm"]: x for x in arms_out}
    gate = O["regex"].get("gate")
    summary = {"test": "C9", "design": "plan/c9.md", "report": "results/c9/findings.md", "best_by_audited_f1": best, "gold_noisy": S["audit"]["gold_noisy"],
               "caveat": CAVEAT, "documents": S["documents"], "audited_documents": S["audit"]["documents"], "best_by_original_f1": max(arms, key=lambda a: O[a]["f1"]), "gold_audit": {"form": "machine audit (claude-haiku-5-5), spot check of non-ok verdicts by claude-opus-5-5", "file": "results/c9/gold-audit.json", **S["audit"]},
               "gate": gate, "bootstrap": S["bootstrap"], "arms": arms_out}
    rx = by["regex"]
    row = {"id": "C9", "capability": "personal data detection in German text", "design": "plan/c9.md", "report": "results/c9/findings.md",
           "best_local_model": shown, "best_by_value": best, "tie": tie, "licence_class": C9_ARMS[shown]["licence_class"], "gold_noisy": True,
           "quality": {"metric": "span F1 on the audited gold of 40 documents (exact boundaries, type-agnostic); gold noisy, descriptive only", "value": by[shown]["quality"]["value"],
                       "ci95": by[shown]["quality"]["ci95"], "recall": by[shown]["quality"]["recall"], "baseline": "regex (e-mail, IBAN, phone)", "baseline_value": rx["quality"]["value"],
                       "reference": "none published for this German subset; internal sanity gate regex e-mail recall >= 0.95", "reference_ours": round(gate["value"], 6) if gate else None,
                       "original_gold_476_docs": by[shown]["quality"]["original_gold_476_docs"], "caveat": CAVEAT},
           "speed": {"unit": "median ms per document", "value": by[shown]["speed"]["value"]},
           "vram_peak_gb": by[shown]["vram_peak_gb"],
           "energy": {"unit": "GPU joules per 1,000 documents", "value": by[shown]["energy"]["j_per_1000_docs"]},
           "fits_24gb": by[shown]["fits_24gb"],
           "smallest_within_interval": {"arm": smallest, "value": by[smallest]["quality"]["value"], "vram_peak_gb": by[smallest]["vram_peak_gb"],
                                        "rule": "smallest peak VRAM among the arms the paired bootstrap on the 40 audited documents does not separate from the best"},
           "unfinished": []}
    return summary, md(summary, S), row


def md(s, S):
    types = list(next(iter(S["arms"].values()))["typed_f1"])
    L = ["# C9 score: personal data detection in German text", "",
         "Generated by `bench/score-cap.py score` from `results/c9/score.json` (`bench/c9/score_c9.py`, run on the Mac over the queue outputs; the documents and the predicted texts are not committed) and the queue status files. Design: `plan/c9.md`.", "",
         f"**{CAVEAT}**", "",
         f"Gold audit: {s['gold_audit']['documents']} documents, {s['gold_audit']['gold_spans']} gold spans, {s['gold_audit']['kept']} kept as they were, "
         f"{s['gold_audit']['missed_added']} missed spans added; gold precision {s['gold_audit']['gold_precision']:.3f}, gold recall {s['gold_audit']['gold_recall']:.3f}.",
         f"Sanity gate (regex recall on gold e-mail spans, at least {s['gate']['minimum']}): {s['gate']['value']:.3f}, {'passed' if s['gate']['passed'] else 'failed'}.", "",
         "## Quality (audited gold, 40 documents)", "",
         "| arm | runs on | span F1 | 95 % CI | recall | precision | p | difference shown | diff to best |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        q = x["quality"]
        vb = q["vs_best"]
        L.append(f"| {x['arm']} | {x['runs_on']} | {q['value']:.3f} | [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}] | {q['recall']:.3f} | {q['precision']:.3f} | "
                 + (f"{vb['p']:.3f} | {'yes' if vb['difference_shown'] else 'no'} | {vb['diff']:.3f} |" if vb else "best | best | best |"))
    L += ["", "## Original gold (all 476 documents)", "",
          "| arm | span F1 | 95 % CI | recall | precision | mask recall | p |", "|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        o = x["quality"]["original_gold_476_docs"]
        L.append(f"| {x['arm']} | {o['f1']:.3f} | [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}] | {o['recall']:.3f} | {o['precision']:.3f} | {o['mask_recall']:.3f} | "
                 + (f"{o['p_vs_best']:.3f} |" if o["p_vs_best"] is not None else "best |"))
    L += ["", "Intervals: 95 % bootstrap over documents (1,000 resamples, seed 20261009). Mask recall: share of gold characters covered by any predicted span. p: paired bootstrap against the best arm of that table (on the same documents).", "",
          "Typed strict F1 per type (original gold, 476 documents, descriptive):", "",
          "| arm | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for x in s["arms"]:
        L.append(f"| {x['arm']} | " + " | ".join(f"{S['arms'][x['arm']]['typed_f1'][t]:.2f}" for t in types) + " |")
    L += ["", "## Machine", "", "| arm | median ms per document | GPU J per 1,000 documents | peak VRAM GB (over idle) | peak VRAM GB (total) |", "|---|---|---|---|---|"]
    for x in s["arms"]:
        L.append(f"| {x['arm']} | {x['speed']['value']:.1f} | {x['energy']['j_per_1000_docs'] if x['energy']['j_per_1000_docs'] is not None else ''} | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} |")
    L += ["", "Peak VRAM from nvidia-smi over the whole job; energy is GPU only (timed section). Presidio and regex run on the CPU, so their GPU figures are the idle card.", ""]
    return "\n".join(L)
