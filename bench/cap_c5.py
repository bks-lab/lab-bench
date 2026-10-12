"""C5 (translation) for the capability map, called by bench/score-cap.py.

Translations of FLORES+ are derivatives of a gated ShareAlike text and are
not committed (plan/c5.md, note of 2026-10-12). So the rows are scored on the
Mac by bench/c5/score_c5.py and only the scores are committed:
results/c5/score.json (chrF++, BLEU, COMET, paired bootstrap per direction
and arm). This module imports the plumbing of the queue jobs (meta and
status without PC paths, no rows) and builds the summary, score.md and the
capability row from score.json plus the status files.

    python bench/score-cap.py import      (calls import_c5)
    python bench/score-cap.py score       (calls score_c5)

Value of the capability row: mean of the two directions' chrF++ (0 to 100),
interval: mean of the two directions' interval bounds (the directions share
the 1,012 sentences, so no narrower interval is claimed). An arm is separated
from the best when the paired bootstrap gives p < 0.05 in either direction.
"""
import glob
import json
import os
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIC = {"licence": "Apache-2.0", "licence_class": "open source"}
C5_ARMS = {
    "opus-mt": {"model": "Helsinki-NLP/opus-mt-de-en and opus-mt-en-de (MarianMT)", "licence": "Apache-2.0 (de-en), CC BY 4.0 (en-de)", "licence_class": "open source", "params_b": 0.075, "dedicated": True},
    "madlad-3b": {"model": "google/madlad400-3b-mt", **LIC, "params_b": 3.0, "dedicated": True},
    "qwen3.8-27b": {"model": "qwen3.8:27b (Qwen/Qwen3.8-27B)", **LIC, "params_b": 27.0},
    "gemma4-12b": {"model": "gemma4:12b (google/gemma-4-12B-it)", **LIC, "params_b": 12.0},
    "gemma4-26b": {"model": "gemma4:26b (google/gemma-4-26B-A4B-it)", **LIC, "params_b": 26.0},
    "mistral-small-3.2": {"model": "mistral-small3.2:24b (mistralai/Mistral-Small-3.2-24B-Instruct-2506)", **LIC, "params_b": 24.0},
    "eurollm-9b": {"model": "hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q8_0 (utter-project/EuroLLM-9B-Instruct)", **LIC, "params_b": 9.0},
}
DIRS = ("de-en", "en-de")
CARD_MIB = 24564


def _sc():
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("score_cap", ROOT / "bench" / "score-cap.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def import_c5(queue):
    """Meta and status of the C5 jobs, never the result rows."""
    sc = _sc()
    n = 0
    for d in sorted(glob.glob(os.path.join(queue, "*-c5-*"))):
        st_files = glob.glob(os.path.join(d, "*.status.json"))
        if not st_files:
            continue
        status = json.load(open(st_files[0], encoding="utf-8"))
        arm = status["arm"]
        out = ROOT / "results" / "c5" / ("gate" if arm == "gate" else arm)
        out.mkdir(parents=True, exist_ok=True)
        if os.path.exists(os.path.join(d, "meta.json")):
            sc.wr_json(out / "meta.json", sc.scrub(json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))))
        if os.path.exists(os.path.join(d, "gate.json")):
            sc.wr_json(out / "gate.json", sc.scrub(json.load(open(os.path.join(d, "gate.json"), encoding="utf-8"))))
        sc.wr_json(out / "queue-status.json", sc.scrub(status))
        n += 1
        print(f"imported {os.path.basename(d)} -> results/c5/{out.name}/")
    return n


def score_c5():
    sj = ROOT / "results" / "c5" / "score.json"
    if not sj.exists():
        return None
    sc = _sc()
    S = json.load(open(sj, encoding="utf-8"))
    D = S["directions"]
    arms = [a for a in C5_ARMS if (ROOT / "results" / "c5" / a / "queue-status.json").exists() and a in D["de-en"]["arms"]]
    status = {a: json.load(open(ROOT / "results" / "c5" / a / "queue-status.json", encoding="utf-8")) for a in arms}
    meta = {a: json.load(open(ROOT / "results" / "c5" / a / "meta.json", encoding="utf-8")) for a in arms}
    gate = json.load(open(ROOT / "results" / "c5" / "gate" / "gate.json", encoding="utf-8"))
    val = {a: statistics.mean(D[d]["arms"][a]["chrf++"] for d in DIRS) for a in arms}
    ci = {a: [statistics.mean(D[d]["arms"][a]["chrf++"] + s * D[d]["arms"][a]["chrf++_ci"] for d in DIRS) for s in (-1, 1)] for a in arms}
    vram = {a: status[a]["gpu"]["peak_over_idle_mib"] / 1024 for a in arms}
    best = max(arms, key=lambda a: val[a])
    sep = lambda a: a == best or any((D[d]["arms"][a]["chrf++_p_vs_best"] or 1.0) < 0.05 for d in DIRS)  # noqa: E731
    # the best arm of a direction can differ from the best on the mean; the paired p of S is against the
    # best of each direction, so with a different best the mean-best is checked on its own p as well
    order = sorted(arms, key=lambda a: -val[a])
    shown, tie = best, None
    nxt = order[1]
    if not sep(nxt):
        if vram[nxt] < vram[best]:
            shown = nxt
        tie = {"top_by_value": best, "next": nxt, "note": "the paired test does not separate the two in either direction; the one with less peak VRAM is shown"}
    small = [a for a in arms if not sep(a)]
    smallest = min(small, key=lambda a: vram[a])
    arms_out = []
    for a in order:
        info = C5_ARMS[a]
        g = status[a]["gpu"]
        d = {"arm": a, "model": info["model"], "licence": info["licence"], "licence_class": info["licence_class"], "params_b": info["params_b"],
             "dedicated_mt": bool(info.get("dedicated")), "finished": True,
             "quality": {"metric": "chrF++ on FLORES+ devtest, 1,012 sentences per direction", "value": round(val[a], 2), "ci95": [round(x, 2) for x in ci[a]],
                         "per_direction": {dd: {k: (round(D[dd]["arms"][a][k], 4) if D[dd]["arms"][a].get(k) is not None else None)
                                                for k in ("chrf++", "chrf++_ci", "chrf++_p_vs_best", "bleu", "comet", "errors")} for dd in DIRS}},
             "speed": {"unit": "segments per second at batch 1", "value": round(statistics.mean(D[dd]["arms"][a]["segments_per_s_batch1"] for dd in DIRS), 2),
                       **({"segments_per_s_batch32": round(statistics.mean(D[dd]["arms"][a]["segments_per_s_batch32"] for dd in DIRS), 2)} if info.get("dedicated") else {})},
             "energy": {"j_per_1000_segments": round(g["energy_j"] / 2024 * 1000), "note": "GPU only, whole job (model load, both directions, and for dedicated MT the batch 32 pass) divided by 2,024 segments"},
             "vram_peak_gb": round(vram[a], 2), "vram_peak_total_gb": round(g["peak_mem_mib"] / 1024, 2),
             "fits_24gb": True if g["peak_mem_mib"] < CARD_MIB - 512 else "full"}
        arms_out.append(d)
    by = {x["arm"]: x for x in arms_out}
    copy = {dd: D[dd]["arms"]["copy"] for dd in DIRS}
    base = statistics.mean(copy[dd]["chrf++"] for dd in DIRS)
    summary = {"test": "C5", "design": "plan/c5.md", "report": "results/c5/findings.md", "best_by_chrf": best,
               "gate": gate, "baseline": {"chrf++": round(base, 2), "rule": "source sentence copied as the translation"},
               "seed": S["seed"], "arms": arms_out}
    row = {"id": "C5", "capability": "translation German to English and English to German", "design": "plan/c5.md", "report": "results/c5/findings.md",
           "best_local_model": shown, "best_by_value": best, "tie": tie, "licence_class": C5_ARMS[shown]["licence_class"],
           "quality": {"metric": "chrF++ on FLORES+ devtest (mean of de-en and en-de, 0 to 100)", "value": by[shown]["quality"]["value"], "ci95": by[shown]["quality"]["ci95"],
                       "baseline": "source copied as the translation", "baseline_value": round(base, 2),
                       "reference": "OPUS-MT leaderboard opus-2020-02-26 on flores101-devtest, plain chrF 66.70 de-en and 63.63 en-de (sanity gate, tolerance 1.5)",
                       "reference_ours": round(statistics.mean(gate["directions"][dd]["chrf"] for dd in DIRS), 2)},
           "speed": {"unit": "segments per second at batch 1", "value": by[shown]["speed"]["value"]},
           "vram_peak_gb": by[shown]["vram_peak_gb"],
           "energy": {"unit": "GPU joules per 1,000 segments", "value": by[shown]["energy"]["j_per_1000_segments"]},
           "fits_24gb": by[shown]["fits_24gb"],
           "smallest_within_interval": {"arm": smallest, "value": by[smallest]["quality"]["value"], "vram_peak_gb": by[smallest]["vram_peak_gb"],
                                        "rule": "smallest peak VRAM among the arms the paired test does not separate from the best (p >= 0.05 in both directions)"},
           "unfinished": []}
    return summary, md(summary, D), row


def pf(v):
    return "best" if v is None else f"{v:.3f}"


def md(s, D):
    L = ["# C5 score: translation German to English and English to German", "",
         "Generated by `bench/score-cap.py score` from `results/c5/score.json` (`bench/c5/score_c5.py`, run on the Mac over the fetched rows; the rows are not committed because FLORES+ is gated and ShareAlike) and the queue status files. Design: `plan/c5.md`.", "",
         f"Sanity gate (OPUS-MT, plain chrF against the leaderboard, tolerance 1.5): de-en {s['gate']['directions']['de-en']['chrf']:.2f} against 66.70, en-de {s['gate']['directions']['en-de']['chrf']:.2f} against 63.63: {'passed' if s['gate']['passed'] else 'failed'}.",
         f"Copy baseline chrF++ (mean of both directions): {s['baseline']['chrf++']:.2f}.", "",
         "## Quality", "",
         "| arm | params (B) | chrF++ mean | de-en | en-de | BLEU de-en | BLEU en-de | COMET de-en | COMET en-de | p vs best de-en | p vs best en-de | failed calls |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in s["arms"]:
        p = x["quality"]["per_direction"]
        L.append(f"| {x['arm']} | {x['params_b']} | {x['quality']['value']:.2f} | {p['de-en']['chrf++']:.1f} | {p['en-de']['chrf++']:.1f} | {p['de-en']['bleu']:.1f} | {p['en-de']['bleu']:.1f} | "
                 f"{p['de-en']['comet']:.3f} | {p['en-de']['comet']:.3f} | {pf(p['de-en']['chrf++_p_vs_best'])} | {pf(p['en-de']['chrf++_p_vs_best'])} | {p['de-en']['errors'] + p['en-de']['errors']} |")
    L += ["", "p values: paired bootstrap against the best arm of that direction (1,000 resamples), from `results/c5/score.json`; best marks the best arm of that direction.", "",
          "## Machine", "", "| arm | segments/s batch 1 | segments/s batch 32 | J per 1,000 segments | peak VRAM GB (over idle) | peak VRAM GB (total) |", "|---|---|---|---|---|---|"]
    for x in s["arms"]:
        L.append(f"| {x['arm']} | {x['speed']['value']:.2f} | {x['speed'].get('segments_per_s_batch32', '')} | {x['energy']['j_per_1000_segments']} | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} |")
    L += ["", "Peak VRAM from nvidia-smi over the whole job; energy is GPU only and covers the whole job including model load.", ""]
    return "\n".join(L)
