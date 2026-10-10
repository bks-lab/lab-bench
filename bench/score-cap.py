#!/usr/bin/env python3
"""Scores the capability tests C1 (retrieval), C2 (speech to text) and C3
(invoice fields) exactly as preregistered in plan/c0-common.md, plan/c1.md,
plan/c2.md and plan/c3.md, and through bench/cap_c4c8.py the tests C4
(structured output), C6 (reading comprehension), C7 (long context) and C8
(throughput) as preregistered in their plans. C4 needs the per-item verdicts
of bench/c4/check_ast.py next to its rows.

    uv run --with numpy python bench/score-cap.py import --queue .cache/queue
    uv run --with numpy python bench/score-cap.py score

`import` copies the queue outputs of every C1 to C3 job (bench/queue/fetch.sh
puts them in .cache/queue/<job id>/) into results/<cN>/: the result rows as
results/<cN>/<arm dir>/<date>-run1.jsonl, the runner meta and the queue
status (PC paths and the host name removed) next to them, and the 1 s GPU log
as results/<cN>/gpu/<arm dir>.csv. A job that did not finish keeps its status
and GPU log, and the per-invoice lines of its job log become
<date>-run1-unfinished.jsonl (no predictions; the runner writes those only at
the end). Arm dir = arm name with "+" replaced by "-" and a leading "+"
dropped.

`score` reads only results/ and the committed gold, so it runs without the
PC. It recomputes every metric from the rows (it does not copy the runner's
summary), adds the 95 % bootstrap intervals (1,000 resamples, numpy
default_rng(20261009), unit as each plan names it), paired bootstrap tests
against the best arm, GPU energy and peak VRAM from the GPU log, and writes
results/<cN>/score.md, results/<cN>/summary.json and results/capabilities.json.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import statistics
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "c3"))
sys.path.insert(0, str(ROOT / "bench"))
from fields import FIELDS, MONEY, score_row  # noqa: E402

SEED = 20261009
RESAMPLES = 1000
CARD_GB = 24564 / 1024  # memory.total of the RTX 4090 in the machine record
FITS_RULE = ("Ollama arms: 'offloaded' when nvidia-smi memory.used reached the card (peak within 0.5 GB of 24,564 MiB): "
             "Ollama fills the card only when it splits a model between GPU and CPU; the GPU power and per-call time "
             "of those arms confirm it (results/c3/findings.md). Torch and CTranslate2 arms cannot offload, so a "
             "finished run fits.")

# ---------------------------------------------------------------- arms

# params: counted from the safetensors headers of the pinned revision on the PC
# (2026-10-10), or the Ollama parameter_size, or the model card where neither exists.
C1_ARMS = {
    "bge-m3": {"model": "BAAI/bge-m3", "params_m": 568, "params_source": "model card (weights are pytorch_model.bin, not counted)", "licence": "MIT", "licence_class": "open source"},
    "e5-large": {"model": "intfloat/multilingual-e5-large", "params_m": 560, "params_source": "safetensors header", "licence": "MIT", "licence_class": "open source"},
    "e5-large-instruct": {"model": "intfloat/multilingual-e5-large-instruct", "params_m": 560, "params_source": "safetensors header", "licence": "MIT", "licence_class": "open source"},
    "gte-multi": {"model": "Alibaba-NLP/gte-multilingual-base", "params_m": 305, "params_source": "safetensors header", "licence": "Apache-2.0", "licence_class": "open source"},
    "nomic-v2-moe": {"model": "nomic-ai/nomic-embed-text-v2-moe", "params_m": 475, "params_source": "safetensors header (all experts; the card gives 305M active)", "licence": "Apache-2.0", "licence_class": "open source"},
    "qwen3-emb-0.6b": {"model": "Qwen/Qwen3-Embedding-0.6B", "params_m": 596, "params_source": "safetensors header", "licence": "Apache-2.0", "licence_class": "open source"},
    "qwen3-emb-8b": {"model": "Qwen/Qwen3-Embedding-8B", "params_m": 7567, "params_source": "safetensors header", "licence": "Apache-2.0", "licence_class": "open source"},
    "bm25": {"model": "BM25 (bm25s, German Snowball stemmer, k1 0.9, b 0.4)", "params_m": 0, "params_source": "no model", "licence": "MIT", "licence_class": "open source", "baseline": True},
    # rerankers: rescore the top 100 of the best embedder (plan/c1.md), so each is a pipeline
    "+bge-rerank": {"model": "qwen3-emb-8b top 100, reranked by BAAI/bge-reranker-v2-m3", "params_m": 568, "params_source": "safetensors header of the reranker (the base embedder adds 7,567)", "licence": "Apache-2.0", "licence_class": "open source", "reranker": True, "base": "qwen3-emb-8b"},
    "+qwen3-rerank": {"model": "qwen3-emb-8b top 100, reranked by Qwen/Qwen3-Reranker-0.6B", "params_m": 596, "params_source": "safetensors header of the reranker (the base embedder adds 7,567)", "licence": "Apache-2.0", "licence_class": "open source", "reranker": True, "base": "qwen3-emb-8b"},
}
C2_ARMS = {
    "canary-1b-v2": {"model": "nvidia/canary-1b-v2", "params_m": 979, "params_source": "safetensors header", "licence": "CC BY 4.0", "licence_class": "open source (CC BY)"},
    "parakeet-v3": {"model": "nvidia/parakeet-tdt-0.6b-v3", "params_m": 627, "params_source": "safetensors header", "licence": "CC BY 4.0", "licence_class": "open source (CC BY)"},
    "whisper-large-v3": {"model": "openai/whisper-large-v3 (Systran/faster-whisper-large-v3)", "params_m": 1550, "params_source": "model card", "licence": "Apache-2.0 weights, MIT conversion", "licence_class": "open source"},
    "whisper-large-v3-turbo": {"model": "openai/whisper-large-v3-turbo (deepdml/faster-whisper-large-v3-turbo-ct2)", "params_m": 809, "params_source": "model card", "licence": "MIT", "licence_class": "open source"},
    "whisper-turbo-de": {"model": "primeline/whisper-large-v3-turbo-german", "params_m": 809, "params_source": "safetensors header", "licence": "Apache-2.0", "licence_class": "open source"},
    "distil-whisper-de": {"model": "primeline/distil-whisper-large-v3-german", "params_m": 756, "params_source": "safetensors header", "licence": "Apache-2.0", "licence_class": "open source"},
}
C3_ARMS = {
    "qwen3-vl-8b": {"model": "qwen3-vl:8b (Qwen/Qwen3-VL-8B-Instruct)", "kind": "vision", "params_b": 8, "params_source": "Ollama tag (nominal)", "licence": "Apache-2.0", "licence_class": "open source"},
    "qwen3-vl-32b": {"model": "qwen3-vl:32b (Qwen/Qwen3-VL-32B-Instruct)", "kind": "vision", "params_b": 32, "params_source": "Ollama tag (nominal)", "licence": "Apache-2.0", "licence_class": "open source"},
    "gemma4-12b": {"model": "gemma4:12b (google/gemma-4-12B-it)", "kind": "vision", "licence": "Apache-2.0", "licence_class": "open source"},
    "gemma4-26b": {"model": "gemma4:26b (google/gemma-4-26B-A4B-it)", "kind": "vision", "licence": "Apache-2.0", "licence_class": "open source"},
    "mistral-small-3.2": {"model": "mistral-small3.2:24b (mistralai/Mistral-Small-3.2-24B-Instruct-2506)", "kind": "vision", "licence": "Apache-2.0", "licence_class": "open source"},
    "docling+qwen3.8-27b": {"model": "docling EasyOCR + qwen3.8:27b", "kind": "ocr+text", "licence": "MIT (docling), Apache-2.0 (EasyOCR, model)", "licence_class": "open source"},
    "docling+gemma4-12b": {"model": "docling EasyOCR + gemma4:12b", "kind": "ocr+text", "licence": "MIT (docling), Apache-2.0 (EasyOCR, model)", "licence_class": "open source"},
    "pdftext+qwen3.8-27b": {"model": "pdftotext text layer + qwen3.8:27b (reference arm, not a scan)", "kind": "text layer", "licence": "Apache-2.0 (model)", "licence_class": "open source", "reference_arm": True},
}
ARMS = {"c1": C1_ARMS, "c2": C2_ARMS, "c3": C3_ARMS}


def arm_dir(arm):
    return arm.lstrip("+").replace("+", "-")


# ---------------------------------------------------------------- helpers

def rd_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wr_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def rng():
    return np.random.default_rng(SEED)


def ci_mean(values, clusters=None):
    """95 % percentile interval of the mean; resamples items or clusters."""
    v = np.asarray(values, dtype=float)
    r = rng()
    if clusters is None:
        idx = r.integers(0, len(v), size=(RESAMPLES, len(v)))
        st = v[idx].mean(axis=1)
    else:
        keys = sorted(set(clusters)); pos = {k: i for i, k in enumerate(keys)}
        s = np.zeros(len(keys)); c = np.zeros(len(keys))
        for x, k in zip(v, clusters):
            s[pos[k]] += x; c[pos[k]] += 1
        idx = r.integers(0, len(keys), size=(RESAMPLES, len(keys)))
        st = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def ci_ratio(num, den, clusters):
    """95 % interval of sum(num) / sum(den), resampling clusters (corpus WER)."""
    keys = sorted(set(clusters)); pos = {k: i for i, k in enumerate(keys)}
    n = np.zeros(len(keys)); d = np.zeros(len(keys))
    for a, b, k in zip(num, den, clusters):
        n[pos[k]] += a; d[pos[k]] += b
    idx = rng().integers(0, len(keys), size=(RESAMPLES, len(keys)))
    st = n[idx].sum(axis=1) / d[idx].sum(axis=1)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def ci_median(values):
    v = np.asarray(values, dtype=float)
    idx = rng().integers(0, len(v), size=(RESAMPLES, len(v)))
    st = np.median(v[idx], axis=1)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def paired_mean(a, b, clusters=None):
    """Paired bootstrap of mean(a) - mean(b) over the same items (or clusters).
    Returns diff, 95 % interval, two-sided p (twice the smaller tail share at 0)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    d = a - b
    r = rng()
    if clusters is None:
        idx = r.integers(0, len(d), size=(RESAMPLES, len(d)))
        st = d[idx].mean(axis=1)
    else:
        keys = sorted(set(clusters)); pos = {k: i for i, k in enumerate(keys)}
        s = np.zeros(len(keys)); c = np.zeros(len(keys))
        for x, k in zip(d, clusters):
            s[pos[k]] += x; c[pos[k]] += 1
        idx = r.integers(0, len(keys), size=(RESAMPLES, len(keys)))
        st = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    p = min(1.0, 2 * min((st <= 0).mean(), (st >= 0).mean()))
    return float(d.mean()), [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], float(p)


def paired_ratio(na, nb, den, clusters):
    """Paired cluster bootstrap of sum(na)/sum(den) - sum(nb)/sum(den) (WER difference)."""
    keys = sorted(set(clusters)); pos = {k: i for i, k in enumerate(keys)}
    A = np.zeros(len(keys)); B = np.zeros(len(keys)); D = np.zeros(len(keys))
    for x, y, z, k in zip(na, nb, den, clusters):
        A[pos[k]] += x; B[pos[k]] += y; D[pos[k]] += z
    idx = rng().integers(0, len(keys), size=(RESAMPLES, len(keys)))
    st = (A[idx].sum(axis=1) - B[idx].sum(axis=1)) / D[idx].sum(axis=1)
    diff = (A.sum() - B.sum()) / D.sum()
    p = min(1.0, 2 * min((st <= 0).mean(), (st >= 0).mean()))
    return float(diff), [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], float(p)


def overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def r3(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------- GPU log

def gpu_samples(csv_path):
    """[(naive local datetime, memory MiB, power W)] from the 1 s nvidia-smi log."""
    out = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        if len(r) < 3:
            continue
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            out.append((t, float(r[1].split()[0]), float(r[2].split()[0])))
        except ValueError:
            continue
    return out


def local_naive(iso):
    return dt.datetime.fromisoformat(iso).replace(tzinfo=None)


def window(samples, start, end):
    a, b = local_naive(start), local_naive(end)
    return [s for s in samples if a <= s[0] <= b]


def energy_j(samples, start, end):
    """Sum of the 1 s power samples in the section (joules, GPU only)."""
    return float(sum(s[2] for s in window(samples, start, end)))


def peak_mib(samples, start=None, end=None):
    w = window(samples, start, end) if start else samples
    return max((s[1] for s in w), default=None)


# ---------------------------------------------------------------- import

PATH_RE = re.compile(r"[A-Za-z]:\\\\?[^\"']*")


def scrub(o):
    """Drop PC paths, the host name and job plumbing from meta and status."""
    drop = {"host", "out", "log", "job", "argv", "path", "base_run", "JMB_OUT", "JMB_GPU_CSV"}
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if isinstance(o, str) and re.match(r"^[A-Za-z]:\\", o):
        return "<PC path>"
    if isinstance(o, str):
        return re.sub(r"[A-Za-z]:\\[^\s,;\"']*", "<PC path>", o)
    return o


def do_import(queue):
    n = 0
    arms_all = dict(ARMS)
    try:
        import cap_c4c8  # noqa: PLC0415
        arms_all.update(cap_c4c8.ARMS)
    except ImportError:
        pass
    for d in sorted(glob.glob(os.path.join(queue, "*-c[1-9]-*"))):
        jid = os.path.basename(d)
        st_files = glob.glob(os.path.join(d, "*.status.json"))
        if not st_files:
            continue
        status = json.load(open(st_files[0], encoding="utf-8"))
        test, arm = status["test"].lower(), status["arm"]
        if arm not in arms_all.get(test, {}):
            print(f"skip {jid}: arm {arm} not in the plan", file=sys.stderr)
            continue
        ad = arm_dir(arm)
        out = ROOT / "results" / test / ad
        out.mkdir(parents=True, exist_ok=True)
        meta_p = os.path.join(d, "meta.json")
        date = status["start"][:10]
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p, encoding="utf-8"))
            date = meta["started"][:10]
            wr_json(out / "meta.json", scrub(meta))
            shutil.copyfile(os.path.join(d, "result.jsonl"), out / f"{date}-run1.jsonl")
            if os.path.exists(os.path.join(d, "ocr.jsonl")):
                shutil.copyfile(os.path.join(d, "ocr.jsonl"), out / f"{date}-run1-ocr.jsonl")
        elif test != "c3":
            # no meta: a job that stopped early; keep its rows (the C4 to C8 runners write them as they go)
            if os.path.exists(os.path.join(d, "result.jsonl")):
                shutil.copyfile(os.path.join(d, "result.jsonl"), out / f"{date}-run1-unfinished.jsonl")
        else:
            # unfinished: per-invoice lines of the job log (id, correct cells, seconds, error)
            rows = []
            for line in open(glob.glob(os.path.join(d, "*.log"))[0], encoding="utf-8", errors="replace"):
                m = re.match(r"^(\d\d\.\d\d[a-z]) (\d+)/12 (\S+) ?(.*)$", line.rstrip("\n"))
                if m:
                    rows.append({"id": m[1], "arm": arm, "n_correct": int(m[2]),
                                 "seconds_model": None if m[3] == "None" else float(m[3]),
                                 "error": m[4].strip() or None, "source": "job log line, no prediction"})
            with open(out / f"{date}-run1-unfinished.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        wr_json(out / "queue-status.json", scrub(status))
        (ROOT / "results" / test / "gpu").mkdir(exist_ok=True)
        shutil.copyfile(os.path.join(d, "gpu.csv"), ROOT / "results" / test / "gpu" / f"{ad}.csv")
        n += 1
        print(f"imported {jid} -> results/{test}/{ad}/")
    print(f"{n} jobs imported")


def latest_rows(test, arm, tag=""):
    files = sorted(glob.glob(str(ROOT / "results" / test / arm_dir(arm) / f"*-run1{tag}.jsonl")))
    return rd_jsonl(files[-1]) if files else None


def load_arm(test, arm):
    p = ROOT / "results" / test / arm_dir(arm)
    meta = json.load(open(p / "meta.json", encoding="utf-8")) if (p / "meta.json").exists() else None
    status = json.load(open(p / "queue-status.json", encoding="utf-8"))
    gpu = gpu_samples(ROOT / "results" / test / "gpu" / f"{arm_dir(arm)}.csv")
    return meta, status, gpu


def machine_cols(status, gpu, meta_sections=None):
    g = status["gpu"]
    peak_over = g["peak_over_idle_mib"]
    return {
        "vram_peak_gb": round(peak_over / 1024, 2),
        "vram_peak_total_gb": round(g["peak_mem_mib"] / 1024, 2),
        "vram_idle_gb": round(g["idle_mem_mib"] / 1024, 2),
        "vram_note": "nvidia-smi memory.used, peak of the whole job minus the idle value before the model loaded (queue status)",
        "gpu_minutes_job": g["gpu_min"],
        "energy_job_j": g["energy_j"],
    }


# ---------------------------------------------------------------- C1

def score_c1():
    res = {}
    for arm, info in C1_ARMS.items():
        meta, status, gpu = load_arm("c1", arm)
        rows = latest_rows("c1", arm)
        per = {}
        for s in ("miracl", "germandpr"):
            rs = [r for r in rows if r["set"] == s]
            nd = [r["ndcg10"] for r in rs]; rc = [r["recall100"] for r in rs]; mr = [r["mrr10"] for r in rs]
            lat = [r["latency_ms"] for r in rs]
            sec = {x["set"]: x for x in meta["sections"] if x["name"] == "index"}.get(s)
            qsec = {x["set"]: x for x in meta["sections"] if x["name"] == "queries_batch1"}.get(s)
            e_idx = energy_j(gpu, sec["start"], sec["end"]) if sec else None
            pk = peak_mib(gpu, sec["start"], sec["end"]) if sec else None
            per[s] = {
                "queries": len(rs), "qids": [r["qid"] for r in rs],
                "ndcg10_items": nd,
                "ndcg10": float(np.mean(nd)), "ndcg10_ci95": ci_mean(nd),
                "recall100": float(np.mean(rc)), "recall100_ci95": ci_mean(rc),
                "mrr10": float(np.mean(mr)), "mrr10_ci95": ci_mean(mr),
                "query_ms_median": float(np.median(lat)), "query_ms_median_ci95": ci_median(lat),
                "index_seconds": sec["seconds"] if sec else None,
                "passages_per_s": (sec["passages"] / sec["seconds"]) if sec else None,
                "index_energy_j": e_idx,
                "index_energy_j_per_1000_passages": (e_idx / sec["passages"] * 1000) if sec else None,
                "index_peak_vram_gb": round((pk - status["gpu"]["idle_mem_mib"]) / 1024, 2) if pk else None,
                "truncated_passages": meta["summary"][s].get("truncated_passages"),
            }
        res[arm] = {"info": info, "meta": meta, "status": status, "per": per}
    # A reranker arm is a pipeline: the base embedder indexes and retrieves the
    # top 100, the reranker rescores it. Index numbers come from the base run,
    # the query time is base plus rerank per query, and peak VRAM is the larger
    # of the two jobs (they run one after the other, never together).
    for arm, info in C1_ARMS.items():
        if not info.get("reranker"):
            continue
        r, b = res[arm], res[info["base"]]
        rows = latest_rows("c1", arm); brows = latest_rows("c1", info["base"])
        blat = {(x["set"], x["qid"]): x["latency_ms"] for x in brows}
        for s in ("miracl", "germandpr"):
            p, bp = r["per"][s], b["per"][s]
            rr = [x["latency_ms"] for x in rows if x["set"] == s]
            tot = [x["latency_ms"] + blat[(s, x["qid"])] for x in rows if x["set"] == s]
            p["rerank_ms_median"] = float(np.median(rr))
            p["query_ms_median"] = float(np.median(tot)); p["query_ms_median_ci95"] = ci_median(tot)
            for k in ("index_seconds", "passages_per_s", "index_energy_j", "index_energy_j_per_1000_passages", "index_peak_vram_gb", "truncated_passages"):
                p[k] = bp[k]
        st = json.loads(json.dumps(r["status"]))
        st["gpu"]["peak_over_idle_mib"] = max(r["status"]["gpu"]["peak_over_idle_mib"], b["status"]["gpu"]["peak_over_idle_mib"])
        st["gpu"]["peak_mem_mib"] = max(r["status"]["gpu"]["peak_mem_mib"], b["status"]["gpu"]["peak_mem_mib"])
        r["status_own"], r["status"] = r["status"], st
    bm = res["bm25"]["per"]["miracl"]["ndcg10"]
    models = [a for a in C1_ARMS if not C1_ARMS[a].get("baseline")]
    best = max(models, key=lambda a: res[a]["per"]["miracl"]["ndcg10"])
    order = sorted(models, key=lambda a: -res[a]["per"]["miracl"]["ndcg10"])
    for a in models + ["bm25"]:
        m = res[a]["per"]["miracl"]
        res[a]["skill"] = (m["ndcg10"] - bm) / (1 - bm)
        if a != best:
            assert res[a]["per"]["miracl"]["qids"] == res[best]["per"]["miracl"]["qids"]
            diff, ci, p = paired_mean(res[best]["per"]["miracl"]["ndcg10_items"], m["ndcg10_items"])
            res[a]["vs_best"] = {"best": best, "diff": diff, "diff_ci95": ci, "p": p,
                                 "intervals_overlap": overlap(m["ndcg10_ci95"], res[best]["per"]["miracl"]["ndcg10_ci95"]),
                                 "difference_shown": p < 0.05 or not overlap(m["ndcg10_ci95"], res[best]["per"]["miracl"]["ndcg10_ci95"])}
    gates = {a: res[a]["meta"].get("gate") for a in ("bge-m3", "e5-large")}
    return res, best, order, gates


def c1_outputs(res, best, order, gates):
    arms_out = []
    for a in order + ["bm25"]:
        r = res[a]; m = r["per"]["miracl"]; g = r["per"]["germandpr"]
        mc = machine_cols(r["status"], None)
        arms_out.append({
            "arm": a, "model": r["info"]["model"], "revision": r["meta"]["arm_config"].get("revision"),
            "params_m": r["info"]["params_m"], "params_source": r["info"]["params_source"],
            "licence": r["info"]["licence"], "licence_class": r["info"]["licence_class"],
            "baseline": bool(r["info"].get("baseline")),
            **({"pipeline": {"base_arm": r["info"]["base"], "rerank_depth": r["meta"]["arm_config"].get("rerank_depth"),
                             "reranker_job_vram_peak_gb": round(r["status_own"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
                             "note": "the base embedder indexes and retrieves the top 100, the reranker rescores it; index numbers are the base run's, query time is base plus rerank per query, peak VRAM is the larger of the two jobs"}}
               if r["info"].get("reranker") else {}),
            "quality": {
                "metric": "nDCG@10, MIRACL de dev (hard negatives), 305 queries",
                "value": r3(m["ndcg10"]), "ci95": [r3(x) for x in m["ndcg10_ci95"]],
                "skill_vs_bm25": r3(r["skill"]),
                "recall100": r3(m["recall100"]), "recall100_ci95": [r3(x) for x in m["recall100_ci95"]],
                "mrr10": r3(m["mrr10"]), "mrr10_ci95": [r3(x) for x in m["mrr10_ci95"]],
                "germandpr_ndcg10": r3(g["ndcg10"]), "germandpr_ndcg10_ci95": [r3(x) for x in g["ndcg10_ci95"]],
                "germandpr_recall100": r3(g["recall100"]), "germandpr_mrr10": r3(g["mrr10"]),
                **({"vs_best": {k: (r3(v) if isinstance(v, float) else ([r3(x) for x in v] if isinstance(v, list) else v)) for k, v in r["vs_best"].items()}} if "vs_best" in r else {}),
            },
            "speed": {
                "index_passages_per_s": r3(m["passages_per_s"]), "index_seconds_71277": r3(m["index_seconds"]),
                "query_ms_median": r3(m["query_ms_median"]), "query_ms_median_ci95": [r3(x) for x in m["query_ms_median_ci95"]],
                **({"rerank_ms_median": r3(m["rerank_ms_median"])} if "rerank_ms_median" in m else {}),
                "note": "indexing batch 64 (fp16 for the GPU models), query = encode plus exact search at batch 1 (rerankers: plus rescoring the top 100, pair batch 32); BM25 runs on the CPU",
            },
            "energy": {"index_j": r3(m["index_energy_j"]), "index_j_per_1000_passages": r3(m["index_energy_j_per_1000_passages"]), "note": "GPU only, sum of 1 s power samples during the MIRACL index section"},
            "vram_peak_gb": mc["vram_peak_gb"], "vram_peak_index_gb": m["index_peak_vram_gb"],
            "vram_peak_total_gb": mc["vram_peak_total_gb"], "fits_24gb": True,
            "truncated_passages": m["truncated_passages"],
        })
    return arms_out


# ---------------------------------------------------------------- C2

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_c2():
    res = {}
    for arm, info in C2_ARMS.items():
        meta, status, gpu = load_arm("c2", arm)
        rows = latest_rows("c2", arm)
        rows.sort(key=lambda r: (r["id"], r["file"]))
        err = [r["errors"] for r in rows]; den = [r["ref_words"] for r in rows]; cl = [r["id"] for r in rows]
        cerr = [edit_distance(r["ref_norm"], r["hyp_norm"]) for r in rows]; cden = [len(r["ref_norm"]) for r in rows]
        rtf = [r["seconds"] / r["duration_s"] for r in rows]
        audio_h = sum(r["duration_s"] for r in rows) / 3600
        secs = {s["name"]: s for s in meta["sections"]}
        b1, b16 = secs["batch1"], secs["batch16"]
        e1 = energy_j(gpu, b1["start"], b1["end"])
        long_ = [r for r in rows if r["duration_s"] > 30]
        short = [r for r in rows if r["duration_s"] <= 30]
        res[arm] = {
            "info": info, "meta": meta, "status": status, "rows": rows,
            "wer": sum(err) / sum(den), "wer_ci95": ci_ratio(err, den, cl),
            "cer": sum(cerr) / sum(cden), "cer_ci95": ci_ratio(cerr, cden, cl),
            "wer_num2words": meta["summary"].get("wer_num2words"),
            "wer_le30s": sum(r["errors"] for r in short) / sum(r["ref_words"] for r in short),
            "wer_gt30s": sum(r["errors"] for r in long_) / sum(r["ref_words"] for r in long_),
            "n_gt30s": len(long_),
            "rtf_median": float(np.median(rtf)), "rtf_median_ci95": ci_median(rtf),
            "rtf_pooled": sum(r["seconds"] for r in rows) / sum(r["duration_s"] for r in rows),
            "audio_h": audio_h, "utterances": len(rows), "sentences": len(set(cl)),
            "batch16_audio_h_per_wall_h": meta["summary"].get("batch16_audio_h_per_wall_h"),
            "batch16_seconds": b16["seconds"],
            "energy_batch1_j": e1, "energy_j_per_audio_h_batch1": e1 / audio_h,
            "energy_batch16_j": energy_j(gpu, b16["start"], b16["end"]),
            "share_with_digits": meta["summary"].get("share_with_digits"),
            "peak_vram_gb_torch": meta["summary"].get("peak_vram_gb_torch"),
        }
    best = min(res, key=lambda a: res[a]["wer"])
    order = sorted(res, key=lambda a: res[a]["wer"])
    for a in res:
        if a == best:
            continue
        rb, ra = res[best]["rows"], res[a]["rows"]
        assert [(r["id"], r["file"]) for r in rb] == [(r["id"], r["file"]) for r in ra]
        diff, ci, p = paired_ratio([r["errors"] for r in ra], [r["errors"] for r in rb], [r["ref_words"] for r in rb], [r["id"] for r in rb])
        res[a]["vs_best"] = {"best": best, "diff": diff, "diff_ci95": ci, "p": p,
                             "intervals_overlap": overlap(res[a]["wer_ci95"], res[best]["wer_ci95"]),
                             "difference_shown": p < 0.05 or not overlap(res[a]["wer_ci95"], res[best]["wer_ci95"])}
    # anchor of plan/c2.md: whisper-large-v3
    anchor = "whisper-large-v3"
    for a in res:
        if a == anchor:
            continue
        rb, ra = res[anchor]["rows"], res[a]["rows"]
        diff, ci, p = paired_ratio([r["errors"] for r in ra], [r["errors"] for r in rb], [r["ref_words"] for r in rb], [r["id"] for r in rb])
        res[a]["vs_anchor"] = {"anchor": anchor, "diff": diff, "diff_ci95": ci, "p": p}
    gate = res["canary-1b-v2"]["meta"].get("gate")
    return res, best, order, gate


def c2_outputs(res, order):
    out = []
    for a in order:
        r = res[a]; mc = machine_cols(r["status"], None)
        pc = lambda x: None if x is None else round(100 * x, 2)
        q = {
            "metric": "corpus WER in percent, FLEURS de_de test, 862 utterances, fixed normaliser bench/lib/norm_de.py",
            "value": pc(r["wer"]), "ci95": [pc(x) for x in r["wer_ci95"]],
            "cer": pc(r["cer"]), "cer_ci95": [pc(x) for x in r["cer_ci95"]],
            "wer_num2words_sensitivity": pc(r["wer_num2words"]),
            "wer_utterances_le_30s": pc(r["wer_le30s"]), "wer_utterances_gt_30s": pc(r["wer_gt30s"]), "utterances_gt_30s": r["n_gt30s"],
        }
        for k in ("vs_best", "vs_anchor"):
            if k in r:
                v = r[k]
                q[k] = {**{x: v[x] for x in v if x not in ("diff", "diff_ci95", "p")},
                        "diff_points": pc(v["diff"]), "diff_ci95_points": [pc(x) for x in v["diff_ci95"]], "p": round(v["p"], 3)}
        out.append({
            "arm": a, "model": r["info"]["model"], "revision": r["meta"]["arm_config"].get("revision"),
            "runtime": r["meta"]["arm_config"].get("runtime"), "dtype": r["meta"]["arm_config"].get("dtype") or r["meta"]["arm_config"].get("compute_type"),
            "params_m": r["info"]["params_m"], "params_source": r["info"]["params_source"],
            "licence": r["info"]["licence"], "licence_class": r["info"]["licence_class"],
            "quality": q,
            "speed": {"rtf_median_batch1": r3(r["rtf_median"]), "rtf_median_ci95": [r3(x) for x in r["rtf_median_ci95"]],
                      "rtf_pooled_batch1": r3(r["rtf_pooled"]), "x_realtime_batch1_pooled": round(1 / r["rtf_pooled"], 1),
                      "audio_h_per_wall_h_batch16": round(r["batch16_audio_h_per_wall_h"], 1)},
            "energy": {"j_per_audio_h_batch1": round(r["energy_j_per_audio_h_batch1"]), "batch1_j": round(r["energy_batch1_j"]), "batch16_j": round(r["energy_batch16_j"]),
                       "note": "GPU only, sum of 1 s power samples during the timed section"},
            "vram_peak_gb": mc["vram_peak_gb"], "vram_peak_total_gb": mc["vram_peak_total_gb"],
            "vram_peak_torch_gb": r3(r["peak_vram_gb_torch"]) if r["peak_vram_gb_torch"] else None, "fits_24gb": True,
        })
    return out


# ---------------------------------------------------------------- C3

def score_c3():
    gold = {g["id"]: g for g in rd_jsonl(ROOT / "reference" / "c3" / "gold.jsonl")}
    ids = sorted(gold)
    base = {r["id"]: r for r in rd_jsonl(ROOT / "results" / "c3" / "regex-baseline" / "result.jsonl")}
    base_cells = sum(sum(score_row(gold[i], base[i].get("pred")).values()) for i in ids)
    res = {}
    for arm, info in C3_ARMS.items():
        meta, status, gpu = load_arm("c3", arm)
        if meta is None:
            part = latest_rows("c3", arm, "-unfinished") or []
            res[arm] = {"info": info, "status": status, "finished": False, "partial": part, "gpu": gpu}
            continue
        rows = {r["id"]: r for r in latest_rows("c3", arm)}
        assert sorted(rows) == ids, arm
        cor = {i: score_row(gold[i], rows[i]["pred"] if isinstance(rows[i]["pred"], dict) else None) for i in ids}
        stored = sum(rows[i]["n_correct"] for i in ids); mine = sum(sum(c.values()) for c in cor.values())
        assert stored == mine, f"{arm}: rescored {mine} cells, the runner stored {stored}"
        per_inv = [sum(cor[i].values()) / 12 for i in ids]
        money_inv = [sum(cor[i][f] for f in MONEY) / 4 for i in ids]
        iban_inv = [float(cor[i]["iban"]) for i in ids]
        all12 = [float(all(cor[i].values())) for i in ids]
        secs = [(rows[i]["seconds_model"] or 0) + (rows[i]["seconds_ocr"] or 0) for i in ids if rows[i]["seconds_model"] is not None and not (info["kind"] == "ocr+text" and rows[i]["seconds_ocr"] is None)]
        spp = [((rows[i]["seconds_model"] or 0) + (rows[i]["seconds_ocr"] or 0)) / rows[i]["pages"] for i in ids if rows[i]["seconds_model"] is not None and not (info["kind"] == "ocr+text" and rows[i]["seconds_ocr"] is None)]
        e = energy_j(gpu, meta["timed_start"], meta["timed_end"])
        res[arm] = {
            "info": info, "meta": meta, "status": status, "finished": True, "cor": cor,
            "acc": float(np.mean(per_inv)), "acc_ci95": ci_mean(per_inv), "per_inv": per_inv,
            "money": float(np.mean(money_inv)), "money_ci95": ci_mean(money_inv),
            "iban": float(np.mean(iban_inv)), "iban_ci95": ci_mean(iban_inv),
            "all12": int(sum(all12)), "all12_share": float(np.mean(all12)),
            "per_field": {f: sum(cor[i][f] for i in ids) / len(ids) for f in FIELDS},
            "money_all_null": all(rows[i]["pred"] and all(rows[i]["pred"].get(f) is None for f in MONEY) for i in ids),
            "errors": sum(1 for i in ids if rows[i]["error"]), "truncated": sum(1 for i in ids if rows[i]["truncated"]),
            "s_per_invoice_median": float(np.median(secs)), "s_per_invoice_median_ci95": ci_median(secs),
            "s_per_page_median": float(np.median(spp)), "timed_invoices": len(secs),
            "energy_timed_j": e, "energy_j_per_invoice": e / len(ids),
        }
    finished = [a for a in res if res[a]["finished"] and not C3_ARMS[a].get("reference_arm")]
    best = max(finished, key=lambda a: (res[a]["acc"], -res[a]["status"]["gpu"]["peak_over_idle_mib"]))
    for a in res:
        if not res[a]["finished"] or a == best:
            continue
        diff, ci, p = paired_mean(res[best]["per_inv"], res[a]["per_inv"])
        res[a]["vs_best"] = {"best": best, "diff": diff, "diff_ci95": ci, "p": p,
                             "intervals_overlap": overlap(res[a]["acc_ci95"], res[best]["acc_ci95"]),
                             "difference_shown": p < 0.05 or not overlap(res[a]["acc_ci95"], res[best]["acc_ci95"])}
    order = sorted(finished, key=lambda a: -res[a]["acc"]) + [a for a in res if res[a]["finished"] and C3_ARMS[a].get("reference_arm")] + [a for a in res if not res[a]["finished"]]
    return res, best, order, base_cells / (12 * len(ids))


def c3_outputs(res, order, base):
    out = []
    for a in order:
        r = res[a]; info = r["info"]; mc = machine_cols(r["status"], None)
        d = {"arm": a, "model": info["model"], "kind": info["kind"], "licence": info["licence"], "licence_class": info["licence_class"],
             "reference_arm": bool(info.get("reference_arm")), "finished": r["finished"]}
        if r["finished"]:
            det = r["meta"]["model"]["details"]
            d.update({
                "ollama_digest": r["meta"]["model"]["digest"][:12], "quant": det.get("quantization_level"),
                "params_b": float(det["parameter_size"].rstrip("B")), "params_source": "Ollama parameter_size",
                "quality": {
                    "metric": "field accuracy, exact match after normalisation, 34 invoices x 12 fields = 408 cells",
                    "value": r3(r["acc"]), "ci95": [r3(x) for x in r["acc_ci95"]],
                    "skill_vs_regex": r3((r["acc"] - base) / (1 - base)),
                    "money_accuracy": r3(r["money"]), "money_ci95": [r3(x) for x in r["money_ci95"]],
                    "iban_accuracy": r3(r["iban"]), "iban_ci95": [r3(x) for x in r["iban_ci95"]],
                    "all12_invoices": r["all12"], "per_field": {f: r3(v) for f, v in r["per_field"].items()},
                    "money_fields_always_null": r["money_all_null"], "failed_calls": r["errors"], "truncated_calls": r["truncated"],
                    **({"vs_best": {"best": r["vs_best"]["best"], "diff": r3(r["vs_best"]["diff"]), "diff_ci95": [r3(x) for x in r["vs_best"]["diff_ci95"]], "p": round(r["vs_best"]["p"], 3), "difference_shown": r["vs_best"]["difference_shown"]}} if "vs_best" in r else {}),
                },
                "speed": {"s_per_invoice_median": r3(r["s_per_invoice_median"]), "s_per_invoice_median_ci95": [r3(x) for x in r["s_per_invoice_median_ci95"]],
                          "s_per_page_median": r3(r["s_per_page_median"]), "timed_invoices": r["timed_invoices"],
                          "note": "model call (plus docling OCR in the docling arms) per invoice, one call at a time; the first three docling invoices have their OCR time in the warm-up and are left out of the docling medians"},
                "energy": {"j_per_invoice": round(r["energy_j_per_invoice"]), "note": "GPU only, timed pass energy divided by 34"},
                "vram_peak_gb": mc["vram_peak_gb"], "vram_peak_total_gb": mc["vram_peak_total_gb"],
                "fits_24gb": True if mc["vram_peak_total_gb"] < CARD_GB - 0.5 else "offloaded",
                "fits_rule": FITS_RULE,
            })
        else:
            part = r["partial"]
            st = r["status"]
            d.update({
                "params_b": info.get("params_b"), "params_source": info.get("params_source"),
                "quality": None,
                "unfinished": {
                    "reason": st.get("reason", "").split(";")[0],
                    "timeout_min": round(st["gpu"]["gpu_min"]),
                    "invoices_logged": len(part),
                    "cells_right_logged": sum(p["n_correct"] for p in part),
                    "cells_logged": 12 * len(part),
                    "failed_calls_logged": sum(1 for p in part if p["error"]),
                    "s_per_invoice_median_logged": r3(statistics.median([p["seconds_model"] for p in part if p["seconds_model"] is not None])) if part else None,
                    "mean_power_w": st["gpu"]["mean_power_w"],
                    "note": "descriptive only, from the job log; not scored (plan/c3.md scores all 34 invoices)",
                },
                "vram_peak_gb": mc["vram_peak_gb"], "vram_peak_total_gb": mc["vram_peak_total_gb"],
                "fits_24gb": "offloaded" if mc["vram_peak_total_gb"] >= CARD_GB - 0.5 else True,
                "fits_rule": FITS_RULE,
            })
        out.append(d)
    return out


# ---------------------------------------------------------------- capability map

def pick_best(arms, key_value, key_ci, key_vram, higher_better=True, eligible=None):
    """ROADMAP rule: highest primary; if its interval overlaps the next model's and
    no paired test separates them, show the one with less peak VRAM and state the tie."""
    el = [a for a in arms if (eligible is None or eligible(a))]
    el.sort(key=lambda a: -key_value(a) if higher_better else key_value(a))
    top, nxt = el[0], el[1]
    shown, tie = top, None
    sep = arms_sep(top, nxt)
    if not sep:
        if key_vram(nxt) < key_vram(top):
            shown = nxt
        tie = {"top_by_value": top, "next": nxt, "note": "intervals overlap and the paired test does not separate the two; the one with less peak VRAM is shown"
               + ("" if key_vram(nxt) != key_vram(top) else " (equal peak VRAM, so the better value is shown)")}
    return shown, top, tie


def arms_sep(a, b):  # filled per capability
    return _SEP(a, b)


_SEP = None


def capability_rows(c1, c2, c3):
    global _SEP
    rows = []
    # C1
    res, best, order, gates = c1
    models = [a for a in order]
    d = {a: res[a]["per"]["miracl"] for a in models}
    _SEP = lambda a, b: res[b]["vs_best"]["difference_shown"] if res[a].get("vs_best") is None else res[a]["vs_best"]["difference_shown"]
    shown, top, tie = pick_best(models, lambda a: d[a]["ndcg10"], None, lambda a: res[a]["status"]["gpu"]["peak_over_idle_mib"])
    small = [a for a in models if a != top and not res[a]["vs_best"]["difference_shown"]]
    smallest = min([top] + small, key=lambda a: res[a]["status"]["gpu"]["peak_over_idle_mib"])
    rows.append({
        "id": "C1", "capability": "German retrieval for RAG (embedder, optional reranker)", "design": "plan/c1.md", "report": "results/c1/findings.md",
        "best_local_model": shown, "best_by_value": top, "tie": tie,
        "licence_class": C1_ARMS[shown]["licence_class"],
        "quality": {"metric": "nDCG@10 on MIRACL de dev", "value": r3(d[shown]["ndcg10"]), "ci95": [r3(x) for x in d[shown]["ndcg10_ci95"]],
                    "baseline": "BM25", "baseline_value": r3(res["bm25"]["per"]["miracl"]["ndcg10"]),
                    "reference": "MTEB bge-m3 0.5759 (sanity gate)", "reference_ours": r3(gates["bge-m3"]["value"])},
        "speed": {"unit": "ms per query (batch 1, encode plus exact search over 71,277 passages" + (", plus reranking the top 100)" if C1_ARMS[shown].get("reranker") else ")"), "value": r3(d[shown]["query_ms_median"]),
                  "index_passages_per_s": r3(d[shown]["passages_per_s"])},
        "vram_peak_gb": round(res[shown]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
        "energy": {"unit": "GPU joules per 1,000 passages indexed", "value": r3(d[shown]["index_energy_j_per_1000_passages"])},
        "fits_24gb": True,
        "smallest_within_interval": {"arm": smallest, "value": r3(d[smallest]["ndcg10"]), "vram_peak_gb": round(res[smallest]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
                                     "rule": "smallest peak VRAM among the arms the paired test does not separate from the best"},
    })
    # C2
    res2, best2, order2, gate2 = c2
    _SEP = lambda a, b: res2[b]["vs_best"]["difference_shown"] if "vs_best" not in res2[a] else res2[a]["vs_best"]["difference_shown"]
    shown2, top2, tie2 = pick_best(order2, lambda a: res2[a]["wer"], None, lambda a: res2[a]["status"]["gpu"]["peak_over_idle_mib"], higher_better=False)
    small2 = [a for a in order2 if a != top2 and not res2[a]["vs_best"]["difference_shown"]]
    smallest2 = min([top2] + small2, key=lambda a: res2[a]["status"]["gpu"]["peak_over_idle_mib"])
    rows.append({
        "id": "C2", "capability": "German speech to text", "design": "plan/c2.md", "report": "results/c2/findings.md",
        "best_local_model": shown2, "best_by_value": top2, "tie": tie2,
        "licence_class": C2_ARMS[shown2]["licence_class"],
        "quality": {"metric": "WER percent, FLEURS de_de test (lower is better)", "value": round(100 * res2[shown2]["wer"], 2), "ci95": [round(100 * x, 2) for x in res2[shown2]["wer_ci95"]],
                    "baseline": "whisper-large-v3 (anchor)", "baseline_value": round(100 * res2["whisper-large-v3"]["wer"], 2),
                    "reference": "canary-1b-v2 card 4.40 (sanity gate)", "reference_ours": round(gate2["value"], 2)},
        "speed": {"unit": "real-time factor, median at batch 1 (seconds of compute per second of audio)", "value": r3(res2[shown2]["rtf_median"]),
                  "audio_h_per_wall_h_batch16": round(res2[shown2]["batch16_audio_h_per_wall_h"], 1)},
        "vram_peak_gb": round(res2[shown2]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
        "energy": {"unit": "GPU joules per audio hour (batch 1)", "value": round(res2[shown2]["energy_j_per_audio_h_batch1"])},
        "fits_24gb": True,
        "smallest_within_interval": {"arm": smallest2, "value": round(100 * res2[smallest2]["wer"], 2), "vram_peak_gb": round(res2[smallest2]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
                                     "rule": "smallest peak VRAM among the arms the paired test does not separate from the best"},
    })
    # C3
    res3, best3, order3, base3 = c3
    cand3 = [a for a in order3 if res3[a]["finished"] and not C3_ARMS[a].get("reference_arm")]
    _SEP = lambda a, b: res3[b]["vs_best"]["difference_shown"] if "vs_best" not in res3[a] else res3[a]["vs_best"]["difference_shown"]
    shown3, top3, tie3 = pick_best(cand3, lambda a: res3[a]["acc"], None, lambda a: res3[a]["status"]["gpu"]["peak_over_idle_mib"])
    small3 = [a for a in cand3 if a != best3 and not res3[a]["vs_best"]["difference_shown"]]
    smallest3 = min([best3] + small3, key=lambda a: res3[a]["status"]["gpu"]["peak_over_idle_mib"])
    rows.append({
        "id": "C3", "capability": "Invoice fields from page images", "design": "plan/c3.md", "report": "results/c3/findings.md",
        "best_local_model": shown3, "best_by_value": top3, "tie": tie3,
        "licence_class": C3_ARMS[shown3]["licence_class"],
        "quality": {"metric": "field accuracy over 408 cells (34 invoices x 12 fields)", "value": r3(res3[shown3]["acc"]), "ci95": [r3(x) for x in res3[shown3]["acc_ci95"]],
                    "baseline": "regex over the PDF text layer", "baseline_value": r3(base3),
                    "reference": "none published; internal render check 391 of 391 (sanity gate)"},
        "speed": {"unit": "seconds per invoice, median (2 to 9 pages)", "value": r3(res3[shown3]["s_per_invoice_median"]), "s_per_page": r3(res3[shown3]["s_per_page_median"])},
        "vram_peak_gb": round(res3[shown3]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
        "energy": {"unit": "GPU joules per invoice", "value": round(res3[shown3]["energy_j_per_invoice"])},
        "fits_24gb": True,
        "smallest_within_interval": {"arm": smallest3, "value": r3(res3[smallest3]["acc"]), "vram_peak_gb": round(res3[smallest3]["status"]["gpu"]["peak_over_idle_mib"] / 1024, 2),
                                     "rule": "smallest peak VRAM among the arms the paired test does not separate from the best"},
        "unfinished": [a for a in order3 if not res3[a]["finished"]],
    })
    return rows


# ---------------------------------------------------------------- reports

def f3(x):
    return "" if x is None else f"{x:.3f}"


def ci(c, k=3):
    return f"[{c[0]:.{k}f}, {c[1]:.{k}f}]"


def write_c1(c1, arms_out):
    res, best, order, gates = c1
    L = ["# C1 score: German retrieval (embedders)", "",
         "Generated by `bench/score-cap.py score` from `results/c1/<arm>/*-run1.jsonl` and the GPU logs. Design: `plan/c1.md`.",
         "Intervals: 95 % bootstrap over queries (1,000 resamples, seed 20261009). Paired test: bootstrap of the per-query nDCG@10 difference to the best arm.", "",
         "## Sanity gate", "", "| arm | set | ours | reference | tolerance | passed |", "|---|---|---|---|---|---|"]
    for a, g in gates.items():
        L.append(f"| {a} | {g['set']} | {g['value']:.4f} | {g['reference']} | {g['tolerance']} | {'yes' if g['passed'] else 'no'} |")
    L += ["", "## MIRACL de dev (primary)", "",
          "| arm | params (M) | nDCG@10 | 95 % CI | skill vs BM25 | Recall@100 | MRR@10 | diff to best | p (paired) | difference shown |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in order + ["bm25"]:
        r = res[a]; m = r["per"]["miracl"]; v = r.get("vs_best")
        L.append(f"| {a} | {C1_ARMS[a]['params_m']} | {m['ndcg10']:.3f} | {ci(m['ndcg10_ci95'])} | {r['skill']:.3f} | {m['recall100']:.3f} | {m['mrr10']:.3f} | "
                 + (f"{-v['diff']:+.3f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} |" if v else "best | | |"))
    L += ["", "## GermanDPR test (secondary)", "", "| arm | nDCG@10 | 95 % CI | Recall@100 | MRR@10 |", "|---|---|---|---|---|"]
    for a in order + ["bm25"]:
        g = res[a]["per"]["germandpr"]
        L.append(f"| {a} | {g['ndcg10']:.3f} | {ci(g['ndcg10_ci95'])} | {g['recall100']:.3f} | {g['mrr10']:.3f} |")
    L += ["", "## Machine (MIRACL, 71,277 passages)", "",
          "| arm | index passages/s | index s | query ms (median, batch 1) | 95 % CI | peak VRAM GB (job) | peak VRAM GB (index) | index energy J | J per 1,000 passages | truncated passages |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in arms_out:
        s = x["speed"]; e = x["energy"]
        L.append(f"| {x['arm']} | {s['index_passages_per_s']:.0f} | {s['index_seconds_71277']:.1f} | {s['query_ms_median']:.1f} | {ci(s['query_ms_median_ci95'], 1)} | {x['vram_peak_gb']:.2f} | {x['vram_peak_index_gb'] if x['vram_peak_index_gb'] is not None else ''} | {e['index_j']:.0f} | {e['index_j_per_1000_passages']:.0f} | {x['truncated_passages'] if x['truncated_passages'] is not None else 'n/a'} |")
    L += ["", "Peak VRAM is nvidia-smi memory.used minus the idle value before the model loaded; BM25 runs on the CPU. Energy is GPU only.", ""]
    (ROOT / "results" / "c1" / "score.md").write_text("\n".join(L), encoding="utf-8")


def write_c2(c2, arms_out):
    res, best, order, gate = c2
    L = ["# C2 score: German speech to text", "",
         "Generated by `bench/score-cap.py score` from `results/c2/<arm>/*-run1.jsonl` and the GPU logs. Design: `plan/c2.md`.",
         "WER and CER are corpus rates in percent after `bench/lib/norm_de.py`. Intervals: 95 % bootstrap over sentence ids (347 clusters, 1,000 resamples, seed 20261009). Paired test: cluster bootstrap of the WER difference.", "",
         "## Sanity gate", "", f"canary-1b-v2: WER {gate['value']:.2f} against the card's 4.40, tolerance 1.5 points: {'passed' if gate['passed'] else 'failed'}. The parakeet card gives 5.04; ours is {100 * res['parakeet-v3']['wer']:.2f}.", "",
         "## Quality", "",
         "| arm | params (M) | WER | 95 % CI | CER | diff to best (points) | p | difference shown | diff to whisper-large-v3 | p | WER <=30 s | WER >30 s (n) | WER num2words |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        r = res[a]; v = r.get("vs_best"); w = r.get("vs_anchor")
        L.append(f"| {a} | {C2_ARMS[a]['params_m']} | {100 * r['wer']:.2f} | {ci([100 * x for x in r['wer_ci95']], 2)} | {100 * r['cer']:.2f} | "
                 + (f"{100 * v['diff']:+.2f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} | " if v else "best | | | ")
                 + (f"{100 * w['diff']:+.2f} | {w['p']:.3f} | " if w else "anchor | | ")
                 + f"{100 * r['wer_le30s']:.2f} | {100 * r['wer_gt30s']:.2f} ({r['n_gt30s']}) | {100 * r['wer_num2words']:.2f} |")
    L += ["", "## Machine (3.15 h of audio)", "",
          "| arm | runtime | RTF median batch 1 | 95 % CI | x real time (pooled, batch 1) | audio h per wall h (batch 16) | peak VRAM GB (nvidia-smi) | peak VRAM GB (torch) | J per audio h (batch 1) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in arms_out:
        s = x["speed"]
        L.append(f"| {x['arm']} | {x['runtime']} | {s['rtf_median_batch1']:.4f} | {ci(s['rtf_median_ci95'], 4)} | {s['x_realtime_batch1_pooled']} | {s['audio_h_per_wall_h_batch16']} | {x['vram_peak_gb']:.2f} | {x['vram_peak_torch_gb'] if x['vram_peak_torch_gb'] is not None else 'n/a'} | {x['energy']['j_per_audio_h_batch1']} |")
    L += ["", "The nvidia-smi peak includes memory the framework reserves but does not use (the NeMo arms run in float32 and reserve much more than torch reports allocated). Energy is GPU only.", ""]
    (ROOT / "results" / "c2" / "score.md").write_text("\n".join(L), encoding="utf-8")


def write_c3(c3, arms_out):
    res, best, order, base = c3
    L = ["# C3 score: invoice fields from rendered e-invoices", "",
         "Generated by `bench/score-cap.py score` from `results/c3/<arm>/*-run1.jsonl`, `reference/c3/gold.jsonl` and the GPU logs. Design: `plan/c3.md`, method `results/c3/method.md`.",
         "Every cell is rescored with `bench/c3/fields.py` and must equal the runner's count. Intervals: 95 % bootstrap over invoices (34, 1,000 resamples, seed 20261009). Paired test: bootstrap of the per-invoice accuracy difference to the best arm.", "",
         f"Regex baseline over the PDF text layer: {base:.3f} ({round(base * 408)} of 408 cells).", "",
         "## Quality", "",
         "| arm | kind | params (B) | field accuracy | 95 % CI | skill vs regex | money fields | IBAN | all 12 right | diff to best | p | difference shown |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in arms_out:
        if not x["finished"]:
            u = x["unfinished"]
            L.append(f"| {x['arm']} | {x['kind']} | {x['params_b']} | not finished | | | | | | | | |")
            continue
        q = x["quality"]; v = q.get("vs_best")
        L.append(f"| {x['arm']}{' (reference, text layer)' if x['reference_arm'] else ''} | {x['kind']} | {x['params_b']} | {q['value']:.3f} | {ci(q['ci95'])} | {q['skill_vs_regex']:.3f} | {q['money_accuracy']:.3f} | {q['iban_accuracy']:.3f} | {q['all12_invoices']} | "
                 + (f"{-v['diff']:+.3f} | {v['p']:.3f} | {'yes' if v['difference_shown'] else 'no'} |" if v else "best | | |"))
    L += ["", "## Per field (share of 34 invoices right, descriptive)", "", "| arm | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for x in arms_out:
        if x["finished"]:
            L.append(f"| {x['arm']} | " + " | ".join(f"{x['quality']['per_field'][f]:.2f}" for f in FIELDS) + " |")
    L += ["", "## Machine", "",
          "| arm | s per invoice (median) | 95 % CI | s per page (median) | peak VRAM GB (over idle) | peak VRAM GB (total) | J per invoice | fits on 24 GB |",
          "|---|---|---|---|---|---|---|---|"]
    for x in arms_out:
        if x["finished"]:
            s = x["speed"]
            L.append(f"| {x['arm']} | {s['s_per_invoice_median']:.2f} | {ci(s['s_per_invoice_median_ci95'], 2)} | {s['s_per_page_median']:.2f} | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} | {x['energy']['j_per_invoice']} | {'yes' if x['fits_24gb'] is True else x['fits_24gb']} |")
        else:
            u = x["unfinished"]
            L.append(f"| {x['arm']} | not finished: {u['reason']}; {u['invoices_logged']} of 34 invoices logged, median {u['s_per_invoice_median_logged']} s | | | {x['vram_peak_gb']:.2f} | {x['vram_peak_total_gb']:.2f} | | {'yes' if x['fits_24gb'] is True else x['fits_24gb']} |")
    L += ["", "## Unfinished arms (descriptive, from the job log, not scored)", "",
          "| arm | invoices logged | cells right of logged | failed calls | median s per logged invoice | mean GPU power W |", "|---|---|---|---|---|---|"]
    for x in arms_out:
        if not x["finished"]:
            u = x["unfinished"]
            L.append(f"| {x['arm']} | {u['invoices_logged']} | {u['cells_right_logged']} of {u['cells_logged']} | {u['failed_calls_logged']} | {u['s_per_invoice_median_logged']} | {u['mean_power_w']} |")
    L += ["", "Peak VRAM from nvidia-smi over the whole job; total = including the idle desktop. Energy is GPU only, timed pass divided by 34.", ""]
    (ROOT / "results" / "c3" / "score.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["import", "score"])
    ap.add_argument("--queue", default=str(ROOT / ".cache" / "queue"))
    a = ap.parse_args()
    if a.cmd == "import":
        do_import(a.queue)
        return
    c1 = score_c1(); c2 = score_c2(); c3 = score_c3()
    o1 = c1_outputs(c1[0], c1[1], c1[2], c1[3]); o2 = c2_outputs(c2[0], c2[2]); o3 = c3_outputs(c3[0], c3[2], c3[3])
    common = {"machine": "local-rtx4090: one RTX 4090 24 GB, Windows 11, Ryzen 9 7900, 128 GB RAM (results/<cN>/<arm>/queue-status.json)", "generator": "bench/score-cap.py"}
    wr_json(ROOT / "results" / "c1" / "summary.json", {"schema": "jev-match-bench capability summary v1", "test": "C1", "design": "plan/c1.md", "report": "results/c1/findings.md", **common,
                                                       "best_by_ndcg10": c1[1], "gates": c1[3], "arms": o1})
    wr_json(ROOT / "results" / "c2" / "summary.json", {"schema": "jev-match-bench capability summary v1", "test": "C2", "design": "plan/c2.md", "report": "results/c2/findings.md", **common,
                                                       "best_by_wer": c2[1], "gate": c2[3], "arms": o2})
    wr_json(ROOT / "results" / "c3" / "summary.json", {"schema": "jev-match-bench capability summary v1", "test": "C3", "design": "plan/c3.md", "report": "results/c3/findings.md", **common,
                                                       "best_by_field_accuracy": c3[1], "regex_baseline": r3(c3[3]), "arms": o3})
    write_c1(c1, o1); write_c2(c2, o2); write_c3(c3, o3)
    caps = capability_rows(c1, c2, c3)
    sys.path.insert(0, str(ROOT / "bench"))
    import cap_c4c8  # noqa: PLC0415
    for test, fn in (("c4", cap_c4c8.score_c4), ("c6", cap_c4c8.score_c6), ("c7", cap_c4c8.score_c7), ("c8", cap_c4c8.score_c8)):
        got = fn()
        if got is None:
            continue
        summ, md, row = got
        wr_json(ROOT / "results" / test / "summary.json", {"schema": "jev-match-bench capability summary v1", **{k: v for k, v in summ.items() if k != "arms"}, **common, "arms": summ["arms"]})
        (ROOT / "results" / test / "score.md").write_text(md, encoding="utf-8")
        if row is not None:
            caps.append(row)
    wr_json(ROOT / "results" / "capabilities.json", {"schema": "jev-match-bench capability map v1", "columns": "plan/ROADMAP.md, section Capability map for the website", **common, "rows": caps})
    for c in caps:
        print(c["id"], c["best_local_model"], c["quality"]["value"], c["quality"]["ci95"], "tie" if c["tie"] else "", "smallest:", c["smallest_within_interval"]["arm"])


if __name__ == "__main__":
    main()
