"""C3 runner: invoice fields from rendered e-invoices (plan/c3.md, results/c3/method.md).

Runs one arm against a local Ollama, writes <out>/result.jsonl and <out>/meta.json.
Stdlib only, except docling for the docling+ arms.

  python bench/c3/run_c3.py --arm qwen3-vl-8b --data D:\\bench-data\\c3 --out D:\\jmb-queue\\out\\<id>
  python bench/c3/run_c3.py --arm qwen3-vl-8b --data cases/c3 --out /tmp/x --mock   (no model, IO check)

--data holds png/, text/ and manifest.json (cases/c3 of the repo, or the copy on the PC).
Gold and method come from the repo (cwd = repo root).
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import FIELDS, MONEY, score_row  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
NUM_CTX = 32768
OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX}
KEEP = "30m"

# arm -> (kind, ollama tag)
ARMS = {
    "qwen3-vl-8b": ("vision", "qwen3-vl:8b"),
    "qwen3-vl-32b": ("vision", "qwen3-vl:32b"),
    "gemma4-12b": ("vision", "gemma4:12b"),
    "gemma4-26b": ("vision", "gemma4:26b"),
    "mistral-small-3.2": ("vision", "mistral-small3.2:24b"),
    "docling+qwen3.8-27b": ("docling", "qwen3.8:27b"),
    "docling+gemma4-12b": ("docling", "gemma4:12b"),
    "pdftext+qwen3.8-27b": ("pdftext", "qwen3.8:27b"),
}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_method(repo):
    s = open(os.path.join(repo, "results", "c3", "method.md"), encoding="utf-8").read()
    instr = re.search(r"## Instruction\s+```text\n(.*?)\n```", s, re.S).group(1)
    schema = json.loads(re.search(r"```json\n(.*?)\n```", s, re.S).group(1))
    return instr, schema


def http(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def model_info(tag):
    try:
        for m in http("/api/tags", timeout=30).get("models", []):
            if m.get("name") == tag or m.get("model") == tag:
                info = {"digest": m.get("digest"), "size": m.get("size"), "details": m.get("details")}
                try:  # weight blobs named in the Modelfile (FROM lines), the digests the library page lists
                    sh = http("/api/show", {"model": tag}, timeout=60)
                    info["blobs"] = re.findall(r"sha256-([0-9a-f]{64})", sh.get("modelfile", ""))
                    info["capabilities"] = sh.get("capabilities")
                except Exception as e:  # noqa: BLE001
                    info["blobs_error"] = repr(e)
                return info
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return {"error": "tag not found in /api/tags"}


def chat(tag, content, images, schema, keep_alive, think_off):
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    body = {"model": tag, "messages": [msg], "format": schema, "options": OPTIONS,
            "stream": False, "keep_alive": keep_alive}
    if think_off:  # thinking off for every model that can think (results/c3/method.md, Options)
        body["think"] = False
    last = None
    for _ in range(2):
        try:
            t0 = time.perf_counter()
            r = http("/api/chat", body)
            dt = time.perf_counter() - t0
            raw = r.get("message", {}).get("content", "")
            return {"raw": raw, "pred": json.loads(raw), "seconds": dt,
                    "prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count"),
                    "load_duration_s": (r.get("load_duration") or 0) / 1e9,
                    "total_duration_s": (r.get("total_duration") or 0) / 1e9, "error": None}
        except Exception as e:  # noqa: BLE001
            last = repr(e)
    return {"raw": None, "pred": None, "seconds": None, "prompt_eval_count": None, "eval_count": None,
            "load_duration_s": None, "total_duration_s": None, "error": last}


def mock_chat(gold_row):
    pred = {f: gold_row[f] for f in FIELDS}
    return {"raw": json.dumps(pred), "pred": pred, "seconds": 0.0, "prompt_eval_count": 0, "eval_count": 0,
            "load_duration_s": 0.0, "total_duration_s": 0.0, "error": None}


class Docling:
    """docling image pipeline with EasyOCR set explicitly (docling's own option defaults:
    languages en, es, fr, de; GPU if torch sees one). The automatic engine choice of
    docling 2.136 fell through to no engine at all when rapidocr 3.10 was installed
    (probe 2026-10-09) and returned empty Markdown, so the engine is named here."""

    ENGINE = "easyocr"

    def __init__(self):
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter, ImageFormatOption  # noqa: PLC0415
        self.ocr_options = EasyOcrOptions()
        opts = PdfPipelineOptions(do_ocr=True, ocr_options=self.ocr_options)
        self.conv = DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})

    def page(self, png):
        md = self.conv.convert(png).document.export_to_markdown()
        if len(md.strip()) < 50:
            raise RuntimeError(f"docling returned no text for {png} (OCR engine missing?)")
        return md


def versions(kind):
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        v["ollama"] = http("/api/version", timeout=10).get("version")
    except Exception as e:  # noqa: BLE001
        v["ollama"] = repr(e)
    if kind == "docling":
        from importlib.metadata import version, PackageNotFoundError  # noqa: PLC0415
        for p in ("docling", "docling-core", "docling-ibm-models", "easyocr", "rapidocr", "rapidocr-onnxruntime",
                  "torch", "onnxruntime", "onnxruntime-gpu"):
            try:
                v[p] = version(p)
            except PackageNotFoundError:
                pass
        try:
            import torch  # noqa: PLC0415
            v["torch_cuda_available"] = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--limit", type=int, default=0, help="first N invoices only (probe)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mock", action="store_true", help="no model calls, gold as prediction (IO check)")
    a = ap.parse_args()
    kind, tag = ARMS[a.arm]
    os.makedirs(a.out, exist_ok=True)
    instr, schema = load_method(a.repo)
    gold = [json.loads(l) for l in open(os.path.join(a.repo, "reference", "c3", "gold.jsonl"), encoding="utf-8")]
    manifest = json.load(open(os.path.join(a.data, "manifest.json"), encoding="utf-8"))
    items = {x["id"]: x for x in manifest["items"]}

    # input integrity: every file of the frozen set must match the manifest
    bad = []
    for g in gold:
        it = items[g["id"]]
        for p in it["png"]:
            if sha(os.path.join(a.data, "png", p["file"])) != p["sha256"]:
                bad.append(p["file"])
        if sha(os.path.join(a.data, "text", g["id"] + ".txt")) != it["text_sha256"]:
            bad.append(g["id"] + ".txt")
    if bad:
        sys.exit("input hash mismatch: " + ", ".join(bad[:10]))
    if a.limit:
        gold = gold[: a.limit]

    meta = {"test": "C3", "arm": a.arm, "kind": kind, "ollama_tag": tag, "num_ctx": NUM_CTX,
            "options": OPTIONS, "job_id": os.environ.get("JMB_JOB_ID"), "mock": a.mock,
            "model": None if a.mock else model_info(tag), "versions": versions(kind),
            "inputs": {"manifest_sha256": sha(os.path.join(a.data, "manifest.json")),
                       "gold_sha256": sha(os.path.join(a.repo, "reference", "c3", "gold.jsonl")),
                       "method_sha256": sha(os.path.join(a.repo, "results", "c3", "method.md"))},
            "started": now()}

    caps = (meta["model"] or {}).get("capabilities") or []
    think_off = "thinking" in caps
    meta["think"] = False if think_off else "not supported by the model"
    doc = None
    ocr_cache = {}
    if kind == "docling" and not a.mock:
        t0 = time.perf_counter()
        meta["docling_init_start"] = now()
        doc = Docling()
        meta["docling_init_s"] = time.perf_counter() - t0
        meta["docling_ocr"] = {"engine": Docling.ENGINE, "options": doc.ocr_options.model_dump(mode="json")}

    def build(g):
        it = items[g["id"]]
        pngs = [os.path.join(a.data, "png", p["file"]) for p in it["png"]]
        if kind == "vision":
            return instr, [base64.b64encode(open(p, "rb").read()).decode() for p in pngs], 0.0
        if kind == "pdftext":
            txt = open(os.path.join(a.data, "text", g["id"] + ".txt"), encoding="utf-8").read()
            return instr + "\n\nRechnung:\n" + txt, None, 0.0
        if g["id"] not in ocr_cache:
            t0 = time.perf_counter()
            md = "\n\n".join(doc.page(p) for p in pngs) if doc else "(mock)"
            ocr_cache[g["id"]] = (md, time.perf_counter() - t0)
        md, ocr_s = ocr_cache[g["id"]]
        return instr + "\n\nRechnung:\n" + md, None, ocr_s

    # warm-up, not scored (also loads the model)
    meta["warmup_start"] = now()
    for g in gold[: a.warmup]:
        content, images, _ = build(g)
        if not a.mock:
            chat(tag, content, images, schema, KEEP, think_off)
    meta["timed_start"] = now()

    rows = []
    for n, g in enumerate(gold):
        it = items[g["id"]]
        content, images, ocr_s = build(g)
        if kind == "docling" and n < a.warmup:
            ocr_s = None  # OCR of these pages ran during warm-up; its time is in the warm-up, not here
        last = n == len(gold) - 1
        r = mock_chat(g) if a.mock else chat(tag, content, images, schema, "0" if last else KEEP, think_off)
        pred = r["pred"] if isinstance(r["pred"], dict) else None
        corr = score_row(g, pred)
        trunc = bool(r["prompt_eval_count"] and r["prompt_eval_count"] >= NUM_CTX - 64)
        row = {"id": g["id"], "arm": a.arm, "pages": it["pages"], "pred": pred, "raw": r["raw"],
               "correct": corr, "n_correct": sum(corr.values()), "seconds_model": r["seconds"],
               "seconds_ocr": ocr_s, "prompt_eval_count": r["prompt_eval_count"], "eval_count": r["eval_count"],
               "load_duration_s": r["load_duration_s"], "truncated": trunc, "error": r["error"]}
        if kind == "docling":
            row["ocr_markdown_sha256"] = hashlib.sha256(ocr_cache[g["id"]][0].encode()).hexdigest()
        rows.append(row)
        print(f"{g['id']} {row['n_correct']}/12 {r['seconds']} {r['error'] or ''}", flush=True)
    meta["timed_end"] = now()

    if kind == "docling" and not a.mock:
        with open(os.path.join(a.out, "ocr.jsonl"), "w", encoding="utf-8") as fo:
            for i, (md, s) in ocr_cache.items():
                fo.write(json.dumps({"id": i, "seconds": s, "markdown": md}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for row in rows:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    cells = len(rows) * len(FIELDS)
    secs = [r["seconds_model"] for r in rows if r["seconds_model"] is not None]
    per_inv = [r["seconds_model"] + (r["seconds_ocr"] or 0) for r in rows
               if r["seconds_model"] is not None and (kind != "docling" or r["seconds_ocr"] is not None)]
    meta["summary"] = {
        "invoices": len(rows), "cells": cells,
        "field_accuracy": sum(r["n_correct"] for r in rows) / cells if cells else None,
        "money_accuracy": sum(r["correct"][f] for r in rows for f in MONEY) / (4 * len(rows)) if rows else None,
        "iban_accuracy": sum(r["correct"]["iban"] for r in rows) / len(rows) if rows else None,
        "all12": sum(r["n_correct"] == 12 for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "median_s_per_invoice": statistics.median(per_inv) if per_inv else None,
        "median_s_per_page": statistics.median((r["seconds_model"] + (r["seconds_ocr"] or 0)) / r["pages"]
                                               for r in rows if r["seconds_model"] is not None) if secs else None,
        "pages": sum(r["pages"] for r in rows),
    }
    meta["finished"] = now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(meta["summary"]))
    if rows and meta["summary"]["errors"] == len(rows):
        sys.exit("every call failed, see result.jsonl")


if __name__ == "__main__":
    main()
