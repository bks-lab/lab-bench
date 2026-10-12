"""C9 runner: personal data detection in German text (plan/c9.md, notes of 2026-10-12).

  python bench/c9/run_c9.py --arm regex --out <dir> --gate 0.95      sanity arm, CPU, exit 3 below the gate
  python bench/c9/run_c9.py --arm presidio --out <dir>               CPU
  python bench/c9/run_c9.py --arm gliner-multi-pii --out <dir>       gliner package (v1), GPU
  python bench/c9/run_c9.py --arm gliner2.5-multi --out <dir>        gliner2 package, GPU
  python bench/c9/run_c9.py --arm qwen3.8-27b --out <dir>            Ollama
  ... [--limit N] [--mock]

All arms read the 476 frozen documents (docs.jsonl under --data, hash-checked
against cases/c9/manifest.json)
and write <out>/result.jsonl (one row per document: predicted spans with
character offsets and one of the seven types) and <out>/meta.json (versions,
settings, timed sections for the queue's GPU sampler, summary). Scoring is
bench/c9/score_c9.py; the runner prints a span F1 per arm only for the log.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import c9lib  # noqa: E402
import ollama as ol  # noqa: E402

GLINER = {
    "gliner-multi-pii": {"model": "urchade/gliner_multi_pii-v1", "rev": "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d", "lib": "gliner"},
    "gliner-pii-domains": {"model": "E3-JSI/gliner-multi-pii-domains-v1", "rev": "66b7951b1e020535d2f470d4b35bc90149a1ed0d", "lib": "gliner"},
    "gliner2.5-multi": {"model": "fastino/gliner2.5-multi-v1", "rev": "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3", "lib": "gliner2"},
}
THRESHOLD = 0.5
LLMS = {"qwen3.8-27b": "qwen3.8:27b", "gemma4-12b": "gemma4:12b"}
OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": 8192, "num_predict": 2048}
KEEP = "30m"
SPACY_MODEL = "de_core_news_lg"


def llm_format(types):
    return {"type": "object", "properties": {"entities": {"type": "array", "items": {
        "type": "object", "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": types}},
        "required": ["text", "type"]}}}, "required": ["entities"]}


def llm_prompt(labels):
    lines = "\n".join(f"- {t}: {d}" for t, d in labels["llm_types"].items())
    return ("Finde alle personenbezogenen Daten im folgenden Dokument. Erlaubte Typen:\n"
            f"{lines}\n\n"
            "Gib jede Fundstelle genau so wieder, wie sie im Dokument steht (gleiche Schreibweise, "
            "keine Ergänzungen), mit ihrem Typ. Firmennamen, allgemeine Datumsangaben und Beträge "
            "sind keine personenbezogenen Daten. Antworte nur mit JSON der Form "
            "{\"entities\": [{\"text\": \"...\", \"type\": \"...\"}]}.\n\n"
            "Dokument:\n{document}")


def versions(names):
    from importlib.metadata import version  # noqa: PLC0415
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:  # noqa: BLE001
            out[n] = None
    return out


def snapshot_rev(repo, rev):
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        return os.path.basename(snapshot_download(repo, revision=rev, local_files_only=True))
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e!r}"[:200]


def merge(preds):
    """Unique (start, end, type), keeping the highest score."""
    best = {}
    for p in preds:
        k = (p["start"], p["end"], p["type"])
        if k not in best or (p.get("score") or 0) > (best[k].get("score") or 0):
            best[k] = p
    return sorted(best.values(), key=lambda p: (p["start"], p["end"], p["type"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["regex", "presidio"] + sorted(GLINER) + sorted(LLMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--data", default=None, help="folder with docs.jsonl (the workstation: D:\\bench-data\\c9)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gate", type=float, default=None, help="regex arm: minimum recall on the gold email spans")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man, docs, labels = c9lib.load(a.repo, a.data)
    if a.limit:
        docs = docs[: a.limit]
    types = labels["types"]
    meta = {"test": "C9", "arm": a.arm, "job_id": os.environ.get("JMB_JOB_ID"), "mock": a.mock, "limit": a.limit,
            "inputs": {"docs_sha256": man["docs_sha256"], "labels_sha256": c9lib.sha(os.path.join(a.repo, "reference", "c9", "labels.json"))},
            "boundary_rule": "c9lib.norm", "sections": [], "started": ol.now()}

    def section(name, **kw):
        s = {"name": name, "start": ol.now(), **kw}
        meta["sections"].append(s)
        return s

    load = section("load")
    t_load = time.perf_counter()
    if a.arm == "regex":
        meta["versions"] = versions([])

        def detect(text):
            return [{"start": s, "end": e, "type": t} for s, e, t in c9lib.regex_spans(text)], None, {}

    elif a.arm == "presidio":
        meta["versions"] = versions(["presidio-analyzer", "spacy", SPACY_MODEL.replace("_", "-")])
        meta["device"] = "cpu"
        ent_map = labels["presidio_entities"]
        if a.mock:
            analyzer = None
        else:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry  # noqa: PLC0415
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: PLC0415
            from presidio_analyzer.predefined_recognizers import (  # noqa: PLC0415
                CreditCardRecognizer, EmailRecognizer, IbanRecognizer, IpRecognizer, PhoneRecognizer, SpacyRecognizer)
            nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [
                {"lang_code": "de", "model_name": SPACY_MODEL}]}).create_engine()
            reg = RecognizerRegistry(supported_languages=["de"])
            # the built-in pattern recognisers default to English; the same classes, language de
            for cls, kw in ((EmailRecognizer, {}), (IbanRecognizer, {}), (CreditCardRecognizer, {}), (IpRecognizer, {}),
                            (PhoneRecognizer, {"supported_regions": ["DE", "AT", "CH", "US", "GB"]}),
                            (SpacyRecognizer, {})):
                reg.add_recognizer(cls(supported_language="de", **kw))
            analyzer = AnalyzerEngine(registry=reg, nlp_engine=nlp, supported_languages=["de"])
            meta["presidio"] = {"recognizers": [r.name for r in reg.recognizers], "score_threshold": "library default",
                                "entities": sorted(ent_map), "phone_regions": ["DE", "AT", "CH", "US", "GB"]}

        def detect(text):
            if a.mock:
                return [], None, {}
            res = analyzer.analyze(text=text, language="de", entities=sorted(ent_map))
            return [{"start": r.start, "end": r.end, "type": ent_map[r.entity_type], "label": r.entity_type,
                     "score": round(float(r.score), 4)} for r in res if r.entity_type in ent_map], None, {}

    elif a.arm in GLINER:
        spec = GLINER[a.arm]
        type_of = labels["gliner_labels"]
        label_list = list(type_of)
        meta["model"] = {**spec, "snapshot": None if a.mock else snapshot_rev(spec["model"], spec["rev"]),
                         "threshold": THRESHOLD, "labels": label_list,
                         "chunk_words": c9lib.CHUNK_WORDS, "chunk_overlap": c9lib.CHUNK_OVERLAP}
        if a.mock:
            model = None
        else:
            import torch  # noqa: PLC0415
            meta["versions"] = versions(["torch", "transformers", spec["lib"]])
            meta["versions"]["cuda_available"] = torch.cuda.is_available()
            if spec["lib"] == "gliner":
                from gliner import GLiNER  # noqa: PLC0415
                model = GLiNER.from_pretrained(spec["model"], revision=spec["rev"])
            else:
                from gliner2 import AutoExtractor  # noqa: PLC0415
                model = AutoExtractor.from_pretrained(spec["model"], revision=spec["rev"])
            model.to(a.device)
            model.eval()

        def run_chunk(chunk):
            if spec["lib"] == "gliner":
                return [{"start": e["start"], "end": e["end"], "label": e["label"], "score": float(e["score"])}
                        for e in model.predict_entities(chunk, label_list, threshold=THRESHOLD)]
            res = model.extract_entities(chunk, label_list, threshold=THRESHOLD, include_confidence=True, include_spans=True)
            out = []
            for lab, items in (res.get("entities") or {}).items():
                for it in items:
                    if isinstance(it, dict) and "start" in it:
                        out.append({"start": it["start"], "end": it["end"], "label": lab, "score": float(it.get("confidence", 1.0))})
            return out

        def detect(text):
            if a.mock:
                return [], None, {"chunks": len(c9lib.chunks(text))}
            preds = []
            spans = c9lib.chunks(text)
            for cs, ce in spans:
                for e in run_chunk(text[cs:ce]):
                    preds.append({"start": cs + e["start"], "end": cs + e["end"], "type": type_of[e["label"]],
                                  "label": e["label"], "score": round(e["score"], 4)})
            if a.device == "cuda":
                torch.cuda.synchronize()
            return merge(preds), None, {"chunks": len(spans)}

    else:
        tag = LLMS[a.arm]
        prompt = llm_prompt(labels)
        fmt = llm_format(types)
        info = None if a.mock else ol.model_info(tag)
        think = None if a.mock else ol.think_setting(info, False)
        meta.update({"ollama_tag": tag, "model": info, "options": OPTIONS, "format": fmt, "prompt": prompt,
                     "think": think if think is not None else "not supported by the model",
                     "offsets": "every non-overlapping exact occurrence of the returned text; not found = false positive",
                     "versions": {"python": sys.version.split()[0], "ollama": None if a.mock else ol.version()}})

        def detect(text):
            if a.mock:
                return [], None, {}
            body = {"model": tag, "messages": [{"role": "user", "content": prompt.replace("{document}", text)}],
                    "format": fmt, "options": OPTIONS, "stream": False, "keep_alive": KEEP}
            if think is not None:
                body["think"] = think
            resp, _, err = ol.call("/api/chat", body, timeout=900)
            raw = ((resp or {}).get("message") or {}).get("content")
            preds, not_found = [], []
            try:
                ents = json.loads(raw)["entities"]
            except Exception:  # noqa: BLE001
                return [], err or "answer did not parse", {"raw": raw, **ol.counters(resp)}
            for e in ents:
                t, s = e.get("type"), (e.get("text") or "").strip()
                hits = c9lib.find_all(text, s) if t in types else []
                if not hits:
                    not_found.append({"text": s, "type": t})
                for hs, he in hits:
                    preds.append({"start": hs, "end": he, "type": t})
            return merge(preds), err, {"raw": raw, "not_found": not_found, **ol.counters(resp)}

    load["end"] = ol.now()
    load["seconds"] = round(time.perf_counter() - t_load, 3)

    warm = section("warmup")
    for d in docs[: a.warmup]:
        detect(d["text"])
    warm["end"] = ol.now()
    timed = section("timed", items=len(docs))
    if a.arm in LLMS and not a.mock:
        timed["ps_before"] = ol.ps()
    tp = fp = fn = 0
    email_hit = email_n = 0
    secs_all = []
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for n, d in enumerate(docs):
            t0 = time.perf_counter()
            try:
                preds, err, extra = detect(d["text"])
            except Exception as e:  # noqa: BLE001
                preds, err, extra = [], f"{type(e).__name__}: {e}"[:300], {}
            secs = time.perf_counter() - t0
            secs_all.append(secs)
            for p in preds:
                p["text"] = d["text"][p["start"]:p["end"]]
            row = {"id": d["id"], "arm": a.arm, "pred": preds, "seconds": round(secs, 4), "chars": len(d["text"]),
                   "error": err, **extra}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            g = {(s, e) for s, e, _ in c9lib.gold(d, labels)}
            pb = {b for b in (c9lib.norm(d["text"], p["start"], p["end"]) for p in preds) if b}
            tp += len(g & pb)
            fp += len(pb - g)
            fn += len(g - pb)
            for s, e, t in c9lib.gold(d, labels):
                if t == "email":
                    email_n += 1
                    email_hit += (s, e) in pb
            if n == 1 and a.arm in LLMS and not a.mock:
                timed["ps_during"] = ol.ps()
            if n % 50 == 0:
                print(f"{n}/{len(docs)} {secs:.2f}s {err or ''}", flush=True)
    timed["end"] = ol.now()
    if a.arm in LLMS and not a.mock:
        ol.unload(LLMS[a.arm])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    meta["summary"] = {"documents": len(docs), "span_precision": prec, "span_recall": rec,
                       "span_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                       "email_recall": email_hit / email_n if email_n else None, "email_gold": email_n,
                       "median_s": statistics.median(secs_all) if secs_all else None}
    if a.gate is not None:
        v = meta["summary"]["email_recall"]
        meta["gate"] = {"rule": "regex recall on the gold email spans", "minimum": a.gate, "value": v,
                        "passed": v is not None and v >= a.gate}
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(meta["summary"]))
    if meta.get("gate") and not meta["gate"]["passed"] and not a.limit:
        sys.exit(3)


if __name__ == "__main__":
    main()
