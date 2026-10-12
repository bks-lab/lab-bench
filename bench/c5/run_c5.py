"""C5 runner: translation German to English and English to German (plan/c5.md, notes of 2026-10-12).

  python bench/c5/run_c5.py --arm opus-mt --data D:\\bench-data\\c5 --out <dir>      dedicated MT, transformers
  python bench/c5/run_c5.py --arm madlad-3b --data D:\\bench-data\\c5 --out <dir>
  python bench/c5/run_c5.py --arm qwen3.8-27b --data D:\\bench-data\\c5 --out <dir>   Ollama, stdlib only
  ... [--limit N] [--mock]

Reads the 1,012 FLORES+ devtest pairs (hash-checked against
cases/c5/manifest.json, every sentence) and translates both directions, one
sentence per call at batch 1 (2,024 segments). Dedicated MT arms then run a
second, untimed-for-quality pass at batch 32 for throughput only. Writes
<out>/result.jsonl (id, direction, hypothesis, seconds) and <out>/meta.json.
The texts and translations stay on the machine and in .cache/; nothing of
them is committed (plan/c5.md, Data). Scoring: bench/c5/gate.py (sanity
gate, chrF of opus-mt), bench/c5/comet_score.py (COMET on the GPU after all arms),
bench/c5/score_c5.py (chrF++, BLEU, COMET, bootstrap).
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import ollama as ol  # noqa: E402

DIRS = ("de-en", "en-de")
SRC = {"de-en": "deu_Latn", "en-de": "eng_Latn"}
TGT = {"de-en": "eng_Latn", "en-de": "deu_Latn"}
MT = {
    "opus-mt": {"de-en": ("Helsinki-NLP/opus-mt-de-en", "1a922f3b32a8e809e17a47d4b32142d8105924e5"),
                "en-de": ("Helsinki-NLP/opus-mt-en-de", "6183067f769a302e3861815543b9f312c71b0ca4")},
    "madlad-3b": {"de-en": ("google/madlad400-3b-mt", "fa184c675da0b5c9e1c8694fccd4e12e2d422094"),
                  "en-de": ("google/madlad400-3b-mt", "fa184c675da0b5c9e1c8694fccd4e12e2d422094")},
}
MADLAD_PREFIX = {"de-en": "<2en> ", "en-de": "<2de> "}
LLMS = {
    "qwen3.8-27b": "qwen3.8:27b",
    "gemma4-12b": "gemma4:12b",
    "gemma4-26b": "gemma4:26b",
    "mistral-small-3.2": "mistral-small3.2:24b",
    "eurollm-9b": "hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q8_0",
}
INSTRUCTION = {
    "de-en": "Translate the following German text into English. Output only the translation.",
    "en-de": "Translate the following English text into German. Output only the translation.",
}
OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": 2048, "num_predict": 512}
MAX_NEW = 512
BATCH = 32


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load(data, repo, limit):
    man = json.load(open(os.path.join(repo, "cases", "c5", "manifest.json"), encoding="utf-8"))
    text = {}
    for lang in ("deu_Latn", "eng_Latn"):
        p = os.path.join(data, f"{lang}.jsonl")
        raw = open(p, "rb").read()
        if sha(raw) != man["files"][lang]["sha256"]:
            sys.exit(f"{p} does not match cases/c5/manifest.json")
        rows = {int(r["id"]): r["text"] for r in (json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip())}
        text[lang] = [rows[i] for i in man["ids"]]
        bad = [i for i, (t, h) in enumerate(zip(text[lang], man["sentences"][lang])) if sha(t.encode("utf-8")) != h]
        if bad:
            sys.exit(f"{lang}: {len(bad)} sentences differ from the manifest")
    ids = man["ids"][:limit] if limit else man["ids"]
    return man, ids, {k: v[: len(ids)] for k, v in text.items()}


def versions(names):
    from importlib.metadata import version  # noqa: PLC0415
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:  # noqa: BLE001
            out[n] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(MT) + sorted(LLMS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man, ids, text = load(a.data, a.repo, a.limit)
    meta = {"test": "C5", "arm": a.arm, "job_id": os.environ.get("JMB_JOB_ID"), "mock": a.mock, "limit": a.limit,
            "inputs": {"manifest_sha256": sha(open(os.path.join(a.repo, "cases", "c5", "manifest.json"), "rb").read()),
                       "files": man["files"]}, "sections": [], "started": ol.now()}
    rows = []

    if a.arm in MT:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
        meta["versions"] = versions(["torch", "transformers", "sentencepiece"])
        meta["versions"]["cuda_available"] = torch.cuda.is_available()
        dtype = torch.float16 if a.arm == "madlad-3b" else torch.float32
        meta["mt"] = {d: {"model": m, "rev": r} for d, (m, r) in MT[a.arm].items()}
        meta["mt_settings"] = {"generation": "the model's generation_config (recorded below)", "max_new_tokens": MAX_NEW,
                               "dtype": str(dtype), "batch_timed": 1, "batch_throughput": BATCH}
        cache = {}

        def model_for(d):
            m, r = MT[a.arm][d]
            if (m, r) not in cache:
                t0 = time.perf_counter()
                s = {"name": f"load-{m}", "start": ol.now()}
                tok = AutoTokenizer.from_pretrained(m, revision=r)
                mod = None if a.mock else AutoModelForSeq2SeqLM.from_pretrained(m, revision=r, torch_dtype=dtype).to(a.device).eval()
                s.update(end=ol.now(), seconds=round(time.perf_counter() - t0, 2))
                meta["sections"].append(s)
                meta.setdefault("generation_config", {})[m] = mod.generation_config.to_dict() if mod is not None else None
                cache[(m, r)] = (tok, mod)
            return cache[(m, r)]

        def translate(d, srcs):
            tok, mod = model_for(d)
            if a.arm == "madlad-3b":
                srcs = [MADLAD_PREFIX[d] + s for s in srcs]
            if a.mock:
                return list(srcs)
            enc = tok(srcs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(a.device)
            with torch.inference_mode():
                out = mod.generate(**enc, max_new_tokens=MAX_NEW)
            if a.device == "cuda":
                torch.cuda.synchronize()
            return tok.batch_decode(out, skip_special_tokens=True)

        for d in DIRS:
            srcs = text[SRC[d]]
            model_for(d)
            w = {"name": f"warmup-{d}", "start": ol.now()}
            for s in srcs[: a.warmup]:
                translate(d, [s])
            w["end"] = ol.now()
            meta["sections"].append(w)
            sec = {"name": f"timed-{d}", "start": ol.now(), "items": len(srcs), "batch": 1}
            for i, s in zip(ids, srcs):
                t0 = time.perf_counter()
                try:
                    hyp, err = translate(d, [s])[0], None
                except Exception as e:  # noqa: BLE001
                    hyp, err = "", f"{type(e).__name__}: {e}"[:300]
                rows.append({"id": i, "dir": d, "hyp": hyp, "seconds": round(time.perf_counter() - t0, 4), "error": err})
            sec["end"] = ol.now()
            meta["sections"].append(sec)
            # throughput at batch 32: hypotheses kept only to count how many differ from batch 1
            sec = {"name": f"batch{BATCH}-{d}", "start": ol.now(), "items": len(srcs), "batch": BATCH}
            t0 = time.perf_counter()
            hyps = []
            for k in range(0, len(srcs), BATCH):
                hyps += translate(d, srcs[k:k + BATCH])
            sec["seconds"] = round(time.perf_counter() - t0, 3)
            sec["segments_per_s"] = len(srcs) / sec["seconds"] if sec["seconds"] else None
            b1 = [r["hyp"] for r in rows if r["dir"] == d]
            sec["differs_from_batch1"] = sum(x != y for x, y in zip(b1, hyps))
            sec["end"] = ol.now()
            meta["sections"].append(sec)
            print(f"{d}: done, batch {BATCH} {sec['segments_per_s']} seg/s, {sec['differs_from_batch1']} differ", flush=True)
    else:
        tag = LLMS[a.arm]
        info = None if a.mock else ol.model_info(tag)
        think = None if a.mock else ol.think_setting(info, False)
        meta.update({"ollama_tag": tag, "model": info, "options": OPTIONS, "instruction": INSTRUCTION,
                     "think": think if think is not None else "not supported by the model",
                     "versions": {"python": sys.version.split()[0], "ollama": None if a.mock else ol.version()}})

        def translate(d, s):
            if a.mock:
                return s, None, {}
            body = {"model": tag, "messages": [{"role": "user", "content": f"{INSTRUCTION[d]}\n\n{s}"}],
                    "options": OPTIONS, "stream": False, "keep_alive": "30m"}
            if think is not None:
                body["think"] = think
            resp, _, err = ol.call("/api/chat", body, timeout=600)
            hyp = (((resp or {}).get("message") or {}).get("content") or "").strip()
            return hyp, err, ol.counters(resp)

        w = {"name": "warmup", "start": ol.now()}
        for s in text["deu_Latn"][: a.warmup]:
            translate("de-en", s)
        w["end"] = ol.now()
        meta["sections"].append(w)
        for d in DIRS:
            srcs = text[SRC[d]]
            sec = {"name": f"timed-{d}", "start": ol.now(), "items": len(srcs), "batch": 1}
            if not a.mock:
                sec["ps_before"] = ol.ps()
            for n, (i, s) in enumerate(zip(ids, srcs)):
                t0 = time.perf_counter()
                hyp, err, cnt = translate(d, s)
                rows.append({"id": i, "dir": d, "hyp": hyp, "seconds": round(time.perf_counter() - t0, 4), "error": err, **cnt})
                if n % 200 == 0:
                    print(f"{d} {n}/{len(srcs)} {rows[-1]['seconds']}s {err or ''}", flush=True)
            sec["end"] = ol.now()
            meta["sections"].append(sec)
        if not a.mock:
            ol.unload(tag)

    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta["summary"] = {"segments": len(rows), "errors": sum(1 for r in rows if r["error"]),
                       "empty": sum(1 for r in rows if not r["hyp"]),
                       "median_s": {d: statistics.median([r["seconds"] for r in rows if r["dir"] == d]) for d in DIRS}}
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(meta["summary"]))


if __name__ == "__main__":
    main()
