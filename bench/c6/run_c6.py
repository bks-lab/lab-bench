"""C6 runner: German reading comprehension (plan/c6.md, results/c6/method.md).

  python bench/c6/run_c6.py --arm gelectra-large --data D:\\bench-data\\c6 --out <dir> --gate 88.1 --tol 2
  python bench/c6/run_c6.py --arm qwen3-8b --out <dir> [--limit 20]
  python bench/c6/run_c6.py --arm qwen3-8b --out /tmp/x --mock

The reader arm answers all 2,204 test questions (D:\\bench-data\\c6\\test.parquet,
hash-checked) and checks the sanity gate: F1 over all 2,204 within --tol of
--gate, else exit 3. LLM arms answer the 1,000 questions of cases/c6/sample.jsonl.
Writes <out>/result.jsonl and <out>/meta.json; F1 and EM per row are written
for the log, bench/score-cap.py recomputes them.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import ollama as ol  # noqa: E402
import squad  # noqa: E402

PARQUET_SHA = "c154513e016e73e4e32b46b48010f24a435ad7b0e96cc3858cfc20c7d29649bc"
READER = {"model": "deepset/gelectra-large-germanquad", "rev": "c16378c846ebc75aa971101acd0192535fed9fed",
          "max_answer_len": 30}
NUM_CTX = 8192
OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX, "num_predict": 256}
FORMAT = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
KEEP = "30m"
LLMS = {
    "qwen3-8b": "qwen3:8b",
    "qwen3.8-27b": "qwen3.8:27b",
    "gemma4-12b": "gemma4:12b",
    "gemma4-26b": "gemma4:26b",
    "mistral-small-3.2": "mistral-small3.2:24b",
    "winnow-12b": "hf.co/EldanRing/Winnow-12B:Q8_0",
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_prompt(repo):
    s = open(os.path.join(repo, "results", "c6", "method.md"), encoding="utf-8").read()
    return re.search(r"## Prompt\s+```text\n(.*?)\n```", s, re.S).group(1)


def load_all(data):
    p = os.path.join(data, "test.parquet")
    if sha(p) != PARQUET_SHA:
        sys.exit("test.parquet does not match plan/c6.md")
    import pyarrow.parquet as pq  # noqa: PLC0415
    rows = []
    for r in pq.read_table(p).to_pylist():
        rows.append({"id": str(r["id"]), "context": r["context"], "question": r["question"],
                     "answers": [str(x) for x in r["answers"]["text"]]})
    rows.sort(key=lambda r: int(r["id"]))
    return rows


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
    ap.add_argument("--arm", required=True, choices=["gelectra-large"] + sorted(LLMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.repo, "cases", "c6", "manifest.json"), encoding="utf-8"))
    sp = os.path.join(a.repo, "cases", "c6", "sample.jsonl")
    if sha(sp) != man["sample"]["sha256"]:
        sys.exit("cases/c6/sample.jsonl does not match its manifest")
    sample_ids = set(man["sample"]["ids"])
    meta = {"test": "C6", "arm": a.arm, "job_id": os.environ.get("JMB_JOB_ID"), "mock": a.mock, "limit": a.limit,
            "inputs": {"sample_sha256": man["sample"]["sha256"],
                       "method_sha256": sha(os.path.join(a.repo, "results", "c6", "method.md"))},
            "sections": [], "started": ol.now()}
    reader = a.arm == "gelectra-large"
    if reader:
        rows = load_all(a.data)
        meta["inputs"]["parquet_sha256"] = PARQUET_SHA
    else:
        rows = [json.loads(l) for l in open(sp, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    if reader:
        import torch  # noqa: PLC0415
        from transformers import pipeline  # noqa: PLC0415
        meta["versions"] = versions(["transformers", "torch", "tokenizers"])
        meta["versions"]["cuda_available"] = torch.cuda.is_available()
        meta["reader"] = READER
        t0 = time.perf_counter()
        load = {"name": "load", "start": ol.now()}
        qa = None if a.mock else pipeline("question-answering", model=READER["model"], revision=READER["rev"],
                                          tokenizer=READER["model"], device=0 if torch.cuda.is_available() else -1)
        load["end"] = ol.now()
        load["seconds"] = time.perf_counter() - t0
        meta["sections"].append(load)

        def answer(r):
            if a.mock:
                return {"answer": r["answers"][0]}, 0.0, None
            t = time.perf_counter()
            out = qa(question=r["question"], context=r["context"], max_answer_len=READER["max_answer_len"])
            if a.mock is False and torch.cuda.is_available():
                torch.cuda.synchronize()
            return {"answer": out["answer"], "score": float(out["score"])}, time.perf_counter() - t, None
    else:
        tag = LLMS[a.arm]
        prompt = load_prompt(a.repo)
        info = None if a.mock else ol.model_info(tag)
        think = None if a.mock else ol.think_setting(info, False)
        meta.update({"ollama_tag": tag, "model": info, "options": OPTIONS, "format": FORMAT,
                     "think": think if think is not None else "not supported by the model",
                     "versions": {"python": sys.version.split()[0], "ollama": None if a.mock else ol.version()}})

        def answer(r):
            if a.mock:
                return {"answer": r["answers"][0]}, 0.0, None
            body = {"model": tag, "messages": [{"role": "user", "content": prompt.replace("{context}", r["context"]).replace("{question}", r["question"])}],
                    "format": FORMAT, "options": OPTIONS, "stream": False, "keep_alive": KEEP}
            if think is not None:
                body["think"] = think
            resp, secs, err = ol.call("/api/chat", body, timeout=600)
            raw = ((resp or {}).get("message") or {}).get("content")
            try:
                ans = json.loads(raw)["answer"]
                ans = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
                perr = None
            except Exception:  # noqa: BLE001
                ans, perr = "", "answer did not parse"
            return {"answer": ans, "raw": raw, **ol.counters(resp)}, secs, err or perr

    warm = {"name": "warmup", "start": ol.now()}
    for r in rows[: a.warmup]:
        answer(r)
    warm["end"] = ol.now()
    meta["sections"].append(warm)
    timed = {"name": "timed", "items": len(rows), "start": ol.now()}
    if not reader and not a.mock:
        timed["ps_before"] = ol.ps()
    out_rows = []
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for n, r in enumerate(rows):
            res, secs, err = answer(r)
            ans = res["answer"]
            row = {"id": r["id"], "arm": a.arm, "in_sample": r["id"] in sample_ids, **res, "seconds": round(secs, 4),
                   "error": err, "f1": squad.best(squad.f1, ans, r["answers"]), "em": squad.best(squad.em, ans, r["answers"])}
            out_rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if n == 1 and not reader and not a.mock:
                timed["ps_during"] = ol.ps()
            if n % 100 == 0:
                print(f"{n}/{len(rows)} f1={row['f1']:.2f} {secs:.2f}s {err or ''}", flush=True)
    timed["end"] = ol.now()
    meta["sections"].append(timed)
    if not reader and not a.mock:
        ol.unload(LLMS[a.arm])
    f1_all = 100 * statistics.mean(x["f1"] for x in out_rows)
    em_all = 100 * statistics.mean(x["em"] for x in out_rows)
    meta["summary"] = {"questions": len(out_rows), "f1": f1_all, "em": em_all,
                       "errors": sum(1 for x in out_rows if x["error"]),
                       "median_s": statistics.median(x["seconds"] for x in out_rows)}
    if reader and a.gate is not None:
        meta["gate"] = {"reference_f1": a.gate, "tolerance": a.tol, "value": f1_all, "questions": len(out_rows),
                        "passed": abs(f1_all - a.gate) <= a.tol}
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(meta["summary"]))
    if meta.get("gate") and not meta["gate"]["passed"] and not a.limit:
        sys.exit(3)


if __name__ == "__main__":
    main()
