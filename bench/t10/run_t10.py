"""T10 runner: batched throughput of GLiNER Decide on route1 MASSIVE de (plan/t10.md).

  python bench/t10/run_t10.py --gliner fastino/GLiNER2.5-multi-Decide --arm gliner2.5-multi-decide --out <dir>
      [--batches 1,8,32,128] [--repeats 3] [--gate-ref results/route1/gliner2.5-multi-decide/2026-10-09-run1.jsonl]

The 5,948 questions of the 2,974 MASSIVE de requests (scenario, intent), in
file order, built exactly like bench/run-gliner.py in mode prompt. For every
batch size, `repeats` timed passes over all questions through gliner2's own
batched scorer (Classifier.scorer.batch_score with one compiled schema per
question, the path Classifier.score takes with batch size 1). Wall time per
pass includes tokenisation and a CUDA synchronise at the end. Choices of the
first pass of every batch size are kept: result.jsonl holds one row per
question with the choice per batch size, meta.json the timings. Gate (exit 3):
the batch 1 choices agree with the committed route1 run of the same arm on at
least --gate-min of the questions.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "t2"))
import ollama as ol  # noqa: E402  (now() only)
import route_local as rl  # noqa: E402
from run_route import gclean, load_requests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gliner", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--requests", default="requests/route1/requests.jsonl")
    ap.add_argument("--lang", default="de")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-ref")
    ap.add_argument("--gate-min", type=float, default=0.99)
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier  # noqa: PLC0415
    from importlib.metadata import version as pkg_version  # noqa: PLC0415
    os.makedirs(a.out, exist_ok=True)
    reqs = [r for r in load_requests(a.requests, a.lang, 0) if r["line"].startswith("massive-")]
    if a.limit:
        reqs = reqs[: a.limit]
    t0 = time.perf_counter()
    clf = Classifier.from_pretrained(a.gliner)
    clf.to(device=a.device)
    clf.eval()
    load_s = time.perf_counter() - t0
    cfg = ClassificationConfig()
    items = []  # (line, qid, text, compiled, task, label of key, options)
    for r in reqs:
        text = rl.state_text(r["request"]["state"])
        for qid, q in r["request"]["questions"].items():
            task = re.sub(r"[^A-Za-z0-9_]", "_", qid)
            labels = {gclean(v): gclean(v) for v in q["criteria"].values()}
            schema = ClassificationSchema().single(task, labels, instruction=gclean(q["instructions"]))
            items.append((r["line"], qid, text, clf.compile_schema(schema), task,
                          {k: gclean(v) for k, v in q["criteria"].items()}, rl.options_of(q)))
    clf.score("This is a test.", ClassificationSchema().single("warmup", {"yes": "Yes", "no": "No"}, instruction="Is this a test?"), config=cfg)
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        rev = os.path.basename(snapshot_download(a.gliner, local_files_only=True))
    except Exception:  # noqa: BLE001
        rev = None
    meta = {"test": "T10", "arm": a.arm, "gliner": a.gliner, "revision": rev, "job_id": os.environ.get("JMB_JOB_ID"),
            "runtime": f"gliner2 {pkg_version('gliner2')}, torch {torch.__version__}, {a.device}",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "questions": len(items), "load_s": load_s, "max_len": cfg.max_len, "passes": [], "started": ol.now()}
    choices = {}
    texts = [it[2] for it in items]
    comp = [it[3] for it in items]
    for b in [int(x) for x in a.batches.split(",")]:
        for rep in range(1, a.repeats + 1):
            # one untimed warm-up batch at this size, so allocator growth is not timed
            clf.scorer.batch_score(texts[:b], comp[:b], batch_size=b, max_len=cfg.max_len)
            if a.device == "cuda":
                torch.cuda.synchronize()
            s = {"batch": b, "repeat": rep, "start": ol.now()}
            t = time.perf_counter()
            with torch.inference_mode():
                scores = clf.scorer.batch_score(texts, comp, batch_size=b, max_len=cfg.max_len)
            if a.device == "cuda":
                torch.cuda.synchronize()
            secs = time.perf_counter() - t
            s.update(end=ol.now(), seconds=round(secs, 3), decisions_per_s=len(items) / secs,
                     peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20) if a.device == "cuda" else None)
            meta["passes"].append(s)
            print(f"batch {b} repeat {rep}: {secs:.1f} s, {len(items) / secs:.0f} decisions/s", flush=True)
            if rep == 1:
                ch = []
                for it, sc in zip(items, scores):
                    p = {k: sc.probability(it[4], lab) for k, lab in it[5].items()}
                    ch.append(max(it[6], key=lambda o: p[o]))
                choices[b] = ch
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for n, it in enumerate(items):
            fo.write(json.dumps({"line": it[0], "question": it[1], "choice": {str(b): c[n] for b, c in choices.items()}}, ensure_ascii=False) + "\n")
    base = choices.get(1)
    if base:
        meta["agree_with_batch1"] = {str(b): sum(x == y for x, y in zip(base, c)) / len(c) for b, c in choices.items()}
    meta["summary"] = {str(b): {"decisions_per_s_mean": statistics.mean(p["decisions_per_s"] for p in meta["passes"] if p["batch"] == b),
                                "min": min(p["decisions_per_s"] for p in meta["passes"] if p["batch"] == b),
                                "max": max(p["decisions_per_s"] for p in meta["passes"] if p["batch"] == b)}
                       for b in choices}
    if a.gate_ref and base:
        ref = {}
        for line in open(a.gate_ref, encoding="utf-8"):
            r = json.loads(line)
            ref[(r["line"], r["question"])] = r.get("choice")
        pairs = [(c, ref[(it[0], it[1])]) for it, c in zip(items, base) if (it[0], it[1]) in ref]
        v = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
        meta["gate"] = {"rule": f"batch 1 choices agree with {a.gate_ref}", "minimum": a.gate_min, "value": v,
                        "questions": len(pairs), "passed": v is not None and v >= a.gate_min}
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({"summary": meta["summary"], "agree": meta.get("agree_with_batch1"), "gate": meta.get("gate")}))
    if meta.get("gate") and not meta["gate"]["passed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
