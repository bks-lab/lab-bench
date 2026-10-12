"""Route runner for queue jobs: T2 stability and T9 robustness (plan/t2.md, plan/t9.md).

  python bench/t2/run_route.py --arm winnow-12b --model hf.co/EldanRing/Winnow-12B:Q8_0 --template gemma4 \
      --lang de --passes plain,shuffled --out <dir>
  python bench/t2/run_route.py --arm gliner2.5-multi-decide --gliner fastino/GLiNER2.5-multi-Decide --lang de --passes plain --out <dir>
  python bench/t2/run_route.py ... --requests cases/t9/requests.jsonl                     (T9)
  python bench/t2/run_route.py ... --limit 250 --passes plain \
      --gate-ref results/route1/winnow-12b/2026-10-09-run1.jsonl --gate-min 0.99         (sanity arm, exit 3 below)

Local LLM arms: the route1 call of bench/run-local.mjs ported to Python
(bench/lib/route_local.py, checked against the Node code in CI): raw prompt
in the model's template, options as letters, temperature 0, one token, top
20 logprobs, num_ctx 16384, prefix as given (''). GLiNER Decide arms: the
route1 call of bench/run-gliner.py, mode prompt (label = option text,
instruction in the task's prompt field), one question per call.

Pass `plain` sends the options in definition order, pass `shuffled` in the
seeded order of bench/normalize.mjs withShuffledOptions with seed 1002 (the
seed run-local.mjs gives the shuffled run of `--runs 2`). Rows have the
run-local.mjs shape plus `pass`; options and probs are always listed in
definition order, so passes compare position by position.
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
import ollama as ol  # noqa: E402
import route_local as rl  # noqa: E402

SHUFFLE_SEED = 1002
NUM_CTX = 16384
RESERVED = ("[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]")


def gclean(s):
    """bench/run-gliner.py clean(): GLiNER's prompt markers may not appear in prompt strings."""
    s = s.replace("(", "[").replace(")", "]")
    for tok in RESERVED:
        s = s.replace(tok, tok[:-1] + " ]")
    return s


def load_requests(path, lang, limit):
    reqs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if lang:
        reqs = [r for r in reqs if r["lang"] == lang]
    return reqs[:limit] if limit else reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", help="Ollama tag (LLM arm)")
    ap.add_argument("--template", choices=rl.TEMPLATES)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--gliner", help="Hugging Face id of a GLiNER2 Decide model (GLiNER arm)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--requests", default="requests/route1/requests.jsonl")
    ap.add_argument("--lang", default="de")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--passes", default="plain,shuffled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-ref", help="committed run rows to agree with (plain pass)")
    ap.add_argument("--gate-min", type=float, default=0.99)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if bool(a.model) == bool(a.gliner):
        sys.exit("give exactly one of --model and --gliner")
    if a.model and not a.template:
        sys.exit("--template is required with --model")
    os.makedirs(a.out, exist_ok=True)
    reqs = load_requests(a.requests, a.lang, a.limit)
    passes = [p for p in a.passes.split(",") if p]
    meta = {"arm": a.arm, "job_id": os.environ.get("JMB_JOB_ID"), "requests": a.requests, "lang": a.lang,
            "limit": a.limit, "passes": passes, "shuffle_seed": SHUFFLE_SEED, "n_requests": len(reqs),
            "mock": a.mock, "sections": [], "started": ol.now()}
    import hashlib  # noqa: PLC0415
    meta["requests_sha256"] = hashlib.sha256(open(a.requests, "rb").read()).hexdigest()

    if a.model:
        info = None if a.mock else ol.model_info(a.model)
        version = None if a.mock else ol.version()
        meta.update({"model": a.model, "template": a.template, "prefix": a.prefix, "num_ctx": NUM_CTX, "model_info": info,
                     "runtime": f"ollama {version}", "system": rl.SYSTEM_ROUTE})
        digest = (info or {}).get("digest")

        def ask(state, q):
            opts = rl.lettered(q)
            if len(opts) > len(rl.LETTERS):
                return {"error": f"too many options ({len(opts)})"}
            if a.mock:
                return {"pByKey": {k: (1.0 if i == 0 else 0.0) for i, (k, _) in enumerate(opts)}, "mass": 1.0, "ms": 0}
            body = {"model": a.model, "raw": True, "stream": False, "prompt": rl.prompt(a.template, state, q, opts, a.prefix),
                    "logprobs": True, "top_logprobs": 20, "keep_alive": "30m",
                    "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX}}
            resp, secs, err = ol.call("/api/generate", body, timeout=600)
            if err:
                return {"error": err, "ms": round(secs * 1000)}
            top = ((resp.get("logprobs") or [{}])[0] or {}).get("top_logprobs") or []
            p, mass = rl.letter_probs(top, opts)
            return {"pByKey": p, "mass": mass, "ms": round(secs * 1000),
                    "compute_ms": round(resp["total_duration"] / 1e6) if resp.get("total_duration") is not None else None,
                    "tokens_in": resp.get("prompt_eval_count"), "first": resp.get("response")}
    else:
        import torch  # noqa: PLC0415
        from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier  # noqa: PLC0415
        from importlib.metadata import version as pkg_version  # noqa: PLC0415
        t = time.perf_counter()
        clf = None if a.mock else Classifier.from_pretrained(a.gliner)
        if clf is not None:
            clf.to(device=a.device)
            clf.eval()
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            rev = os.path.basename(snapshot_download(a.gliner, local_files_only=True))
        except Exception:  # noqa: BLE001
            rev = None
        cfg = ClassificationConfig()
        meta.update({"gliner": a.gliner, "revision": rev, "mode": "prompt", "load_s": time.perf_counter() - t,
                     "runtime": f"gliner2 {pkg_version('gliner2')}, torch {torch.__version__}, {a.device}"})
        digest = rev
        if clf is not None:
            clf.score("This is a test.", ClassificationSchema().single("warmup", {"yes": "Yes", "no": "No"}, instruction="Is this a test?"), config=cfg)

        def ask(state, q, qid="q"):
            labels = {gclean(v): gclean(v) for v in q["criteria"].values()}
            if a.mock:
                return {"pByKey": {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(q["criteria"])}, "mass": 1.0, "ms": 0}
            task = re.sub(r"[^A-Za-z0-9_]", "_", qid)
            schema = ClassificationSchema().single(task, labels, instruction=gclean(q["instructions"]))
            if a.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            scores = clf.score(rl.state_text(state), schema, config=cfg)
            if a.device == "cuda":
                torch.cuda.synchronize()
            ms = round((time.perf_counter() - t0) * 1000)
            p = {k: scores.probability(task, gclean(v)) for k, v in q["criteria"].items()}
            return {"pByKey": p, "mass": None, "ms": ms}

    agree = agree_n = 0
    ref = {}
    if a.gate_ref:
        for line in open(a.gate_ref, encoding="utf-8"):
            r = json.loads(line)
            ref[(r["line"], r["question"])] = r.get("choice")
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as fo:
        for pname in passes:
            sec = {"name": f"timed-{pname}", "start": ol.now()}
            n = errors = low = 0
            t0 = time.perf_counter()
            for req in reqs:
                request = req["request"] if pname == "plain" else rl.with_shuffled_options(req["request"], SHUFFLE_SEED)
                for qid, q in request["questions"].items():
                    try:
                        res = ask(request["state"], q) if a.model else ask(request["state"], q, qid)
                    except Exception as e:  # noqa: BLE001
                        res = {"error": f"{type(e).__name__}: {e}"[:300]}
                    options = rl.options_of(req["request"]["questions"][qid])
                    if a.model and not res.get("error") and res.get("mass") == 0:
                        res["error"] = "no option letter in the top logprobs"
                    probs = [rl.r4(res["pByKey"].get(o, 0.0)) for o in options] if not res.get("error") else None
                    choice = options[probs.index(max(probs))] if probs else None
                    if res.get("error"):
                        errors += 1
                    elif res.get("mass") is not None and res["mass"] < 0.5:
                        low += 1
                    n += 1
                    if pname == "plain" and (req["line"], qid) in ref:
                        agree_n += 1
                        agree += choice == ref[(req["line"], qid)]
                    row = {"pv": req.get("pv"), "unit": req.get("unit"), "line": req["line"], "question": qid, "qtype": q["type"],
                           "lang": req.get("lang"), "arm": a.arm, "arm_version": a.model or a.gliner, "arm_digest": digest,
                           "runtime": meta.get("runtime"), "host": "local-rtx4090", "pass": pname, "shuffled": pname != "plain",
                           "latency_ms": res.get("ms"), "compute_ms": res.get("compute_ms"), "tokens_in": res.get("tokens_in"),
                           "ts": ol.now(), "options": options, "probs": probs, "choice": choice,
                           "mass": rl.r4(res["mass"]) if res.get("mass") is not None else None,
                           "first_token": res.get("first"), "error": res.get("error")}
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if n % 500 == 0:
                        fo.flush()
                        print(f"{pname}: {n} questions, {round(time.perf_counter() - t0)} s", flush=True)
            sec.update({"end": ol.now(), "items": n, "errors": errors, "low_mass": low, "seconds": round(time.perf_counter() - t0, 1)})
            meta["sections"].append(sec)
            print(f"{pname}: {n} questions, {errors} errors, {low} with mass < 0.5, {sec['seconds']} s", flush=True)
    if a.model and not a.mock:
        ol.unload(a.model)
    if a.gate_ref:
        v = agree / agree_n if agree_n else None
        meta["gate"] = {"rule": f"plain-pass choices agree with {a.gate_ref}", "minimum": a.gate_min, "value": v,
                        "questions": agree_n, "passed": v is not None and v >= a.gate_min}
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({"sections": meta["sections"], "gate": meta.get("gate")}))
    if meta.get("gate") and not meta["gate"]["passed"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
