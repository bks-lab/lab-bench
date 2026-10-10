"""C8 runner: throughput and energy per model size (plan/c8.md, generator bench/gen-c8.py).

  python bench/c8/run_c8.py --arm qwen3-14b --data D:\\bench-data\\c6 --out <dir>
        [--levels 1,4,8,16] [--reps 3] [--workloads chat-short,chat-long,decide] [--limit N] [--mock]

Per concurrency level a private `ollama serve` (port 11436) starts with
OLLAMA_NUM_PARALLEL = the level and OLLAMA_MAX_LOADED_MODELS=1, so the
workstation's Ollama service is never restarted (the queue gate has made sure
it holds no model). Per level, workload and repetition: warm-up requests,
then the timed requests from `level` client threads, each taking the next
request as soon as its last one returned (constant concurrency). Chat
requests stream, so the client sees the first token; decide requests are
the one-token raw prompts of bench/run-local.mjs.

Sanity gate (plan/c8.md): the three repetitions of chat-short at level 1
must agree within 5 % (coefficient of variation of the aggregate generation
tokens per second), else exit 3. Before every configuration the GPU is
checked for foreign load (utilisation with our model idle, and the Ollama
service empty); foreign load that does not clear within 10 minutes ends the
run with exit 4.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BENCH, "lib"))
import ollama as ol  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_c8", os.path.join(BENCH, "gen-c8.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# arm -> (Ollama tag, think setting, raw template of the decide workload)
ARMS = {
    "granite4-3b": ("granite4:micro", False, "granite"),
    "qwen3-8b": ("qwen3:8b", False, "chatml"),
    "gemma4-12b": ("gemma4:12b", False, "gemma4"),
    "winnow-12b": ("hf.co/EldanRing/Winnow-12B:Q8_0", False, "gemma4"),
    "qwen3-14b": ("qwen3:14b", False, "chatml"),
    "gpt-oss-20b": ("gpt-oss:20b", "low", "harmony"),
    "mistral-small-3.2": ("mistral-small3.2:24b", False, "mistral"),
    "gemma4-26b": ("gemma4:26b", False, "gemma4"),
    "qwen3.8-27b": ("qwen3.8:27b", False, "chatml"),
}
NUM_CTX = 5120          # chat-long: 4,096 prompt (single prompts up to about 4,400) + 256 generated, per parallel slot
NUM_PREDICT = 256
PORT = 11436
CV_MAX = 0.05


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gpu_now():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.gr",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    try:
        u, m, p, t, c = [float(x) for x in out.splitlines()[0].split(",")]
        return {"util": u, "mem_mib": m, "power_w": p, "temp_c": t, "clock_mhz": c}
    except Exception:  # noqa: BLE001
        return {"raw": out}


def idle_check(main_url):
    """Foreign load: GPU busy while our model sits idle, or a model in the workstation's service."""
    t_end = time.time() + 600
    tries = []
    while True:
        samples = [gpu_now() for _ in range(3) if not time.sleep(1)]
        svc = ol.ps(main_url)
        busy = all(s.get("util", 0) > 30 for s in samples)
        tries.append({"t": ol.now(), "gpu": samples[-1], "service_models": [m.get("name") for m in svc]})
        if not busy and not any(m.get("name") for m in svc):
            return {"idle": True, "checks": tries[-3:]}
        if time.time() > t_end:
            return {"idle": False, "checks": tries[-5:]}
        time.sleep(30)


def stream_chat(url, body, timeout=900):
    """Streams one chat request; returns (first token seconds, last chunk, wall seconds, error)."""
    data = json.dumps({**body, "stream": True}).encode()
    req = urllib.request.Request(url + "/api/chat", data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    last = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                msg = d.get("message") or {}
                if first is None and (msg.get("content") or msg.get("thinking")):
                    first = time.perf_counter() - t0
                if d.get("done"):
                    last = d
        return first, last, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        return first, last, time.perf_counter() - t0, repr(e)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workloads", default="chat-short,chat-long,decide")
    ap.add_argument("--limit", type=int, default=0, help="timed requests per configuration (probe)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    tag, think, template = ARMS[a.arm]
    levels = [int(x) for x in a.levels.split(",")]
    workloads = a.workloads.split(",")
    os.makedirs(a.out, exist_ok=True)
    cdir = os.path.join(a.repo, "cases", "c8")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    for f, h in man["files_sha256"].items():
        if sha(open(os.path.join(cdir, f), "rb").read()) != h:
            sys.exit(f"cases/c8/{f} does not match its manifest")
    chat = [json.loads(l) for l in open(os.path.join(cdir, "chat.jsonl"), encoding="utf-8")]
    decide = [json.loads(l) for l in open(os.path.join(cdir, "decide.jsonl"), encoding="utf-8")]
    pool = gen.passages(os.path.join(a.data, "test.parquet"))
    main_url = ol.DEFAULT_URL
    meta = {"test": "C8", "arm": a.arm, "ollama_tag": tag, "template_decide": template, "levels": levels, "reps": a.reps,
            "workloads": workloads, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "limit": a.limit, "mock": a.mock,
            "job_id": os.environ.get("JMB_JOB_ID"), "inputs": man["files_sha256"], "calibration": {},
            "levels_detail": [], "sections": [], "started": ol.now()}
    fo = open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8")
    lock = threading.Lock()
    chars = {}
    think_field = None
    exit_code = 0

    def write(row):
        with lock:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()

    def chat_body(text):
        b = {"model": tag, "messages": [{"role": "user", "content": gen.INSTRUCTION.format(text=text)}], "keep_alive": "30m",
             "options": {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
        if think_field is not None:
            b["think"] = think_field
        return b

    def one_chat(url, item, w, level, rep):
        text = gen.chat_text(item["order"], pool, chars[w])
        if a.mock:
            first, last, wall, err = 0.01, {"prompt_eval_count": len(text) // 3, "eval_count": NUM_PREDICT,
                                            "prompt_eval_duration": 1e7, "eval_duration": 1e9}, 0.02, None
        else:
            first, last, wall, err = stream_chat(url, chat_body(text))
        c = ol.counters(last)
        return {"workload": w, "level": level, "rep": rep, "idx": item["idx"], "warmup": item["warmup"],
                "ttft_s": first, "wall_s": round(wall, 4), "error": err, **c}

    def one_decide(url, item, level, rep):
        body = {"model": tag, "raw": True, "stream": False, "prompt": item["prompts"][template], "logprobs": True,
                "top_logprobs": 20, "keep_alive": "30m", "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX}}
        if a.mock:
            r, wall, err = {"prompt_eval_count": 200, "eval_count": 1, "response": "A"}, 0.01, None
        else:
            r, wall, err = ol.call("/api/generate", body, timeout=300, url=url)
        return {"workload": "decide", "level": level, "rep": rep, "idx": item["line"], "warmup": item["warmup"],
                "wall_s": round(wall, 4), "error": err, "first_token": (r or {}).get("response"), **ol.counters(r)}

    def calibrate(url):
        """chars of passage text that make 512 and 4,096 prompt tokens with this model's template and tokenizer"""
        out = {}
        for w, target in gen.CHAT.items():
            item = [c for c in chat if c["workload"] == w][0]
            cal = []
            p0 = None
            if not a.mock:
                r, _, err = ol.call("/api/chat", {**chat_body(""), "stream": False,
                                                   "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 1}}, url=url)
                p0 = (r or {}).get("prompt_eval_count") or 0
            else:
                p0 = 40
            n_chars = int((target - p0) * 3.0)
            per = 3.0
            for _ in range(3):
                text = gen.chat_text(item["order"], pool, n_chars)
                if a.mock:
                    n = p0 + len(text) // 3
                else:
                    r, _, err = ol.call("/api/chat", {**chat_body(text), "stream": False,
                                                       "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 1}}, url=url)
                    n = (r or {}).get("prompt_eval_count")
                    if not n:
                        raise SystemExit(f"calibration failed for {w}: {err}")
                cal.append({"chars": n_chars, "tokens": n})
                per = n_chars / max(1, n - p0)
                n_chars = int((target - p0) * per)
            out[w] = {"target_tokens": target, "overhead_tokens": p0, "chars_per_token": per, "chars": n_chars, "calls": cal}
            chars[w] = n_chars
        return out

    try:
        for level in levels:
            # LLAMA_ARG_CACHE_RAM=0: the llama-server under Ollama 0.35 keeps up to 8 GiB of earlier
            # prompts in host memory and restores them; the second and third repetition would then
            # skip prompt processing that the first did (dry run of 2026-10-10). Off, so repetitions are alike.
            env = {"OLLAMA_NUM_PARALLEL": level, "OLLAMA_MAX_LOADED_MODELS": 1, "LLAMA_ARG_CACHE_RAM": 0,
                   "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", r"D:\ollama\models")}
            lv = {"level": level, "env": {k: str(v) for k, v in env.items() if k != "OLLAMA_MODELS"}, "start": ol.now()}
            srv = None
            if not a.mock:
                srv = ol.PrivateServer(PORT, env, os.path.join(a.out, f"ollama-private-{level}.log")).__enter__()
            url = srv.url if srv else "mock"
            try:
                if not a.mock:
                    info = ol.model_info(tag, url=url)
                    think_field = ol.think_setting(info, think)
                    meta.setdefault("model", info)
                    meta["think"] = think_field if think_field is not None else "not supported by the model"
                    meta.setdefault("versions", {"python": sys.version.split()[0], "ollama": ol.version(url)})
                    t0 = time.perf_counter()
                    ol.call("/api/generate", {"model": tag, "prompt": "", "keep_alive": "30m",
                                              "options": {"num_ctx": NUM_CTX}}, url=url)
                    lv["load_s"] = round(time.perf_counter() - t0, 2)
                    lv["layers_on_gpu"] = ol.layers_on_gpu(os.path.join(a.out, f"ollama-private-{level}.log"))
                if not chars:
                    meta["calibration"] = calibrate(url)
                lv["ps"] = ol.ps(url) if not a.mock else []
                for w in workloads:
                    items = [c for c in chat if c["workload"] == w] if w != "decide" else decide
                    warm = [x for x in items if x["warmup"]]
                    timed = [x for x in items if not x["warmup"]]
                    if a.limit:
                        timed = timed[: a.limit]
                    for rep in range(1, a.reps + 1):
                        chk = {"idle": True} if a.mock else idle_check(main_url)
                        if not chk["idle"]:
                            meta["stopped"] = {"reason": "foreign GPU load did not clear within 10 minutes", "level": level,
                                               "workload": w, "rep": rep, "checks": chk["checks"]}
                            exit_code = 4
                            raise StopIteration
                        fn = (lambda x, lv_=level, r_=rep, w_=w: one_chat(url, x, w_, lv_, r_)) if w != "decide" else \
                             (lambda x, lv_=level, r_=rep: one_decide(url, x, lv_, r_))
                        with ThreadPoolExecutor(max_workers=level) as ex:
                            for row in ex.map(fn, warm):
                                write(row)
                        sec = {"name": f"{w}@{level}#{rep}", "workload": w, "level": level, "rep": rep, "requests": len(timed),
                               "gpu_before": chk.get("checks", [{}])[-1].get("gpu") if not a.mock else None, "start": ol.now()}
                        t0 = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=level) as ex:
                            rows = list(ex.map(fn, timed))
                        wall = time.perf_counter() - t0
                        sec["end"] = ol.now()
                        for row in rows:
                            write(row)
                        gen_tok = sum(r["eval_count"] or 0 for r in rows if not r["error"])
                        sec.update({"wall_s": round(wall, 3), "errors": sum(1 for r in rows if r["error"]),
                                    "generated_tokens": gen_tok, "prompt_tokens": sum(r["prompt_eval_count"] or 0 for r in rows),
                                    "agg_gen_tok_s": gen_tok / wall if wall else None,
                                    "requests_per_h": len(rows) / wall * 3600 if wall else None})
                        meta["sections"].append(sec)
                        print(f"{sec['name']}: {len(rows)} req, {wall:.1f}s, {sec['agg_gen_tok_s']:.1f} tok/s agg, "
                              f"{sec['errors']} errors", flush=True)
                    if w == "chat-short" and level == 1:
                        v = [s["agg_gen_tok_s"] for s in meta["sections"] if s["workload"] == "chat-short" and s["level"] == 1]
                        cv = statistics.stdev(v) / statistics.mean(v) if len(v) > 1 and statistics.mean(v) else None
                        meta["gate"] = {"rule": "chat-short at concurrency 1, three repetitions, CV of aggregate generation tokens/s <= 5 %",
                                        "values": v, "cv": cv, "passed": cv is not None and cv <= CV_MAX}
                        print("gate", json.dumps(meta["gate"]), flush=True)
                        if not meta["gate"]["passed"] and not a.limit:
                            exit_code = 3
                            raise StopIteration
            finally:
                lv["end"] = ol.now()
                meta["levels_detail"].append(lv)
                if srv:
                    ol.unload(tag, url=url)
                    srv.__exit__(None, None, None)
    except StopIteration:
        pass
    finally:
        fo.close()
        meta["finished"] = ol.now()
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
