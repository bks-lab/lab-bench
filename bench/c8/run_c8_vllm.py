"""C8 vLLM arm (plan/c8.md): the same workloads as bench/c8/run_c8.py through vLLM
in the workstation's WSL2 Ubuntu, for `gemma4-12b` and `qwen3-14b` with a
4-bit AWQ build (named per arm). Rows and sections have the same fields as
the Ollama runner, so bench/score-cap.py scores both alike.

  python bench/c8/run_c8_vllm.py --arm qwen3-14b --data D:\\bench-data\\c6 --out <dir> [--levels ...] [--limit N]

Per level `vllm serve` starts in WSL (`wsl -d Ubuntu-24.04`, venv
/root/vllm-venv2 on a uv-managed CPython, Linux-only PATH, offline, inside `timeout` so it can never outlive the job)
with --max-num-seqs = the level and --max-model-len 4608, and is stopped
after the level. Chat requests go to /v1/chat/completions (streamed, usage
included, thinking off through the chat template's `enable_thinking`),
decide requests to /v1/completions with the same raw prompt as the Ollama arm
(max_tokens 1, top 20 logprobs). vLLM reports no prompt and generation
durations per request: generation time is the client's time from the first
to the last streamed token.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import ollama as ol  # noqa: E402
import run_c8 as base  # noqa: E402

gen = base.gen
DISTRO = "Ubuntu-24.04"
VENV = "/root/vllm-venv2"
# Linux-only PATH: the Windows PATH that WSL appends carries a non-executable nvcc that torch's
# inductor trips over, and the venv's Python is a uv-managed CPython with headers (triton builds a launcher).
LINUX_PATH = f"{VENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"
# arm -> (repo, revision, quantisation, decide template)
ARMS = {
    "qwen3-14b": ("Qwen/Qwen3-14B-AWQ", "31c69efc29464b6bb0aee1398b5a7b50a99340c3", "AWQ INT4 (official Qwen build)", "chatml"),
    "gemma4-12b": ("cyankiwi/gemma-4-12B-it-AWQ-INT4", "ad2d7e521f3eaee492973ab130025e610b216d2c", "AWQ INT4 (community build of google/gemma-4-12B-it)", "gemma4"),
}


def wsl(cmd, **kw):
    return subprocess.run(["wsl", "-d", DISTRO, "--", "bash", "-lc", cmd], capture_output=True, text=True, **kw)


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class VllmServer:
    def __init__(self, repo, rev, level, log_path):
        self.repo, self.rev, self.level, self.log_path = repo, rev, level, log_path

    def __enter__(self):
        wsl("pkill -f 'vllm serve' || true")
        # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer builds its sampling kernel with nvcc at start-up,
        # and the WSL Ubuntu has no CUDA toolkit; vLLM's PyTorch sampler takes its place (temperature 0 is greedy either way)
        cmd = (f"export PATH={LINUX_PATH}; export VLLM_USE_FLASHINFER_SAMPLER=0; HF_HUB_OFFLINE=1 timeout 4h {VENV}/bin/vllm serve {self.repo} --revision {self.rev} "
               f"--served-model-name m --max-num-seqs {self.level} --max-model-len {base.NUM_CTX} "
               f"--gpu-memory-utilization 0.85 --seed 1 --port {PORT} --host 127.0.0.1")
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.log.write(f"# {ol.now()} {cmd}\n")
        self.log.flush()
        self.proc = subprocess.Popen(["wsl", "-d", DISTRO, "--", "bash", "-lc", cmd], stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=0x08000000 if os.name == "nt" else 0)
        t0 = time.time()
        while time.time() - t0 < 1200:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}, see {self.log_path}")
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
                self.startup_s = time.time() - t0
                return self
            except Exception:  # noqa: BLE001
                time.sleep(3)
        raise RuntimeError("vllm serve did not answer within 20 minutes")

    def __exit__(self, *exc):
        wsl("pkill -f 'vllm serve' || true")
        try:
            self.proc.wait(60)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.log.write(f"# {ol.now()} stopped\n")
        self.log.close()
        return False


def stream_chat(body, timeout=900):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last_tok = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        t = time.perf_counter() - t0
                        first = first if first is not None else t
                        last_tok = t
        return first, last_tok, usage, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        return first, last_tok, usage, time.perf_counter() - t0, repr(e)[:300]


def versions():
    r = wsl(f"{VENV}/bin/python -c \"import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)\"")
    v = r.stdout.split()
    return {"vllm": v[0] if v else None, "torch": v[1] if len(v) > 1 else None, "cuda": v[2] if len(v) > 2 else None,
            "wsl_distro": DISTRO}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workloads", default="chat-short,chat-long,decide")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    repo, rev, quant, template = ARMS[a.arm]
    levels = [int(x) for x in a.levels.split(",")]
    workloads = a.workloads.split(",")
    os.makedirs(a.out, exist_ok=True)
    cdir = os.path.join(a.repo, "cases", "c8")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    for f, h in man["files_sha256"].items():
        if base.sha(open(os.path.join(cdir, f), "rb").read()) != h:
            sys.exit(f"cases/c8/{f} does not match its manifest")
    chat = [json.loads(l) for l in open(os.path.join(cdir, "chat.jsonl"), encoding="utf-8")]
    decide = [json.loads(l) for l in open(os.path.join(cdir, "decide.jsonl"), encoding="utf-8")]
    pool = gen.passages(os.path.join(a.data, "test.parquet"))
    meta = {"test": "C8", "arm": f"vllm-{a.arm}", "engine": "vllm", "hf_model": repo, "revision": rev, "quantisation": quant,
            "template_decide": template, "levels": levels, "reps": a.reps, "workloads": workloads, "num_ctx": base.NUM_CTX,
            "num_predict": base.NUM_PREDICT, "limit": a.limit, "job_id": os.environ.get("JMB_JOB_ID"),
            "versions": versions(), "inputs": man["files_sha256"], "calibration": {}, "levels_detail": [],
            "sections": [], "started": ol.now()}
    fo = open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8")
    lock = threading.Lock()
    chars = {}
    exit_code = 0

    def write(row):
        with lock:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()

    def chat_body(text, max_tokens=base.NUM_PREDICT):
        return {"model": "m", "messages": [{"role": "user", "content": gen.INSTRUCTION.format(text=text)}],
                "max_tokens": max_tokens, "temperature": 0, "seed": 1,
                "chat_template_kwargs": {"enable_thinking": False}}

    def one_chat(item, w, level, rep):
        text = gen.chat_text(item["order"], pool, chars[w])
        first, last_tok, usage, wall, err = stream_chat(chat_body(text))
        n = (usage or {}).get("completion_tokens")
        gen_s = (last_tok - first) if (first is not None and last_tok is not None and last_tok > first) else None
        return {"workload": w, "level": level, "rep": rep, "idx": item["idx"], "warmup": item["warmup"], "ttft_s": first,
                "wall_s": round(wall, 4), "error": err, "prompt_eval_count": (usage or {}).get("prompt_tokens"),
                "eval_count": n, "prompt_eval_s": None, "eval_s": gen_s, "load_s": None, "total_s": wall,
                "done_reason": None, "eval_s_note": "client time from first to last streamed token"}

    def one_decide(item, level, rep):
        body = {"model": "m", "prompt": item["prompts"][template], "max_tokens": 1, "temperature": 0, "logprobs": 20}
        t0 = time.perf_counter()
        try:
            r = post("/v1/completions", body, timeout=300)
            err = None
        except Exception as e:  # noqa: BLE001
            r, err = None, repr(e)[:300]
        wall = time.perf_counter() - t0
        u = (r or {}).get("usage") or {}
        ch = ((r or {}).get("choices") or [{}])[0]
        return {"workload": "decide", "level": level, "rep": rep, "idx": item["line"], "warmup": item["warmup"],
                "wall_s": round(wall, 4), "error": err, "first_token": ch.get("text"),
                "prompt_eval_count": u.get("prompt_tokens"), "eval_count": u.get("completion_tokens"),
                "prompt_eval_s": None, "eval_s": None, "load_s": None, "total_s": wall, "done_reason": ch.get("finish_reason")}

    def calibrate():
        out = {}
        for w, target in gen.CHAT.items():
            item = [c for c in chat if c["workload"] == w][0]
            p0 = post("/v1/chat/completions", {**chat_body("", 1)})["usage"]["prompt_tokens"]
            n_chars, cal = int((target - p0) * 3.0), []
            for _ in range(3):
                n = post("/v1/chat/completions", chat_body(gen.chat_text(item["order"], pool, n_chars), 1))["usage"]["prompt_tokens"]
                cal.append({"chars": n_chars, "tokens": n})
                per = n_chars / max(1, n - p0)
                n_chars = int((target - p0) * per)
            out[w] = {"target_tokens": target, "overhead_tokens": p0, "chars_per_token": per, "chars": n_chars, "calls": cal}
            chars[w] = n_chars
        return out

    try:
        for level in levels:
            lv = {"level": level, "engine": "vllm --max-num-seqs", "start": ol.now()}
            with VllmServer(repo, rev, level, os.path.join(a.out, f"vllm-{level}.log")) as srv:
                lv["startup_s"] = round(srv.startup_s, 1)
                if not chars:
                    meta["calibration"] = calibrate()
                for w in workloads:
                    items = [c for c in chat if c["workload"] == w] if w != "decide" else decide
                    warm = [x for x in items if x["warmup"]]
                    timed = [x for x in items if not x["warmup"]][: a.limit or None]
                    for rep in range(1, a.reps + 1):
                        fn = (lambda x, lv_=level, r_=rep, w_=w: one_chat(x, w_, lv_, r_)) if w != "decide" else \
                             (lambda x, lv_=level, r_=rep: one_decide(x, lv_, r_))
                        with ThreadPoolExecutor(max_workers=level) as ex:
                            for row in ex.map(fn, warm):
                                write(row)
                        sec = {"name": f"{w}@{level}#{rep}", "workload": w, "level": level, "rep": rep,
                               "requests": len(timed), "gpu_before": base.gpu_now(), "start": ol.now()}
                        t0 = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=level) as ex:
                            rows = list(ex.map(fn, timed))
                        wall = time.perf_counter() - t0
                        sec["end"] = ol.now()
                        for row in rows:
                            write(row)
                        gtok = sum(r["eval_count"] or 0 for r in rows if not r["error"])
                        sec.update({"wall_s": round(wall, 3), "errors": sum(1 for r in rows if r["error"]),
                                    "generated_tokens": gtok, "prompt_tokens": sum(r["prompt_eval_count"] or 0 for r in rows),
                                    "agg_gen_tok_s": gtok / wall if wall else None,
                                    "requests_per_h": len(rows) / wall * 3600 if wall else None})
                        meta["sections"].append(sec)
                        print(f"{sec['name']}: {len(rows)} req, {wall:.1f}s, {sec['agg_gen_tok_s']:.1f} tok/s agg, {sec['errors']} errors", flush=True)
                    if w == "chat-short" and level == 1:
                        v = [s["agg_gen_tok_s"] for s in meta["sections"] if s["workload"] == "chat-short" and s["level"] == 1]
                        import statistics  # noqa: PLC0415
                        cv = statistics.stdev(v) / statistics.mean(v) if len(v) > 1 else None
                        meta["gate"] = {"rule": "chat-short at concurrency 1, three repetitions, CV <= 5 %", "values": v, "cv": cv,
                                        "passed": cv is not None and cv <= base.CV_MAX}
                        if not meta["gate"]["passed"] and not a.limit:
                            exit_code = 3
                            raise StopIteration
            lv["end"] = ol.now()
            meta["levels_detail"].append(lv)
    except StopIteration:
        pass
    finally:
        fo.close()
        wsl("pkill -f 'vllm serve' || true")
        meta["finished"] = ol.now()
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
