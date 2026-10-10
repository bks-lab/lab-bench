"""C7 runner: long context on one card (plan/c7.md, generator bench/gen-c7.py).

  python bench/c7/run_c7.py --arm qwen3-14b --data D:\\bench-data\\c6 --out <dir>
        [--lengths 4096,16384,32768] [--kv f16|q8_0] [--limit-per-cell N] [--mock]

Per length: three dry calls measure how many haystack characters give the
target prompt length in the model's own tokens (Ollama prompt_eval_count),
aiming at 99 % of the target; then every trial of that length runs once.
`num_ctx` is the target plus 512. A trial whose prompt lands outside 98 to
102 % of the target (its text tokenizes differently from the calibration
text) is rescaled and sent again, at most twice; the discarded attempts are
recorded in `length_retries`, and a prompt still outside is flagged
(`length_off`) and scored.

Order: the 60 `single` prompts at 4k first. Below 95 % right there, the run
stops with exit 3 before any long prompt (sanity gate of plan/c7.md).

Every pass runs in a private `ollama serve` on port 11435 (one request at a
time, host prompt cache off); --kv q8_0 adds OLLAMA_KV_CACHE_TYPE=q8_0 and
flash attention (q8_0 needs it). After each calibration the runner reads the
server log's "offloaded X/Y layers to GPU" line: below Y the model is split
between GPU and CPU at that length.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BENCH, "lib"))
import ollama as ol  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_c7", os.path.join(BENCH, "gen-c7.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

ARMS = {
    "qwen3-8b": ("qwen3:8b", False),
    "qwen3-14b": ("qwen3:14b", False),
    "gemma4-12b": ("gemma4:12b", False),
    "gemma4-26b": ("gemma4:26b", False),
    "mistral-small-3.2": ("mistral-small3.2:24b", False),
    "qwen3.8-27b": ("qwen3.8:27b", False),
    "gpt-oss-20b": ("gpt-oss:20b", "low"),
}
NUM_PREDICT = 512
AIM = 0.99
GATE = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--lengths", default="4096,16384,32768")
    ap.add_argument("--kv", default="f16", choices=["f16", "q8_0"])
    ap.add_argument("--limit-per-cell", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true", help="only for a rerun that names why")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    tag, think = ARMS[a.arm]
    lengths = [int(x) for x in a.lengths.split(",")]
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.repo, "cases", "c7", "manifest.json"), encoding="utf-8"))
    body = open(os.path.join(a.repo, "cases", "c7", "trials.jsonl"), encoding="utf-8").read()
    if __import__("hashlib").sha256(body.encode("utf-8")).hexdigest() != man["trials_sha256"]:
        sys.exit("cases/c7/trials.jsonl does not match its manifest")
    trials = [json.loads(l) for l in body.splitlines()]
    pool = gen.passages(os.path.join(a.data, "test.parquet"))
    if a.limit_per_cell:
        trials = [t for t in trials if t["trial"] < a.limit_per_cell]

    server = None
    url = None
    log_path = os.path.join(a.out, "ollama-private.log")
    if not a.mock:
        # Every pass runs in a private `ollama serve` (port 11435): one request at a time, no host
        # prompt cache (the calibration calls would otherwise feed the first prompts), and a log
        # whose "offloaded X/Y layers" line tells whether the model stayed on the card
        # (plan/c7.md, notes of 2026-10-11). f16 is the default cache type; q8_0 needs flash attention.
        env = {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "LLAMA_ARG_CACHE_RAM": "0",
               "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", r"D:\ollama\models")}
        if a.kv == "q8_0":
            env.update({"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"})
        server = ol.PrivateServer(11435, env, log_path).__enter__()
        url = server.url
    info = None if a.mock else ol.model_info(tag, url=url)
    think_field = None if a.mock else ol.think_setting(info, think)
    meta = {"test": "C7", "arm": a.arm, "ollama_tag": tag, "kv_cache": a.kv, "lengths": lengths,
            "num_predict": NUM_PREDICT, "aim": AIM, "think": think_field if think_field is not None else "not supported by the model",
            "server": ("private ollama serve, port 11435, OLLAMA_NUM_PARALLEL=1, LLAMA_ARG_CACHE_RAM=0"
                       + (", OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_FLASH_ATTENTION=1" if a.kv == "q8_0" else ", f16 cache and flash attention as Ollama decides")),
            "model": info, "versions": {"python": sys.version.split()[0], "ollama": None if a.mock else ol.version(url)},
            "env_seen": {k: os.environ.get(k) for k in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_PARALLEL")},
            "inputs": {"trials_sha256": man["trials_sha256"]}, "job_id": os.environ.get("JMB_JOB_ID"),
            "calibration": {}, "sections": [], "started": ol.now()}

    def send(text, L, predict=NUM_PREDICT):
        if a.mock:
            n = len(text) // 3
            return {"message": {"content": " ".join(gen.CODE_RE.findall(text)[:1])}, "prompt_eval_count": n, "eval_count": 1}, 0.0, None
        b = {"model": tag, "messages": [{"role": "user", "content": text}], "stream": False, "keep_alive": "30m",
             "options": {"temperature": 0, "seed": 1, "num_ctx": L + 512, "num_predict": predict}}
        if think_field is not None:
            b["think"] = think_field
        return ol.call("/api/chat", b, timeout=1800, url=url)

    def calibrate(L):
        """chars of haystack for the target: empty-haystack overhead, then two measured sizes."""
        t0 = trials[0]
        cal = {"start": ol.now(), "calls": []}
        r, _, err = send(gen.PROMPT.format(haystack="", question=t0["question"]), L, predict=1)
        p0 = (r or {}).get("prompt_eval_count") or 0
        cal["calls"].append({"chars": 0, "tokens": p0, "error": err})
        chars = int((L * AIM - p0) * 2.6)  # first guess, safely below num_ctx for German text
        good = 0
        for _ in range(6):
            r, _, err = send(gen.prompt(t0, pool, chars), L, predict=1)
            n = (r or {}).get("prompt_eval_count")
            cal["calls"].append({"chars": chars, "tokens": n, "error": err})
            if not n or n <= p0:
                raise SystemExit(f"calibration failed at {L}: {err}")
            if n >= L + 400:  # cut by the context: the count says nothing, try a smaller haystack
                chars = int(chars * 0.7)
                continue
            per_tok = chars / (n - p0)
            chars = int((L * AIM - p0) * per_tok)
            good += 1
            if good == 2:
                break
        cal.update({"overhead_tokens": p0, "chars_per_token": per_tok, "haystack_chars": chars, "end": ol.now()})
        return chars, cal

    fo = open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8")
    plan = []
    for L in lengths:
        tl = [t for t in trials if t["length"] == L]
        if L == lengths[0] and L == 4096:
            plan.append((L, [t for t in tl if t["task"] == "single"], "gate"))
            plan.append((L, [t for t in tl if t["task"] != "single"], "rest"))
        else:
            plan.append((L, tl, "all"))
    chars_for = {}
    gate = None
    try:
        for L, tl, part in plan:
            if L not in chars_for:
                chars_for[L], meta["calibration"][str(L)] = calibrate(L)
                meta["calibration"][str(L)]["layers_on_gpu"] = None if a.mock else ol.layers_on_gpu(log_path)
            sec = {"name": f"{L}-{part}", "length": L, "prompts": len(tl), "start": ol.now()}
            right = 0
            for n, t in enumerate(tl):
                hc = chars_for[L]
                retries = []
                for attempt in range(3):
                    r, secs, err = send(gen.prompt(t, pool, hc), L)
                    pe0 = (r or {}).get("prompt_eval_count")
                    if err or not pe0 or 0.98 * L <= pe0 <= 1.02 * L or attempt == 2:
                        break
                    # this trial's text tokenizes differently: rescale its haystack and send again
                    retries.append({"haystack_chars": hc, "prompt_eval_count": pe0, "seconds": round(secs, 3)})
                    p0 = meta["calibration"][str(L)]["overhead_tokens"]
                    hc = int(hc * (L * AIM - p0) / max(1, pe0 - p0))
                msg = (r or {}).get("message") or {}
                c = ol.counters(r)
                ok = gen.judge(msg.get("content"), t) if not err else False
                right += ok
                pe = c["prompt_eval_count"]
                row = {"id": t["id"], "arm": a.arm, "kv": a.kv, "length": L, "task": t["task"], "depth": t["depth"],
                       "trial": t["trial"], "answer": msg.get("content"), "thinking_chars": len(msg.get("thinking") or ""),
                       "right": ok, "seconds": round(secs, 3), "error": err, **c,
                       "length_off": bool(pe) and not (0.98 * L <= pe <= 1.02 * L), "haystack_chars": hc,
                       "length_retries": retries}
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
                if n == 0:
                    sec["ps"] = ol.ps(url) if not a.mock else []
                if n % 20 == 0:
                    print(f"{L} {part} {n}/{len(tl)} right={right} pe={pe} {secs:.1f}s {err or ''}", flush=True)
            sec["end"] = ol.now()
            sec["right"] = right
            meta["sections"].append(sec)
            if part == "gate":
                gate = {"rule": "single at 4k, at least 95 % right (plan/c7.md)", "right": right, "prompts": len(tl),
                        "share": right / len(tl) if tl else None, "passed": bool(tl) and right / len(tl) >= GATE}
                meta["gate"] = gate
                print("gate", json.dumps(gate), flush=True)
                if not gate["passed"] and not a.no_gate:
                    break
    finally:
        fo.close()
        if not a.mock:
            ol.unload(tag, url=url)
        if server:
            server.__exit__(None, None, None)
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if gate and not gate["passed"] and not a.no_gate:
        sys.exit(3)


if __name__ == "__main__":
    main()
