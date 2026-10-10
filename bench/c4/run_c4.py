"""C4 runner: structured output and tool calling (plan/c4.md, results/c4/method.md).

One arm per call against the local Ollama; four modes, one after the other:

  fc           BFCL non-live, the 1,000 items with Ollama's native `tools` field
  prompt       BFCL non-live, functions in BFCL's prompt-mode system prompt, answer as text
  free         JSONSchemaBench sample (600), schema in the prompt, plain generation
  constrained  the same 600 with Ollama `format` = the schema

Writes <out>/result.jsonl (one row per item and mode, written as it goes) and
<out>/meta.json. Nothing is scored here: the AST check needs the BFCL code and
runs on the Mac (bench/c4/check_ast.py); schema validity is rechecked there
too. The runner only records what the model returned.

  python bench/c4/run_c4.py --arm qwen3-14b --out D:\\jmb-queue\\out\\<id> [--modes fc,prompt] [--limit 20]
  python bench/c4/run_c4.py --arm qwen3-14b --out /tmp/x --mock          (no model, IO check)
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import ollama as ol  # noqa: E402

NUM_CTX = 8192
BASE_OPTIONS = {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX}
# Generation caps (plan/c4.md, addendum 2026-10-10): BFCL answers are a few
# calls; a schema instance can be long. A capped answer is recorded with
# done_reason "length" and scored as returned.
NUM_PREDICT = {"fc": 1024, "prompt": 1024, "free": 4096, "constrained": 4096}
KEEP = "30m"
CALL_TIMEOUT_S = 600

# arm -> (Ollama tag, think setting when the model can think)
ARMS = {
    "qwen3-8b": ("qwen3:8b", False),
    "qwen3-14b": ("qwen3:14b", False),
    "qwen3.8-27b": ("qwen3.8:27b", False),
    "gemma4-12b": ("gemma4:12b", False),
    "gemma4-26b": ("gemma4:26b", False),
    "mistral-small-3.2": ("mistral-small3.2:24b", False),
    "gpt-oss-20b": ("gpt-oss:20b", "low"),
    "granite4-3b": ("granite4:micro", False),
}
MODES = ["fc", "prompt", "free", "constrained"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_method(repo):
    """The fixed Part B instruction from results/c4/method.md (committed before the first run)."""
    import re  # noqa: PLC0415
    s = open(os.path.join(repo, "results", "c4", "method.md"), encoding="utf-8").read()
    return re.search(r"## Part B instruction\s+```text\n(.*?)\n```", s, re.S).group(1)


def compact(schema_text):
    return json.dumps(json.loads(schema_text), ensure_ascii=False, separators=(",", ":"))


def tool_calls_of(msg):
    out = []
    for tc in (msg or {}).get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        out.append({"name": fn.get("name"), "arguments": args})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--limit", type=int, default=0, help="first N items per mode (probe)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    tag, think = ARMS[a.arm]
    modes = [m for m in a.modes.split(",") if m]
    assert all(m in MODES for m in modes), modes
    os.makedirs(a.out, exist_ok=True)
    cdir = os.path.join(a.repo, "cases", "c4")
    manifest = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    for f, h in manifest["files_sha256"].items():
        if sha(os.path.join(cdir, f)) != h:
            sys.exit(f"input hash mismatch: cases/c4/{f}")
    bfcl = [json.loads(l) for l in open(os.path.join(cdir, "bfcl.jsonl"), encoding="utf-8")]
    schemas = [json.loads(l) for l in open(os.path.join(cdir, "schemas.jsonl"), encoding="utf-8")]
    instr = load_method(a.repo)
    if a.limit:
        bfcl, schemas = bfcl[: a.limit], schemas[: a.limit]

    info = None if a.mock else ol.model_info(tag)
    think_field = None if a.mock else ol.think_setting(info, think)
    meta = {"test": "C4", "arm": a.arm, "ollama_tag": tag, "num_ctx": NUM_CTX, "options": BASE_OPTIONS,
            "num_predict": NUM_PREDICT, "think": think_field if think_field is not None else "not supported by the model",
            "modes": modes, "limit": a.limit, "mock": a.mock, "job_id": os.environ.get("JMB_JOB_ID"),
            "model": info, "versions": {"python": sys.version.split()[0], "ollama": None if a.mock else ol.version()},
            "inputs": {"manifest_sha256": sha(os.path.join(cdir, "manifest.json")),
                       "method_sha256": sha(os.path.join(a.repo, "results", "c4", "method.md"))},
            "sections": [], "started": ol.now()}

    def body_for(mode, item):
        opts = {**BASE_OPTIONS, "num_predict": NUM_PREDICT[mode]}
        if mode == "fc":
            b = {"model": tag, "messages": item["fc"]["messages"], "tools": item["fc"]["tools"]}
        elif mode == "prompt":
            b = {"model": tag, "messages": item["prompt"]["messages"]}
        else:
            msg = instr.replace("{schema}", compact(item["schema"]))
            b = {"model": tag, "messages": [{"role": "user", "content": msg}]}
            if mode == "constrained":
                b["format"] = json.loads(item["schema"])
        b.update({"options": opts, "stream": False, "keep_alive": KEEP})
        if think_field is not None:
            b["think"] = think_field
        return b

    def run_one(mode, item):
        if a.mock:
            return {"message": {"role": "assistant", "content": "[]"}, "prompt_eval_count": 0, "eval_count": 0}, 0.0, None
        return ol.call("/api/chat", body_for(mode, item), timeout=CALL_TIMEOUT_S)

    fo = open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8")
    summary = {}
    for mode in modes:
        items = bfcl if mode in ("fc", "prompt") else schemas
        warm = {"name": f"warmup_{mode}", "start": ol.now()}
        for it in items[: a.warmup]:
            run_one(mode, it)
        warm["end"] = ol.now()
        meta["sections"].append(warm)
        sec = {"name": mode, "items": len(items), "start": ol.now()}
        if not a.mock:
            sec["ps_before"] = ol.ps()
        errs = 0
        for n, it in enumerate(items):
            r, secs, err = run_one(mode, it)
            msg = (r or {}).get("message") or {}
            row = {"id": it["id"], "mode": mode, "arm": a.arm,
                   "category": it.get("category") or it.get("subset"),
                   "content": msg.get("content"), "tool_calls": tool_calls_of(msg),
                   "thinking_chars": len(msg.get("thinking") or ""),
                   "seconds": round(secs, 4), "error": err, **ol.counters(r)}
            row["truncated_prompt"] = bool(row["prompt_eval_count"] and row["prompt_eval_count"] >= NUM_CTX - 8)
            if err:
                errs += 1
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if n % 50 == 0:
                print(f"{mode} {n}/{len(items)} {secs:.2f}s {err or ''}", flush=True)
            if n == 1 and not a.mock:
                sec["ps_during"] = ol.ps()
        sec["end"] = ol.now()
        sec["errors"] = errs
        meta["sections"].append(sec)
        summary[mode] = {"items": len(items), "errors": errs}
        print(f"{mode} done: {len(items)} items, {errs} errors", flush=True)
    fo.close()
    if not a.mock:
        ol.unload(tag)
    meta["summary"] = summary
    meta["finished"] = ol.now()
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if all(s["errors"] == s["items"] for s in summary.values()):
        sys.exit("every call failed, see result.jsonl")


if __name__ == "__main__":
    main()
