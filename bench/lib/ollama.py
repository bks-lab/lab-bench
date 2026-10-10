"""Small Ollama client shared by the C4, C6, C7 and C8 runners (stdlib only).

- http(): one JSON call against OLLAMA_HOST_URL (default the local service)
- model_info(): digest, size, details, capabilities and weight blobs of a tag
- chat(): one /api/chat call, timed, with the counters Ollama reports
- PrivateServer: a second `ollama serve` on its own port with its own
  environment (OLLAMA_NUM_PARALLEL, OLLAMA_KV_CACHE_TYPE, ...), started by
  the job and stopped by it. The always-on Ollama service of the workstation
  is never restarted or reconfigured; the queue gate has already made sure it
  holds no model when the job starts. Both read the same model folder
  (OLLAMA_MODELS of the service), so no weights are copied.
- now(): local time with offset and milliseconds, comparable with the
  nvidia-smi sampler CSV of the queue.
"""
import datetime
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def http(path, body=None, timeout=900, url=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request((url or DEFAULT_URL) + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def version(url=None):
    try:
        return http("/api/version", timeout=10, url=url).get("version")
    except Exception as e:  # noqa: BLE001
        return repr(e)


def model_info(tag, url=None):
    """Digest, size and details from /api/tags, capabilities and weight blobs from /api/show."""
    try:
        for m in http("/api/tags", timeout=30, url=url).get("models", []):
            if m.get("name") == tag or m.get("model") == tag:
                info = {"tag": tag, "digest": m.get("digest"), "size": m.get("size"), "details": m.get("details")}
                try:
                    sh = http("/api/show", {"model": tag}, timeout=60, url=url)
                    info["blobs"] = re.findall(r"sha256-([0-9a-f]{64})", sh.get("modelfile", ""))
                    info["capabilities"] = sh.get("capabilities")
                    mi = sh.get("model_info") or {}
                    info["context_length"] = next((v for k, v in mi.items() if k.endswith(".context_length")), None)
                    info["template_sha256"] = __import__("hashlib").sha256((sh.get("template") or "").encode()).hexdigest()
                except Exception as e:  # noqa: BLE001
                    info["show_error"] = repr(e)
                return info
    except Exception as e:  # noqa: BLE001
        return {"tag": tag, "error": repr(e)}
    return {"tag": tag, "error": "tag not found in /api/tags"}


def think_setting(info, think):
    """The `think` field to send: the arm's setting when the model can think, else None (field left out)."""
    caps = (info or {}).get("capabilities") or []
    return think if "thinking" in caps else None


def call(path, body, timeout=600, url=None):
    """One timed call. Returns (response dict or None, wall seconds, error string or None)."""
    t0 = time.perf_counter()
    try:
        r = http(path, body, timeout=timeout, url=url)
        return r, time.perf_counter() - t0, None
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001
            msg = ""
        return None, time.perf_counter() - t0, f"http {e.code}: {msg}"
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t0, repr(e)[:500]


def counters(r):
    """The timing and token counters of an Ollama response, durations in seconds."""
    r = r or {}
    s = lambda k: (r.get(k) / 1e9) if r.get(k) is not None else None  # noqa: E731
    return {"prompt_eval_count": r.get("prompt_eval_count"), "eval_count": r.get("eval_count"),
            "prompt_eval_s": s("prompt_eval_duration"), "eval_s": s("eval_duration"),
            "load_s": s("load_duration"), "total_s": s("total_duration"), "done_reason": r.get("done_reason")}


def unload(tag, url=None):
    try:
        http("/api/generate", {"model": tag, "keep_alive": 0}, timeout=120, url=url)
    except Exception:  # noqa: BLE001
        pass


def ps(url=None):
    """Loaded models with their VRAM share (size_vram of size): the offload check."""
    try:
        return [{"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram"),
                 "context_length": m.get("context_length")} for m in http("/api/ps", timeout=10, url=url).get("models", [])]
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]


class PrivateServer:
    """`ollama serve` on 127.0.0.1:<port> with extra environment, for the life of a job."""

    def __init__(self, port, env_extra, log_path, exe=None):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.env_extra = {k: str(v) for k, v in env_extra.items()}
        self.log_path = log_path
        self.exe = exe or os.environ.get("OLLAMA_EXE") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        self.proc = None

    def __enter__(self):
        env = {**os.environ, **self.env_extra, "OLLAMA_HOST": f"127.0.0.1:{self.port}"}
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.log.write(f"# {now()} start {self.exe} serve, env {json.dumps(self.env_extra)}\n")
        self.log.flush()
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen([self.exe, "serve"], env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=flags)
        for _ in range(120):
            if self.proc.poll() is not None:
                raise RuntimeError(f"private ollama on port {self.port} exited with {self.proc.returncode}")
            try:
                http("/api/version", timeout=2, url=self.url)
                return self
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise RuntimeError(f"private ollama on port {self.port} did not answer within 60 s")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":
                # the model runner processes of the private server are its children: kill the tree
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.log.write(f"# {now()} stopped\n")
        self.log.close()
        return False
