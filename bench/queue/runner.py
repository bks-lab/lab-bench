"""PC job queue runner for lab-bench (spec: plan/ROADMAP.md, "PC job queue").

One runner process on the workstation takes job files from
D:\\jmb-queue\\incoming\\ in name order and runs one at a time. Agents only
enqueue (bench/queue/enqueue.sh) and fetch (bench/queue/fetch.sh).

Usage on the PC (stdlib only, any Python 3.10+):
    python runner.py run        the loop (started by the scheduled task "bench-queue")
    python runner.py status     running job, queue, last ten results
    python runner.py gate       one reading of the GPU gate, for debugging

Layout under the root (default D:\\jmb-queue, override with JMB_QUEUE_ROOT):
    incoming\\  running\\  done\\  failed\\  logs\\  out\\<id>\\  src\\<sha>[.tar]
    PAUSE              finish the current job, then wait
    HOLD-<name>        no GPU job starts while it exists; JSON body may carry
                       {"reason": "...", "release_after_idle_min": 60,
                        "require_busy_first": true}: the hold is released by
                       itself once the GPU was free that long (after an Ollama
                       model was seen loaded, if require_busy_first)
    runner.lock        PID of the runner holding the queue
    runner.state.json  heartbeat: state, reason, job, updated
    config.json        optional overrides of DEFAULTS below

Job file (incoming\\<yyyymmdd-hhmmss>-<test>-<arm>.json), written by enqueue.sh:
    id, key ("<TEST>-<arm>"), test, arm, repo_ref (commit sha), cwd, cmd (list),
    env, gpu (bool), timeout_min, expected_gpu_min, outputs, after_ok (keys that
    must have succeeded before), requested_by, optional ollama_env.
Placeholders in cwd, cmd, env and outputs: {src} {py} {data} {hf} {out} {root}
and {out:KEY} (out dir of the latest successful job with that key).

GPU gate (before every job with gpu=true), all must hold continuously for
clear_window_s: no HOLD-* file, Ollama /api/ps lists no model, memory.used
below gpu_mem_max_mib (60 s instead of 10 min when the runner's own previous
job ended less than 5 min ago). nvidia-smi --query-compute-apps is not used: under
Windows WDDM it lists every desktop process with "[N/A]" memory, so it cannot
tell a compute job from Explorer.
"""

from __future__ import annotations

import csv
import ctypes
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

DEFAULTS = {
    "py": r"D:\bench-cap\.venv\Scripts\python.exe",
    "data": r"D:\bench-data",
    "hf": r"D:\hf",
    "ollama_url": "http://127.0.0.1:11434",
    "gpu_mem_max_mib": 3000,
    "clear_window_s": 600,
    "clear_window_after_own_s": 60,
    "poll_s": 30,
    "allow_ollama_restart": False,
    "ollama_exe": os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    "ollama_task": "ollama-serve",
}

SUBDIRS = ("incoming", "running", "done", "failed", "logs", "out", "src")
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


# ---------------------------------------------------------------- pure helpers


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def substitute(value, mapping: dict, resolve_out=None):
    """Fill {name} and {out:KEY} placeholders in strings, lists and dicts."""
    if isinstance(value, list):
        return [substitute(v, mapping, resolve_out) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, mapping, resolve_out) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    def repl(m):
        name = m.group(1)
        if name.startswith("out:"):
            if resolve_out is None:
                raise KeyError(name)
            got = resolve_out(name[4:])
            if got is None:
                raise KeyError(f"no successful job for {name[4:]}")
            return got
        if name not in mapping:
            raise KeyError(name)
        return mapping[name]

    return re.sub(r"\{([a-z]+(?::[A-Za-z0-9_.+-]+)?)\}", repl, value)


def latest_status(statuses: list[dict], key: str) -> dict | None:
    """Latest status (by job id, which starts with the enqueue timestamp) for a key."""
    hits = [s for s in statuses if s.get("key") == key]
    return max(hits, key=lambda s: s["id"]) if hits else None


def check_deps(job: dict, statuses: list[dict]) -> tuple[bool, str]:
    for key in job.get("after_ok") or []:
        s = latest_status(statuses, key)
        if s is None:
            return False, f"dependency {key} has not run"
        if not s.get("ok"):
            return False, f"dependency {key} failed ({s['id']})"
    return True, ""


def gpu_csv_stats(path: Path) -> dict:
    """Peak and idle memory, energy (sum of 1 s power samples), mean power, minutes."""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.reader(f):
                if not r or r[0].strip().startswith("timestamp"):
                    continue
                try:
                    mem = float(r[1].strip().split()[0])
                    pw = float(r[2].strip().split()[0])
                except (ValueError, IndexError):
                    continue
                rows.append((r[0].strip(), mem, pw))
    except FileNotFoundError:
        return {"samples": 0}
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "first": rows[0][0],
        "last": rows[-1][0],
        "idle_mem_mib": rows[0][1],
        "peak_mem_mib": max(r[1] for r in rows),
        "peak_over_idle_mib": max(r[1] for r in rows) - rows[0][1],
        "energy_j": round(sum(r[2] for r in rows), 1),
        "mean_power_w": round(sum(r[2] for r in rows) / len(rows), 1),
        "gpu_min": round(len(rows) / 60.0, 2),
        "note": "GPU only, whole job incl. model load; 1 s samples",
    }


def gate_reasons(reading: dict, cfg: dict) -> list[str]:
    """Why the GPU is not free right now (empty list = free)."""
    out = []
    if reading.get("holds"):
        out.append("hold: " + ", ".join(reading["holds"]))
    models = reading.get("ollama_models")
    if models is None:
        out.append("ollama /api/ps unreadable")
    elif models:
        out.append("ollama has a model loaded: " + ", ".join(models))
    mem = reading.get("mem_used_mib")
    if mem is None:
        out.append("nvidia-smi unreadable")
    elif mem >= cfg["gpu_mem_max_mib"]:
        out.append(f"GPU memory used {mem:.0f} MiB >= {cfg['gpu_mem_max_mib']}")
    return out


# ---------------------------------------------------------------- system probes


def run_text(args, timeout=30) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            encoding="utf-8", errors="replace",
        ).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}"


def http_json(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def ollama_models(cfg) -> list[str] | None:
    try:
        return [m.get("name") or m.get("model") for m in http_json(cfg["ollama_url"] + "/api/ps").get("models", [])]
    except Exception:  # noqa: BLE001
        return None


def gpu_mem_used() -> float | None:
    out = run_text(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def ram_gb() -> float | None:
    if os.name != "nt":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.total / 2**30, 1)


def machine_record(cfg) -> dict:
    q = run_text(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.gr,clocks.mem,memory.total",
                  "--format=csv,noheader"]).strip()
    head = run_text(["nvidia-smi"])
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)
    try:
        ollama_v = http_json(cfg["ollama_url"] + "/api/version").get("version")
    except Exception:  # noqa: BLE001
        ollama_v = None
    cpu = run_text(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip() \
        if os.name == "nt" else platform.processor()
    py_v = run_text([cfg["py"], "-c", "import sys;print(sys.version.split()[0])"]).strip() \
        if Path(cfg["py"]).exists() else None
    return {
        "host": platform.node(),
        "label": "local-rtx4090",
        "recorded_at": now_iso(),
        "gpu_query": "name,driver_version,power.limit,clocks.gr,clocks.mem,memory.total (clocks at job start)",
        "gpu": q,
        "cuda_version": cuda.group(1) if cuda else None,
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "cpu": cpu,
        "ram_gb": ram_gb(),
        "ollama_version": ollama_v,
        "job_python": py_v,
        "runner_python": sys.version.split()[0],
    }


# ---------------------------------------------------------------- the queue


class Queue:
    def __init__(self, root: Path):
        self.root = root
        self.cfg = dict(DEFAULTS)
        cfg_file = root / "config.json"
        if cfg_file.exists():
            self.cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
        for d in SUBDIRS:
            (root / d).mkdir(parents=True, exist_ok=True)
        self.idle_since: float | None = None
        self.last_end: float | None = None

    # -- bookkeeping
    def log(self, msg: str):
        line = f"{now_iso()} {msg}\n"
        with open(self.root / "logs" / "runner.log", "a", encoding="utf-8") as f:
            f.write(line)

    def state(self, state: str, **kw):
        body = {"pid": os.getpid(), "state": state, "updated": now_iso(), **kw}
        tmp = self.root / "runner.state.json.tmp"
        tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
        os.replace(tmp, self.root / "runner.state.json")

    def statuses(self) -> list[dict]:
        out = []
        for d in ("done", "failed"):
            for p in (self.root / d).glob("*.status.json"):
                try:
                    out.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    pass
        return out

    def resolve_out(self, key: str) -> str | None:
        s = latest_status([s for s in self.statuses() if s.get("ok")], key)
        return s.get("out") if s else None

    def holds(self) -> list[Path]:
        return sorted(p for p in self.root.glob("HOLD-*") if ".released-" not in p.name)

    def acquire_lock(self) -> bool:
        lock = self.root / "runner.lock"
        if lock.exists():
            try:
                pid = int(lock.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and pid_alive(pid):
                return False
        lock.write_text(str(os.getpid()))
        return True

    # -- gate
    def gate_reading(self) -> dict:
        return {
            "holds": [p.name for p in self.holds()],
            "ollama_models": ollama_models(self.cfg),
            "mem_used_mib": gpu_mem_used(),
        }

    def observe(self) -> dict:
        """Read the gate, track how long the GPU itself has been free, update holds.

        A hold with "release_after_idle_min" is released once the GPU was free
        that long. With "require_busy_first": true it first has to see an Ollama
        model loaded (the held-for work actually ran) after the hold was set, so
        a hold placed before that work starts is not released by the quiet
        before it.
        """
        reading = self.gate_reading()
        t = time.time()
        if gate_reasons({**reading, "holds": []}, self.cfg):
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = t
        for h in self.holds():
            try:
                body = json.loads(h.read_text(encoding="utf-8") or "{}")
            except Exception:  # noqa: BLE001
                continue
            if reading.get("ollama_models") and not body.get("seen_busy_at"):
                body["seen_busy_at"] = now_iso()
                body["seen_models"] = reading["ollama_models"]
                h.write_text(json.dumps(body), encoding="utf-8")
                self.log(f"hold {h.name}: saw ollama busy with {reading['ollama_models']}")
            rel = body.get("release_after_idle_min")
            if rel is None or self.idle_since is None:
                continue
            if body.get("require_busy_first") and not body.get("seen_busy_at"):
                continue
            if t - self.idle_since >= float(rel) * 60:
                h.rename(h.with_name(h.name + f".released-{dt.datetime.now():%Y%m%d-%H%M%S}"))
                self.log(f"hold {h.name} released after {(t - self.idle_since) / 60:.0f} min idle GPU")
        reading["holds"] = [p.name for p in self.holds()]
        reading["gpu_idle_min"] = round((t - self.idle_since) / 60, 1) if self.idle_since else 0
        return reading

    def wait_for_gpu(self, job_id: str) -> bool:
        """Block until the gate is clear for clear_window_s. False if PAUSE appeared."""
        clear_since = None
        window = self.cfg["clear_window_s"]
        if self.last_end and time.time() - self.last_end < 300:
            window = self.cfg["clear_window_after_own_s"]  # the GPU was ours a moment ago
        while True:
            if (self.root / "PAUSE").exists():
                return False
            reading = self.observe()
            reasons = gate_reasons(reading, self.cfg)
            t = time.time()
            if reasons:
                clear_since = None
                self.state("waiting", job=job_id, reason="; ".join(reasons), gpu_idle_min=reading["gpu_idle_min"])
            else:
                clear_since = clear_since or t
                if t - clear_since >= window:
                    return True
                self.state("waiting", job=job_id, reason=f"gate clear for {t - clear_since:.0f}s of {window}s")
            time.sleep(self.cfg["poll_s"])

    # -- one job
    def prepare_src(self, sha: str) -> Path:
        src = self.root / "src" / sha
        if src.exists():
            return src
        tar = self.root / "src" / f"{sha}.tar"
        if not tar.exists():
            raise FileNotFoundError(f"no source for {sha}: upload src\\{sha}.tar (enqueue.sh does)")
        tmp = self.root / "src" / f"{sha}.extracting"
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(tar) as t:
            t.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else t.extractall(tmp)
        os.replace(tmp, src)
        return src

    def finish(self, job: dict, job_path: Path, ok: bool, status: dict):
        dest = self.root / ("done" if ok else "failed")
        status.update({"id": job["id"], "key": job.get("key"), "test": job.get("test"), "arm": job.get("arm"),
                       "ok": ok, "repo_ref": job.get("repo_ref"), "requested_by": job.get("requested_by")})
        (dest / f"{job['id']}.status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        if job_path.exists():
            os.replace(job_path, dest / job_path.name)
        self.log(f"{'done' if ok else 'FAILED'} {job['id']}: {status.get('reason', '')}")

    def unload_ollama(self, before: list[str]):
        for m in ollama_models(self.cfg) or []:
            if m in before:
                continue
            try:
                http_json(self.cfg["ollama_url"] + "/api/generate", {"model": m, "keep_alive": 0}, timeout=60)
                self.log(f"unloaded ollama model {m}")
            except Exception as e:  # noqa: BLE001
                self.log(f"unload {m} failed: {e}")

    def run_job(self, job_path: Path):
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job.setdefault("id", job_path.stem)
            job.setdefault("key", f"{job.get('test')}-{job.get('arm')}")
        except Exception as e:  # noqa: BLE001
            job = {"id": job_path.stem}
            self.finish(job, job_path, False, {"reason": f"unreadable job file: {e}"})
            return

        ok, why = check_deps(job, self.statuses())
        if not ok:
            self.finish(job, job_path, False, {"reason": why})
            return
        if job.get("ollama_env") and not self.cfg["allow_ollama_restart"]:
            self.finish(job, job_path, False, {"reason": "job needs ollama_env, allow_ollama_restart is off"})
            return

        if job.get("gpu", True):
            if not self.wait_for_gpu(job["id"]):
                return  # PAUSE: leave the job in incoming
        # a newer PAUSE or a HOLD that appeared is handled by the loop; we start now
        running = self.root / "running" / job_path.name
        os.replace(job_path, running)
        out = self.root / "out" / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        status = {"start": now_iso(), "out": str(out)}
        try:
            src = self.prepare_src(job["repo_ref"])
            mapping = {"src": str(src), "py": self.cfg["py"], "data": self.cfg["data"], "hf": self.cfg["hf"],
                       "out": str(out), "root": str(self.root)}
            cwd = substitute(job.get("cwd", "{src}"), mapping, self.resolve_out)
            cmd = substitute(job["cmd"], mapping, self.resolve_out)
            env_add = substitute(job.get("env") or {}, mapping, self.resolve_out)
            outputs = substitute(job.get("outputs") or [], mapping, self.resolve_out)
        except Exception as e:  # noqa: BLE001
            status.update(end=now_iso(), reason=f"setup: {e}")
            self.finish(job, running, False, status)
            return

        machine = machine_record(self.cfg)
        (out / "machine.json").write_text(json.dumps(machine, indent=1), encoding="utf-8")
        gpu_csv = out / "gpu.csv"
        env = {**os.environ, **{k: str(v) for k, v in env_add.items()},
               "JMB_JOB_ID": job["id"], "JMB_OUT": str(out), "JMB_GPU_CSV": str(gpu_csv),
               "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        before = ollama_models(self.cfg) or []
        flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        gpu_fh = open(gpu_csv, "w", encoding="utf-8")
        try:
            sampler = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,memory.used,power.draw,utilization.gpu,temperature.gpu",
                 "--format=csv", "-l", "1"], stdout=gpu_fh, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as e:
            sampler = None
            self.log(f"gpu sampler not started: {e}")
        log_path = self.root / "logs" / f"{job['id']}.log"
        timeout_s = float(job.get("timeout_min", 120)) * 60
        self.state("running", job=job["id"], since=status["start"],
                   expected_gpu_min=job.get("expected_gpu_min"), timeout_min=job.get("timeout_min"))
        self.log(f"start {job['id']} cmd={cmd}")
        rc, reason = None, ""
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"# {job['id']} start {status['start']}\n# cwd {cwd}\n# cmd {cmd}\n")
            lf.flush()
            try:
                p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags)
                t0 = time.time()
                while p.poll() is None:
                    if time.time() - t0 > timeout_s:
                        run_text(["taskkill", "/T", "/F", "/PID", str(p.pid)]) if os.name == "nt" else p.kill()
                        p.wait(60)
                        reason = f"timeout after {job.get('timeout_min')} min, killed"
                        break
                    time.sleep(2)
                rc = p.returncode
            except Exception as e:  # noqa: BLE001
                reason = f"could not start: {e}"
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(10)
            except Exception:  # noqa: BLE001
                pass
        gpu_fh.close()
        self.unload_ollama(before)
        missing = [o for o in outputs if not Path(o).exists()]
        ok = rc == 0 and not reason and not missing
        if not reason and rc not in (0, None):
            reason = "sanity gate outside tolerance (exit 3)" if rc == 3 else f"exit code {rc}"
        if missing:
            reason = (reason + "; " if reason else "") + "missing outputs: " + ", ".join(missing)
        status.update(end=now_iso(), exit_code=rc, reason=reason, log=str(log_path),
                      expected_gpu_min=job.get("expected_gpu_min"), gpu=gpu_csv_stats(gpu_csv), machine=machine)
        self.finish(job, running, ok, status)
        self.last_end = time.time()

    # -- loop
    def recover(self):
        for p in (self.root / "running").glob("*.json"):
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                job = {"id": p.stem}
            job.setdefault("id", p.stem)
            self.finish(job, p, False, {"reason": "runner restarted while the job was running", "end": now_iso()})

    def loop(self):
        if not self.acquire_lock():
            print("another runner holds the lock", file=sys.stderr)
            return 1
        self.log(f"runner up, pid {os.getpid()}")
        self.recover()
        while True:
            try:
                self.observe()
                if (self.root / "PAUSE").exists():
                    self.state("paused", reason="PAUSE file present")
                    time.sleep(self.cfg["poll_s"])
                    continue
                jobs = sorted((self.root / "incoming").glob("*.json"))
                if not jobs:
                    self.state("idle", reason="queue empty")
                    time.sleep(self.cfg["poll_s"])
                    continue
                self.run_job(jobs[0])
            except Exception as e:  # noqa: BLE001
                self.log(f"loop error: {e!r}")
                time.sleep(self.cfg["poll_s"])

    def status(self) -> str:
        lines = []
        st = self.root / "runner.state.json"
        lock = self.root / "runner.lock"
        pid = lock.read_text().strip() if lock.exists() else "-"
        alive = pid.isdigit() and pid_alive(int(pid))
        lines.append(f"runner pid {pid} {'alive' if alive else 'NOT RUNNING'}")
        if st.exists():
            s = json.loads(st.read_text(encoding="utf-8"))
            lines.append(f"state {s.get('state')} job={s.get('job', '-')} reason={s.get('reason', '-')} "
                         f"updated={s.get('updated')}")
        if (self.root / "PAUSE").exists():
            lines.append("PAUSE file present")
        for h in self.holds():
            lines.append(f"hold {h.name}: {h.read_text(encoding='utf-8').strip()[:200]}")
        lines.append("")
        lines.append("queue (incoming, in run order):")
        total = 0.0
        for p in sorted((self.root / "incoming").glob("*.json")):
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                total += float(j.get("expected_gpu_min") or 0)
                lines.append(f"  {p.stem}  ~{j.get('expected_gpu_min')} GPU min  after_ok={j.get('after_ok') or []}")
            except Exception:  # noqa: BLE001
                lines.append(f"  {p.stem}  (unreadable)")
        lines.append(f"  total expected GPU min: {total:.0f}")
        lines.append("")
        lines.append("last ten results:")
        for s in sorted(self.statuses(), key=lambda s: s.get("end") or "", reverse=True)[:10]:
            g = s.get("gpu") or {}
            lines.append(f"  {'ok  ' if s.get('ok') else 'FAIL'} {s['id']}  end={s.get('end')}  "
                         f"gpu_min={g.get('gpu_min')} peak={g.get('peak_mem_mib')}MiB  {s.get('reason') or ''}")
        return "\n".join(lines)


def main(argv):
    root = Path(os.environ.get("JMB_QUEUE_ROOT", r"D:\jmb-queue"))
    q = Queue(root)
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "run":
        return q.loop()
    if cmd == "status":
        print(q.status())
        return 0
    if cmd == "gate":
        r = q.gate_reading()
        print(json.dumps({**r, "reasons": gate_reasons(r, q.cfg)}, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
