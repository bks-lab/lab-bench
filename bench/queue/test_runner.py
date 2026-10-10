"""Tests for the pure parts of the queue runner and a full CPU job on a fake root.

Run: python3 -m pytest bench/queue/test_runner.py
"""

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import runner  # noqa: E402

CFG = {**runner.DEFAULTS}


def test_substitute_fills_nested_and_out_key():
    m = {"src": "S", "out": "O", "py": "P"}
    got = runner.substitute(
        {"cmd": ["{py}", "x.py", "--base", "{out:C1-bge-m3}\\top.jsonl"], "cwd": "{src}"},
        m, lambda k: "D:\\q\\out\\abc" if k == "C1-bge-m3" else None)
    assert got == {"cmd": ["P", "x.py", "--base", "D:\\q\\out\\abc\\top.jsonl"], "cwd": "S"}


def test_substitute_unknown_placeholder_raises():
    with pytest.raises(KeyError):
        runner.substitute("{nope}", {}, None)
    with pytest.raises(KeyError):
        runner.substitute("{out:C9-x}", {}, lambda k: None)


def test_substitute_leaves_json_braces_alone():
    assert runner.substitute('{"a": 1}', {}, None) == '{"a": 1}'


def test_deps_latest_wins():
    st = [{"id": "20261009-100000-c1-bge-m3", "key": "C1-bge-m3", "ok": False},
          {"id": "20261009-110000-c1-bge-m3", "key": "C1-bge-m3", "ok": True}]
    assert runner.check_deps({"after_ok": ["C1-bge-m3"]}, st) == (True, "")
    ok, why = runner.check_deps({"after_ok": ["C1-bge-m3"]}, st[:1])
    assert not ok and "failed" in why
    ok, why = runner.check_deps({"after_ok": ["C2-canary"]}, st)
    assert not ok and "not run" in why
    assert runner.check_deps({}, []) == (True, "")


def test_gate_reasons():
    free = {"holds": [], "ollama_models": [], "mem_used_mib": 1200.0}
    assert runner.gate_reasons(free, CFG) == []
    assert runner.gate_reasons({**free, "ollama_models": ["qwen3:8b"]}, CFG)
    assert runner.gate_reasons({**free, "ollama_models": None}, CFG)
    assert runner.gate_reasons({**free, "mem_used_mib": 7162.0}, CFG)
    assert runner.gate_reasons({**free, "holds": ["HOLD-t12"]}, CFG)


def test_gpu_csv_stats(tmp_path):
    p = tmp_path / "gpu.csv"
    p.write_text(
        "timestamp, memory.used [MiB], power.draw [W], utilization.gpu [%], temperature.gpu\n"
        "2026/10/09 15:00:00.000, 1000 MiB, 30.00 W, 0 %, 40\n"
        "2026/10/09 15:00:01.000, 5000 MiB, 300.00 W, 90 %, 60\n"
        "2026/10/09 15:00:02.000, 4000 MiB, 270.00 W, 80 %, 61\n")
    s = runner.gpu_csv_stats(p)
    assert s["samples"] == 3 and s["peak_mem_mib"] == 5000 and s["idle_mem_mib"] == 1000
    assert s["energy_j"] == 600.0 and s["peak_over_idle_mib"] == 4000


def _tar_with(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_cpu_job_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "machine_record", lambda cfg: {"label": "test"})
    monkeypatch.setattr(runner, "ollama_models", lambda cfg: [])
    root = tmp_path / "q"
    q = runner.Queue(root)
    q.cfg["py"] = sys.executable
    (root / "src" / "abc123.tar").write_bytes(_tar_with(
        {"hello.py": "import os,sys\nopen(os.path.join(sys.argv[1],'r.txt'),'w').write('ok')\n"}))
    job = {"id": "20261009-150000-x0-echo", "key": "X0-echo", "test": "X0", "arm": "echo",
           "repo_ref": "abc123", "cwd": "{src}", "cmd": ["{py}", "hello.py", "{out}"], "gpu": False,
           "timeout_min": 1, "outputs": ["{out}/r.txt"], "after_ok": []}
    jp = root / "incoming" / f"{job['id']}.json"
    jp.write_text(json.dumps(job))
    q.run_job(jp)
    st = json.loads((root / "done" / f"{job['id']}.status.json").read_text())
    assert st["ok"] and st["exit_code"] == 0
    assert (root / "done" / jp.name).exists()

    # a dependent job of a failed key goes to failed without running
    bad = {**job, "id": "20261009-150001-x0-dep", "key": "X0-dep", "after_ok": ["X0-missing"]}
    bp = root / "incoming" / f"{bad['id']}.json"
    bp.write_text(json.dumps(bad))
    q.run_job(bp)
    st = json.loads((root / "failed" / f"{bad['id']}.status.json").read_text())
    assert not st["ok"] and "not run" in st["reason"]

    # missing output and nonzero exit both fail
    fail = {**job, "id": "20261009-150002-x0-fail", "key": "X0-fail", "cmd": ["{py}", "-c", "raise SystemExit(3)"]}
    fp = root / "incoming" / f"{fail['id']}.json"
    fp.write_text(json.dumps(fail))
    q.run_job(fp)
    st = json.loads((root / "failed" / f"{fail['id']}.status.json").read_text())
    assert st["exit_code"] == 3 and "sanity gate" in st["reason"] and "missing outputs" in st["reason"]
    assert "X0-echo" in q.status() or "x0-echo" in q.status()


def test_hold_released_only_after_busy_then_idle(tmp_path, monkeypatch):
    q = runner.Queue(tmp_path / "q")
    hold = q.root / "HOLD-t12"
    hold.write_text(json.dumps({"reason": "t12", "release_after_idle_min": 0, "require_busy_first": True}))
    readings = iter([
        {"holds": ["HOLD-t12"], "ollama_models": [], "mem_used_mib": 500.0},          # quiet before T12
        {"holds": ["HOLD-t12"], "ollama_models": ["qwen3:8b"], "mem_used_mib": 9000.0},  # T12 runs
        {"holds": ["HOLD-t12"], "ollama_models": [], "mem_used_mib": 500.0},          # T12 done
    ])
    monkeypatch.setattr(q, "gate_reading", lambda: next(readings))
    assert q.observe()["holds"] == ["HOLD-t12"]
    assert q.observe()["holds"] == ["HOLD-t12"]
    assert "seen_busy_at" in json.loads(hold.read_text())
    assert q.observe()["holds"] == []
    assert list(q.root.glob("HOLD-t12.released-*"))
