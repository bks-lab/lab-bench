#!/usr/bin/env python3
"""Imports the queue jobs of T2, T9 and T10 (plan/t2.md, plan/t9.md, plan/t10.md)
into results/<test>/<arm>/: meta.json and queue-status.json without PC paths
(as bench/score-cap.py import does), and the result rows in compact form,
only the keys the scorers read (bench/t2/score_t2.py): line, question, pass,
choice, error, mass. Options and probabilities are dropped to keep the
repository small; the full rows stay on the Mac in .cache/queue/.

    python bench/import-t.py --queue .cache/queue

The T2 sanity jobs (arm names sanity-<arm>) go to results/t2/sanity-<arm>/. The GPU log is not copied
(energy and peak VRAM are in queue-status.json).
"""
import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("score_cap", ROOT / "bench" / "score-cap.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
KEEP = ("line", "question", "pass", "choice", "error", "mass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(ROOT / ".cache" / "queue"))
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.queue, "*-t*-*"))):
        st_files = glob.glob(os.path.join(d, "*.status.json"))
        if not st_files:
            continue
        status = json.load(open(st_files[0], encoding="utf-8"))
        test, arm = status["test"].lower(), status["arm"]
        if test not in ("t2", "t9", "t10"):
            continue
        name = arm  # the sanity jobs carry "sanity-" in their arm name
        out = ROOT / "results" / test / name
        out.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        sc.wr_json(out / "meta.json", sc.scrub(meta))
        sc.wr_json(out / "queue-status.json", sc.scrub(status))
        date = (meta.get("started") or status["start"])[:10]
        with open(out / f"{date}-run1.jsonl", "w", encoding="utf-8") as fo:
            for line in open(os.path.join(d, "result.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                fo.write(json.dumps({k: r[k] for k in KEEP if k in r}, ensure_ascii=False) + "\n")
        print(f"imported {os.path.basename(d)} -> results/{test}/{name}/")


if __name__ == "__main__":
    main()
