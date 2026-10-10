#!/usr/bin/env python3
"""Build the Hugging Face dataset bks-lab/lab-bench-results from this
repository: two flat tables of aggregated results plus a dataset card.

Only aggregates leave the repository. No case, request, reference or result
row is copied, in particular none of Jev's per-question outputs, which
LICENSE-JEV-OUTPUTS keeps out of any training use.

    python3 hf/build.py --out /tmp/hf-out            # commit = HEAD, tree must be clean
    python3 hf/build.py --out /tmp/hf-out --commit <sha>
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/bks-lab/lab-bench"

# Columns of data/toolmap.jsonl. Scalar fields of a tool map cell go into
# their own column; every other field of the cell is kept in extra_json.
TOOLMAP_FIELDS = [
    "task_id", "kind", "task", "lang", "metric", "family", "family_label",
    "state", "best_arm", "arms_compared", "value", "baseline", "baseline_name",
    "skill", "n", "answered", "top_family", "mcnemar_vs_top_p",
    "mcnemar_vs_jev_p", "p50_ms_per_decision", "measured", "model_version",
    "reason", "report_url",
]
CELL_SCALARS = {
    "state", "best_arm", "arms_compared", "value", "baseline", "baseline_name",
    "skill", "n", "answered", "top_family", "mcnemar_vs_top_p",
    "mcnemar_vs_jev_p", "p50_ms_per_decision", "measured", "model_version",
    "reason",
}


def url(commit, path):
    return f"{REPO}/blob/{commit}/{path}" if path else None


def toolmap_rows(tm, commit):
    for row in tm["rows"]:
        for family, cell in row["cells"].items():
            out: dict = {k: None for k in TOOLMAP_FIELDS}
            out.update(task_id=row["id"], kind=row.get("kind"), task=row.get("task"),
                       lang=row.get("lang"), metric=row.get("metric"), family=family,
                       family_label=tm["families"].get(family))
            for k in CELL_SCALARS:
                if k in cell:
                    out[k] = cell[k]
            out["report_url"] = url(commit, cell.get("source_report") or row.get("source_report"))
            extra = {k: v for k, v in cell.items() if k not in CELL_SCALARS and k != "source_report"}
            out["extra_json"] = json.dumps(extra, ensure_ascii=False, sort_keys=True)
            yield out


def capability_rows(cap, commit):
    for r in cap["rows"]:
        q, s, e = r.get("quality", {}), r.get("speed", {}), r.get("energy", {})
        ci = q.get("ci95") or [None, None]
        yield {
            "id": r["id"],
            "capability": r["capability"],
            "best_local_model": r.get("best_local_model"),
            "best_by_value": r.get("best_by_value"),
            "licence_class": r.get("licence_class"),
            "metric": q.get("metric"),
            "value": q.get("value"),
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "baseline": q.get("baseline"),
            "baseline_value": q.get("baseline_value"),
            "sanity_reference": q.get("reference"),
            "sanity_reference_ours": q.get("reference_ours"),
            "speed_unit": s.get("unit"),
            "speed_value": s.get("value"),
            "vram_peak_gb": r.get("vram_peak_gb"),
            "energy_unit": e.get("unit"),
            "energy_value": e.get("value"),
            "fits_24gb": r.get("fits_24gb"),
            "tie_note": (r.get("tie") or {}).get("note"),
            "unfinished_arms": r.get("unfinished", []),
            "machine": cap.get("machine"),
            "design_url": url(commit, r.get("design")),
            "report_url": url(commit, r.get("report")),
        }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(root, out, commit):
    root, out = Path(root), Path(out)
    tm = json.loads((root / "results/toolmap.json").read_text(encoding="utf-8"))
    cap = json.loads((root / "results/capabilities.json").read_text(encoding="utf-8"))
    write_jsonl(out / "data/toolmap.jsonl", toolmap_rows(tm, commit))
    write_jsonl(out / "data/capabilities.jsonl", capability_rows(cap, commit))
    card = (root / "hf/CARD.md").read_text(encoding="utf-8")
    card = card.replace("{{COMMIT}}", commit).replace("{{GENERATED}}", tm.get("generated", ""))
    (out / "README.md").write_text(card, encoding="utf-8")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description="Build the Hugging Face dataset bks-lab/lab-bench-results.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", help="commit the tables are built from (default HEAD, clean tree)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    if a.commit is None:
        if git(root, "status", "--porcelain", "--", "results", "hf"):
            sys.exit("results/ or hf/ has uncommitted changes; commit first or pass --commit")
        a.commit = git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        sys.exit(f"--commit must be a full 40-character SHA, got {a.commit!r}")
    build(root, a.out, a.commit)
    print(f"built {a.out} from {a.commit}")


if __name__ == "__main__":
    main()
