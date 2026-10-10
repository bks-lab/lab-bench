#!/usr/bin/env python3
"""Build the Hugging Face dataset bks-lab/xrechnung-invoice-fields from C3:
34 XRechnung test invoices as page images, PDF and CII XML, the twelve gold
fields per invoice, and the recorded outputs of every finished model arm.

    python3 hf/c3/build.py --out <dir>                 # commit = HEAD, clean tree
    python3 hf/c3/build.py --out <dir> --commit <sha>
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/bks-lab/lab-bench"
FIELDS = ["invoice_number", "issue_date", "due_date", "seller_name", "seller_vat_id",
          "buyer_name", "iban", "net_total", "vat_total", "gross_total", "amount_due",
          "currency"]
LICENCES = ["LICENSE-xrechnung-testsuite", "LICENSE-xrechnung-visualization",
            "LICENSE-SourceSerifPro-OFL.txt"]
HUB_ID = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)\b")


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_file(root, arm):
    """The scored run of an arm: results/c3/<arm with + as ->/*-run1.jsonl,
    not the OCR side file, the latest if there are several."""
    d = root / "results/c3" / arm.replace("+", "-")
    files = sorted(p for p in d.glob("*-run1.jsonl") if not p.name.endswith("-ocr.jsonl"))
    if not files:
        raise SystemExit(f"no run file for arm {arm} in {d}")
    return files[-1]


def build(root, out, commit):
    root, out = Path(root), Path(out)
    cases = root / "cases/c3"
    gold = jsonl(root / "reference/c3/gold.jsonl")

    pages = []
    for g in gold:
        files = sorted((cases / "png").glob(f"{g['id']}-*.png"),
                       key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        for p in files:
            (out / "pages").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / "pages" / p.name)
            pages.append({"file_name": p.name, "invoice_id": g["id"],
                          "page": int(p.stem.rsplit("-", 1)[1]), "pages": len(files),
                          **{k: g.get(k) for k in FIELDS}})
    write_jsonl(out / "pages/metadata.jsonl", pages)

    invoices = []
    for g in gold:
        for kind in ("xml", "pdf"):
            src = cases / kind / f"{g['id']}.{kind}"
            (out / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / kind / src.name)
        invoices.append({"id": g["id"], "pages": sum(p["invoice_id"] == g["id"] for p in pages),
                         "xml_file": f"xml/{g['id']}.xml", "pdf_file": f"pdf/{g['id']}.pdf",
                         "gold": {k: g.get(k) for k in FIELDS}})
    write_jsonl(out / "invoices.jsonl", invoices)

    summary = json.loads((root / "results/c3/summary.json").read_text(encoding="utf-8"))
    preds = []
    for a in summary["arms"]:
        if not a.get("finished"):
            continue
        m = HUB_ID.search(a.get("model", ""))
        for r in jsonl(run_file(root, a["arm"])):
            preds.append({
                "invoice_id": r["id"], "arm": a["arm"], "model": a.get("model"),
                "hub_model": m.group(1) if m else None, "kind": a.get("kind"),
                "reference_arm": bool(a.get("reference_arm")),
                "quant": a.get("quant"), "licence": a.get("licence"),
                "pred": {k: (r.get("pred") or {}).get(k) for k in FIELDS},
                "correct": {k: bool((r.get("correct") or {}).get(k)) for k in FIELDS},
                "n_correct": sum(bool((r.get("correct") or {}).get(k)) for k in FIELDS),
                "seconds_model": r.get("seconds_model"), "seconds_ocr": r.get("seconds_ocr"),
                "error": r.get("error"),
            })
    write_jsonl(out / "predictions.jsonl", preds)

    for f in LICENCES:
        shutil.copy2(cases / f, out / f)
    card = (root / "hf/c3/CARD.md").read_text(encoding="utf-8")
    best = max((a for a in summary["arms"] if a.get("finished") and not a.get("reference_arm")), key=lambda a: a["quality"]["value"])
    card = (card.replace("{{COMMIT}}", commit).replace("{{REPO}}", REPO)
            .replace("{{N_INVOICES}}", str(len(invoices))).replace("{{N_PAGES}}", str(len(pages)))
            .replace("{{REGEX}}", f"{summary['regex_baseline']:.1%}")
            .replace("{{BEST_ARM}}", best["arm"]).replace("{{BEST}}", f"{best['quality']['value']:.1%}"))
    (out / "README.md").write_text(card, encoding="utf-8")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description="Build the Hugging Face dataset bks-lab/xrechnung-invoice-fields.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if a.commit is None:
        if git(root, "status", "--porcelain", "--", "cases/c3", "reference/c3", "results/c3", "hf"):
            sys.exit("C3 inputs or hf/ have uncommitted changes; commit first or pass --commit")
        a.commit = git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        sys.exit(f"--commit must be a full 40-character SHA, got {a.commit!r}")
    build(root, a.out, a.commit)
    print(f"built {a.out} from {a.commit}")


if __name__ == "__main__":
    main()
