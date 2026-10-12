"""Freezes the C9 inputs (plan/c9.md, notes of 2026-10-12).

    python bench/c9/freeze.py --dir .cache/c9

Reads the Gretel parquet files at revision 7b844d167385 (downloaded into
--dir, hash-checked), and writes:

- .cache/c9/docs.jsonl: the 476 German rows of the `test` split, sorted on the
  integer `index`; per row the text, the raw gold spans exactly as Gretel
  ships them (offsets, label, covered text) and the document type. Planned
  for cases/c9/ (plan/c9.md); kept out of git until GitHub's push protection
  has been cleared for the invented provider-format API keys in two
  documents (plan/c9.md, notes of 2026-10-12); copied to the workstation as
  D:\\bench-data\\c9\\docs.jsonl
- cases/c9/manifest.json: file hashes, row count, ids, the 40 audit ids, the
  label inventory of the `train` split with counts
- reference/c9/labels.json is written by hand (the mapping is a decision,
  not a computation); this script checks that it covers every train label

Audit sample: rows sorted on int(index); the first 40 positions of
numpy default_rng(20261009).permutation(476), kept in id order.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REVISION = "7b844d16738527a04264f50214cb426a4cea0897"
FILES = {
    "test": ("data/test-00000-of-00001.parquet", "014e1057978f030fce4f4cad7c93b9e3377fd0ded713d2bf4c7de00e3e3c8c72"),
    "train": ("data/train-00000-of-00001.parquet", "89bfe27e37e50098347f47f974e5e43f64a521f38171b618600ea7aa9b4fa430"),
}
SEED = 20261009
N_AUDIT = 40


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def spans_of(row):
    sp = row["pii_spans"]
    return json.loads(sp) if isinstance(sp, str) else list(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, ".cache", "c9"))
    a = ap.parse_args()
    paths = {}
    for split, (name, want) in FILES.items():
        p = os.path.join(a.dir, os.path.basename(name))
        if sha(p) != want:
            sys.exit(f"{p} does not match the SHA-256 in bench/c9/freeze.py")
        paths[split] = p

    train_labels = collections.Counter()
    for r in pq.read_table(paths["train"], columns=["pii_spans"]).to_pylist():
        for s in spans_of(r):
            train_labels[s["label"]] += 1

    labels = json.load(open(os.path.join(ROOT, "reference", "c9", "labels.json"), encoding="utf-8"))
    mapped = set(labels["map"]) | set(labels["dropped"])
    missing = sorted(set(train_labels) - mapped)
    if missing:
        sys.exit(f"reference/c9/labels.json does not cover the train labels {missing}")

    rows = [r for r in pq.read_table(paths["test"]).to_pylist() if r["language"] == "German"]
    rows.sort(key=lambda r: int(r["index"]))
    test_only = sorted({s["label"] for r in rows for s in spans_of(r)} - set(train_labels))
    out = []
    for r in rows:
        text = r["generated_text"]
        spans = []
        for s in spans_of(r):
            spans.append({"start": s["start"], "end": s["end"], "label": s["label"], "text": text[s["start"]:s["end"]]})
        spans.sort(key=lambda s: (s["start"], s["end"], s["label"]))
        out.append({"id": str(r["index"]), "document_type": r["document_type"], "expanded_type": r["expanded_type"],
                    "text": text, "spans": spans})
    os.makedirs(os.path.join(ROOT, "cases", "c9"), exist_ok=True)
    os.makedirs(os.path.join(a.dir), exist_ok=True)
    docs = os.path.join(a.dir, "docs.jsonl")
    with open(docs, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids = [r["id"] for r in out]
    perm = np.random.default_rng(SEED).permutation(len(ids))
    audit = sorted((ids[i] for i in perm[:N_AUDIT]), key=int)
    man = {
        "design": "plan/c9.md",
        "dataset": "gretelai/synthetic_pii_finance_multilingual",
        "licence": "Apache-2.0",
        "revision": REVISION,
        "files": {k: {"file": v[0], "sha256": v[1]} for k, v in FILES.items()},
        "filter": "split test, language == 'German'",
        "rows": len(out),
        "docs_sha256": sha(docs),
        "ids": ids,
        "audit": {"n": N_AUDIT, "seed": SEED,
                  "rule": "rows sorted on int(index); first 40 positions of default_rng(20261009).permutation(476); kept in id order",
                  "ids": audit},
        "train_label_counts": dict(train_labels.most_common()),
        "test_labels_not_in_train": test_only,
        "gold_spans": sum(len(r["spans"]) for r in out),
        "chars": sum(len(r["text"]) for r in out),
    }
    json.dump(man, open(os.path.join(ROOT, "cases", "c9", "manifest.json"), "w", encoding="utf-8", newline="\n"),
              indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("rows", "gold_spans", "chars", "test_labels_not_in_train")}))


if __name__ == "__main__":
    main()
