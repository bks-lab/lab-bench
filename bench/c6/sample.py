"""Freezes the C6 inputs (plan/c6.md) and computes the lexical baseline.

    python bench/c6/sample.py --parquet .cache/c6/test.parquet

- checks the GermanQuAD test parquet against the SHA-256 in plan/c6.md
- writes cases/c6/sample.jsonl: the 1,000 questions the LLM arms get, drawn by
  sorting the 2,204 rows on `id` (as integers; the ids are decimal strings)
  and taking the first 1,000 positions of a permutation from
  numpy default_rng(20261009); rows kept in id order
- writes cases/c6/manifest.json (hashes, ids)
- writes results/c6/baseline/result.jsonl: the lexical overlap baseline of
  results/c6/method.md for all 2,204 questions (the sample is scored from it)
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import squad  # noqa: E402

PARQUET_SHA = "c154513e016e73e4e32b46b48010f24a435ad7b0e96cc3858cfc20c7d29649bc"
SEED = 20261009
N = 1000

# Function words left out when counting shared content words (fixed for the baseline).
STOP = set("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes auch auf aus
bei bin bis bist da damit dann das dass dem den denn der des dessen die dies diese diesem diesen dieser dieses
doch dort du durch ein eine einem einen einer eines er es etwas für gegen gewesen hab habe haben hat hatte
hätte ich ihr im in ist ja jede jedem jeden jeder jedes jetzt kann kein keine können man mit muss nach nicht
nichts noch nun nur ob oder ohne sehr sein seine sich sie sind so solche soll sollte sondern über um und uns
unter vom von vor war waren was weil welche welchem welchen welcher welches wenn wer werden wie wieder wir
wird wo wurde wurden zu zum zur zwischen wann wieso warum weshalb wessen wem wen woher wohin womit wodurch
""".split())


def words(s):
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOP]


def sentences(ctx):
    """Split on line breaks and on . ! ? followed by white space."""
    out = []
    for block in ctx.split("\n"):
        for s in re.split(r"(?<=[.!?])\s+", block):
            if s.strip():
                out.append(s.strip())
    return out


def baseline(ctx, q):
    qw = set(words(q))
    best, best_n = "", -1
    for s in sentences(ctx):
        n = len(qw & set(words(s)))
        if n > best_n:  # strict: the first sentence wins a tie
            best, best_n = s, n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    a = ap.parse_args()
    if hashlib.sha256(open(a.parquet, "rb").read()).hexdigest() != PARQUET_SHA:
        sys.exit("GermanQuAD parquet does not match the SHA-256 of plan/c6.md")
    import pandas as pd  # noqa: PLC0415
    df = pd.read_parquet(a.parquet)
    rows = []
    for r in df.to_dict("records"):
        ans = r["answers"]
        rows.append({"id": str(r["id"]), "context": r["context"], "question": r["question"],
                     "answers": [str(x) for x in list(ans["text"])],
                     "answer_start": [int(x) for x in list(ans["answer_start"])]})
    rows.sort(key=lambda r: int(r["id"]))
    perm = np.random.default_rng(SEED).permutation(len(rows))
    pick = sorted(int(i) for i in perm[:N])
    sample = [rows[i] for i in pick]
    cdir = os.path.join(ROOT, "cases", "c6")
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "sample.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sample_sha = hashlib.sha256(open(os.path.join(cdir, "sample.jsonl"), "rb").read()).hexdigest()
    manifest = {"design": "plan/c6.md", "dataset": "deepset/germanquad", "licence": "CC BY 4.0",
                "revision": "refs/convert/parquet a2f3a59f0be843fc305d0417d7292ef0b1a66884",
                "file": "plain_text/test/0000.parquet", "file_sha256": PARQUET_SHA, "rows": len(rows),
                "sample": {"n": N, "seed": SEED, "rule": "rows sorted on int(id); first 1,000 positions of default_rng(20261009).permutation(2204); kept in id order",
                           "ids": [r["id"] for r in sample], "sha256": sample_sha}}
    with open(os.path.join(cdir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    bdir = os.path.join(ROOT, "results", "c6", "baseline")
    os.makedirs(bdir, exist_ok=True)
    ids = set(manifest["sample"]["ids"])
    f1s = []
    with open(os.path.join(bdir, "result.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            p = baseline(r["context"], r["question"])
            x = squad.best(squad.f1, p, r["answers"])
            if r["id"] in ids:
                f1s.append(x)
            f.write(json.dumps({"id": r["id"], "answer": p, "in_sample": r["id"] in ids}, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "sample": len(sample), "sample_sha256": sample_sha,
                      "baseline_f1_sample": round(float(np.mean(f1s)), 4)}))


if __name__ == "__main__":
    main()
