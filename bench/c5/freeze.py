"""Freezes the C5 inputs (plan/c5.md, notes of 2026-10-12).

    python bench/c5/freeze.py --dir .cache/c5

Reads FLORES+ devtest deu_Latn.jsonl and eng_Latn.jsonl at revision
e707e62e762a (gated: downloaded with an accepted Hugging Face account into
--dir, never committed) and writes cases/c5/manifest.json: file hashes, the
1,012 ids and one SHA-256 per sentence and language, so a run can prove it
read the same text without the text being published.
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REVISION = "e707e62e762a9e2faca3933aa315f4665cc3cd8f"
LANGS = ("deu_Latn", "eng_Latn")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, ".cache", "c5"))
    a = ap.parse_args()
    man = {"design": "plan/c5.md", "dataset": "openlanguagedata/flores_plus", "licence": "CC BY-SA 4.0",
           "revision": REVISION, "split": "devtest", "gated": True,
           "access": "accepted on Hugging Face by the bench maintainer's account, first successful download 2026-10-12",
           "files": {}, "sentences": {}}
    rows = {}
    for lang in LANGS:
        p = os.path.join(a.dir, f"{lang}.jsonl")
        raw = open(p, "rb").read()
        man["files"][lang] = {"file": f"devtest/{lang}.jsonl", "sha256": sha_bytes(raw)}
        rows[lang] = {int(r["id"]): r["text"] for r in (json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip())}
    ids = sorted(rows[LANGS[0]])
    assert ids == sorted(rows[LANGS[1]]) and len(ids) == 1012, "expected 1,012 aligned ids"
    man["ids"] = ids
    for lang in LANGS:
        man["sentences"][lang] = [sha_bytes(rows[lang][i].encode("utf-8")) for i in ids]
    os.makedirs(os.path.join(ROOT, "cases", "c5"), exist_ok=True)
    json.dump(man, open(os.path.join(ROOT, "cases", "c5", "manifest.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"{len(ids)} sentence pairs, files {[man['files'][l]['sha256'][:12] for l in LANGS]}")


if __name__ == "__main__":
    main()
