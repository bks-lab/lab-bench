"""Fetch the C1 retrieval sets at their pinned revisions and verify them.

Usage:
  python cases/c1/fetch.py --out D:\\bench-data\\c1            # download + verify
  python cases/c1/fetch.py --out DIR --verify-only            # verify an existing copy
  python cases/c1/fetch.py --out DIR --write-manifest         # (maintainer) rewrite manifest.json

The texts are not in git (MIRACL is CC BY-SA 4.0 and large, GermanDPR is
CC BY 4.0); only ids, revisions, counts and hashes are, in manifest.json.
"""
import argparse, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")

SETS = {
    "miracl": {
        "repo": "mteb/MIRACLRetrievalHardNegatives",
        # The plan's revision 95c8db7d4a6e is gone from the hub (history rewritten
        # 2025-09-10); the head of main on 2026-10-09 is used, see plan/c1.md.
        "planned_revision": "95c8db7d4a6e9c1d8a60601afd63d553ae20a2eb",
        "revision": "332a9acb49f5e83d5397683f79d23e588f685916",
        "split": "dev",
        "licence": "CC BY-SA 4.0",
        "files": {"corpus": "de-corpus/dev-00000-of-00001.parquet",
                  "queries": "de-queries/dev-00000-of-00001.parquet",
                  "qrels": "de-qrels/dev-00000-of-00001.parquet"},
    },
    "germandpr": {
        "repo": "mteb/GermanDPR",
        "planned_revision": "64a4860e55ba6d8fcb923d5306d08e08b1c72794",
        "revision": "64a4860e55ba6d8fcb923d5306d08e08b1c72794",
        "split": "test",
        "licence": "CC BY 4.0",
        "files": {"corpus": "corpus/test-00000-of-00001.parquet",
                  "queries": "queries/test-00000-of-00001.parquet",
                  "qrels": "qrels/test-00000-of-00001.parquet"},
    },
}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_sha(path):
    """SHA-256 over the loaded rows: one canonical JSON line per row, sorted."""
    import pandas as pd
    df = pd.read_parquet(path)
    lines = sorted(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                              ensure_ascii=False, sort_keys=True) for r in df.to_dict("records"))
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8") + b"\n")
    return h.hexdigest(), len(lines)


def verify(out, manifest):
    ok = True
    for name, spec in manifest["sets"].items():
        for part, meta in spec["files"].items():
            p = os.path.join(out, name, part + ".parquet")
            if not os.path.exists(p):
                print("missing", p); ok = False; continue
            if file_sha(p) != meta["sha256"]:
                print("hash mismatch", p); ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        for name, s in SETS.items():
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            for part, f in s["files"].items():
                dst = os.path.join(a.out, name, part + ".parquet")
                if not os.path.exists(dst):
                    shutil.copy(hf_hub_download(s["repo"], f, repo_type="dataset", revision=s["revision"]), dst)
    if a.write_manifest:
        m = {"test": "C1", "note": "texts not committed; hashes of the parquet files and of the loaded rows", "sets": {}}
        for name, s in SETS.items():
            e = {k: s[k] for k in ("repo", "planned_revision", "revision", "split", "licence")}
            e["files"] = {}
            for part, f in s["files"].items():
                p = os.path.join(a.out, name, part + ".parquet")
                rh, n = rows_sha(p)
                e["files"][part] = {"hub_path": f, "sha256": file_sha(p), "rows_sha256": rh, "rows": n}
            m["sets"][name] = e
        with open(MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2, ensure_ascii=False); fh.write("\n")
        print("wrote", MANIFEST)
        return 0
    with open(MANIFEST, encoding="utf-8") as fh:
        m = json.load(fh)
    ok = verify(a.out, m)
    print("verify", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
