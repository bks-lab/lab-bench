"""Fetch FLEURS de_de test at its pinned revision and verify it.

Usage:
  python cases/c2/fetch.py --out D:\\bench-data\\c2           # download, unpack, verify
  python cases/c2/fetch.py --out DIR --verify-only
  python cases/c2/fetch.py --out DIR --write-manifest        # (maintainer) rewrite utterances.tsv

Audio (about 570 MB) is not in git. cases/c2/utterances.tsv holds per
utterance: FLEURS row id (sentence id, the bootstrap cluster), file name,
duration in seconds and SHA-256 of the wav. manifest.json holds the
revision and the hash of test.tsv (which carries the reference text).
"""
import argparse, csv, hashlib, json, os, shutil, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "google/fleurs"
REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FILES = ["data/de_de/test.tsv", "data/de_de/audio/test.tar.gz"]
COLS = ["id", "file", "raw_transcription", "transcription", "chars", "num_samples", "gender"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [dict(zip(COLS, r)) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        for f in FILES:
            dst = os.path.join(a.out, f.split("/")[-1])
            if not os.path.exists(dst):
                shutil.copy(hf_hub_download(REPO, f, repo_type="dataset", revision=REV), dst)
        if not os.path.isdir(os.path.join(a.out, "audio", "test")):
            with tarfile.open(os.path.join(a.out, "test.tar.gz")) as t:
                t.extractall(os.path.join(a.out, "audio"))
    rows = read_tsv(os.path.join(a.out, "test.tsv"))
    if a.write_manifest:
        with open(os.path.join(HERE, "utterances.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "file", "duration_s", "sha256"])
            for r in rows:
                w.writerow([r["id"], r["file"], f"{int(r['num_samples']) / 16000:.3f}",
                            sha(os.path.join(a.out, "audio", "test", r["file"]))])
        m = {"test": "C2", "repo": REPO, "config": "de_de", "split": "test", "revision": REV,
             "licence": "CC BY 4.0", "utterances": len(rows), "sentences": len({r["id"] for r in rows}),
             "audio_hours": round(sum(int(r["num_samples"]) for r in rows) / 16000 / 3600, 4),
             "test_tsv_sha256": sha(os.path.join(a.out, "test.tsv")),
             "test_tar_gz_sha256": sha(os.path.join(a.out, "test.tar.gz"))}
        with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2); f.write("\n")
        print("wrote manifest", m)
        return 0
    m = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    ok = sha(os.path.join(a.out, "test.tsv")) == m["test_tsv_sha256"]
    if not ok:
        print("test.tsv hash mismatch")
    with open(os.path.join(HERE, "utterances.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            p = os.path.join(a.out, "audio", "test", r["file"])
            if not os.path.exists(p) or sha(p) != r["sha256"]:
                print("bad audio", r["file"]); ok = False
    print("verify", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
