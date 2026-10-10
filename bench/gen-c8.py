"""C8 generator: the fixed workloads of the throughput test (plan/c8.md).

    python bench/gen-c8.py --parquet .cache/c6/test.parquet          writes cases/c8/
    python bench/gen-c8.py --parquet ... --check                     rebuilds and compares

- chat-short, chat-long: 68 prompts each (4 warm-up, 64 timed), a German
  instruction to summarise at length plus German business-like text from the
  GermanQuAD test passages (CC BY 4.0, the C6 file), the passages of each
  prompt in a seeded order so no two prompts share text after the
  instruction. The runner measures per model how many characters make 512
  and 4,096 tokens (bench/c8/run_c8.py) and cuts each prompt's text there.
- decide: the first 500 of a numpy default_rng(20261009) permutation of the
  2,974 MASSIVE de requests of requests/route1/requests.jsonl (file order),
  their `scenario` question rendered as the raw one-token prompt of
  bench/run-local.mjs for every template a C8 arm uses
  (bench/c8/decide-prompts.mjs).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_SHA = "c154513e016e73e4e32b46b48010f24a435ad7b0e96cc3858cfc20c7d29649bc"
SEED = 20261009
CHAT = {"chat-short": 512, "chat-long": 4096}
WARMUP, TIMED, DECIDE = 4, 64, 500
INSTRUCTION = ("Fasse den folgenden Text ausführlich auf Deutsch zusammen. Schreibe mindestens 600 Wörter, "
               "gehe auf jeden Abschnitt einzeln ein und lass kein Detail aus.\n\nText:\n{text}")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def passages(parquet):
    if sha_bytes(open(parquet, "rb").read()) != PARQUET_SHA:
        sys.exit(f"{parquet} does not match plan/c6.md")
    import pyarrow.parquet as pq  # noqa: PLC0415
    rows = sorted(pq.read_table(parquet).to_pylist(), key=lambda r: int(r["id"]))
    seen, out = set(), []
    for r in rows:
        if r["context"] not in seen:
            seen.add(r["context"])
            out.append(r["context"])
    return out


def chat_text(order, pool, chars):
    text, i = "", 0
    while len(text) < chars:
        text += pool[order[i % len(order)]] + "\n\n"
        i += 1
    return text[:chars].rstrip()


def build(parquet):
    pool = passages(parquet)
    rng = np.random.default_rng(SEED)
    chat = []
    for w, tokens in CHAT.items():
        for i in range(WARMUP + TIMED):
            chat.append({"workload": w, "target_tokens": tokens, "idx": i, "warmup": i < WARMUP,
                         "order": [int(x) for x in rng.permutation(len(pool))[:40]]})
    reqs = [json.loads(l) for l in open(os.path.join(ROOT, "requests", "route1", "requests.jsonl"), encoding="utf-8")]
    de = [r["line"] for r in reqs if r["line"].startswith("massive") and r["lang"] == "de" and "scenario" in r["request"]["questions"]]
    perm = np.random.default_rng(SEED).permutation(len(de))
    lines = [de[i] for i in perm[:DECIDE + WARMUP]]
    out = subprocess.run(["node", os.path.join(ROOT, "bench", "c8", "decide-prompts.mjs")], input=json.dumps(lines),
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    decide = [{**d, "warmup": i >= DECIDE} for i, d in enumerate(json.loads(out))]
    return pool, chat, decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    pool, chat, decide = build(a.parquet)
    cdir = os.path.join(ROOT, "cases", "c8")
    os.makedirs(cdir, exist_ok=True)
    bodies = {"chat.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chat),
              "decide.jsonl": "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decide)}
    if a.check:
        bad = [f for f, b in bodies.items() if open(os.path.join(cdir, f), encoding="utf-8").read() != b]
        print("differs: " + ", ".join(bad) if bad else "cases/c8 matches")
        sys.exit(1 if bad else 0)
    for f, b in bodies.items():
        with open(os.path.join(cdir, f), "w", encoding="utf-8", newline="\n") as fo:
            fo.write(b)
    man = {"design": "plan/c8.md", "generator": "bench/gen-c8.py", "seed": SEED, "instruction": INSTRUCTION,
           "chat": {"workloads": CHAT, "warmup": WARMUP, "timed": TIMED, "passages": len(pool),
                    "source": {"file": "deepset/germanquad plain_text/test/0000.parquet", "sha256": PARQUET_SHA}},
           "decide": {"source": "requests/route1/requests.jsonl, MASSIVE de, scenario question", "timed": DECIDE,
                      "warmup": WARMUP, "rule": "first 500 (then 4 warm-up) of default_rng(20261009).permutation(2974) over the de lines in file order",
                      "templates": ["chatml", "gemma4", "mistral", "granite", "harmony"]},
           "files_sha256": {f: sha_bytes(b.encode("utf-8")) for f, b in bodies.items()}}
    with open(os.path.join(cdir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"chat": len(chat), "decide": len(decide), "files": man["files_sha256"]}))


if __name__ == "__main__":
    main()
