"""C7 generator: synthetic needles in German text (plan/c7.md).

    python bench/gen-c7.py --parquet .cache/c6/test.parquet         writes cases/c7/trials.jsonl + manifest.json
    python bench/gen-c7.py --parquet ... --check                    rebuilds and compares with the committed files

The model-independent part of every prompt is fixed here and committed
before any run: for each of 3 lengths x 3 tasks x 3 depths x 20 trials the
project names, the codes, where the needles go and the seeded order of the
haystack passages. The model-dependent part, how many characters of that
haystack make the target length in the model's own tokens, is measured by
the runner (bench/c7/run_c7.py) with dry calls, and every real prompt
length is recorded.

Haystack: the distinct GermanQuAD test passages (CC BY 4.0, the C6 file,
hash-checked), without the passages that contain a run of seven digits (a
code could be confused with them), concatenated in an order drawn per trial
from numpy default_rng(20261009) and cut at a passage or sentence end.
Needles are whole sentences inserted between two sentences of the haystack
at a share of its characters (depth).
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_SHA = "c154513e016e73e4e32b46b48010f24a435ad7b0e96cc3858cfc20c7d29649bc"
SEED = 20261009
LENGTHS = [4096, 16384, 32768]
TASKS = ["single", "multikey", "multivalue"]
DEPTHS = [0.1, 0.5, 0.9]
TRIALS = 20

# Invented project names (no real project, company or place of that exact name is meant).
NAMES = """Abendrot Ahornfeld Bernsteinweg Birkenhain Blaufalke Brückenlicht Dämmerstein Eichenpfad Eisvogel
Erlenmoor Falkenblick Feldlerche Felsquelle Fichtenhang Flussperle Funkenflug Glockenturm Grauwacke Habichtsruh
Hafenblick Heidekamm Himmelsleiter Hirschsprung Holunderbach Kiefernwind Kranichzug Kupferhain Lerchenfeld
Lindenbogen Löwenzahn Mondsichel Moosgrund Nebelhorn Nordwind Ostwall Pappelallee Quellgrund Rabenfels
Rosenbogen Sandläufer Schilfrohr Seidenfaden Silberdistel Sonnentau Steinadler Sternschnuppe Tannenzapfen
Uferschwalbe Wacholder Weidenkorb Wellenkamm Wiesengrund Wolkenbruch Zedernholz Zinnober Zugvogel""".split()

NEEDLE = "Der Freigabecode für Projekt {name} lautet {code}."
QUESTION = {
    "single": "Wie lautet der Freigabecode für Projekt {name}?",
    "multikey": "Wie lautet der Freigabecode für Projekt {name}?",
    "multivalue": "Wie lauten die drei Freigabecodes für Projekt {name}? Nenne alle drei.",
}
PROMPT = ("Im folgenden Text sind Freigabecodes für Projekte versteckt. Lies den Text und beantworte danach "
          "die Frage. Antworte nur mit dem Code oder den Codes, ohne weiteren Text.\n\n"
          "<text>\n{haystack}\n</text>\n\nFrage: {question}")
CODE_RE = re.compile(r"(?<!\d)\d{7}(?!\d)")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def passages(parquet):
    if sha(parquet) != PARQUET_SHA:
        sys.exit(f"{parquet} does not match plan/c6.md")
    import pyarrow.parquet as pq  # noqa: PLC0415
    rows = sorted(pq.read_table(parquet).to_pylist(), key=lambda r: int(r["id"]))
    seen, out = set(), []
    for r in rows:
        c = r["context"]
        if c in seen or CODE_RE.search(c):
            continue
        seen.add(c)
        out.append(c)
    return out


def trials(n_passages):
    """Every trial of the design, model-independent, in a fixed order."""
    rng = np.random.default_rng(SEED)
    out = []
    for L in LENGTHS:
        for task in TASKS:
            for depth in DEPTHS:
                for t in range(TRIALS):
                    k = 4 if task == "multikey" else 1
                    names = [NAMES[i] for i in rng.choice(len(NAMES), size=k, replace=False)]
                    ncodes = 3 if task == "multivalue" else k
                    codes = []
                    while len(codes) < ncodes:
                        c = str(int(rng.integers(1_000_000, 10_000_000)))
                        if c not in codes:
                            codes.append(c)
                    if task == "single":
                        needles = [{"name": names[0], "code": codes[0], "depth": depth, "asked": True}]
                    elif task == "multikey":
                        others = [float(x) for x in rng.uniform(0.02, 0.98, size=3)]
                        needles = [{"name": names[0], "code": codes[0], "depth": depth, "asked": True}] + \
                                  [{"name": names[i], "code": codes[i], "depth": round(others[i - 1], 4), "asked": False} for i in (1, 2, 3)]
                    else:
                        others = [float(x) for x in rng.uniform(0.02, 0.98, size=2)]
                        needles = [{"name": names[0], "code": codes[0], "depth": depth, "asked": True}] + \
                                  [{"name": names[0], "code": codes[i], "depth": round(others[i - 1], 4), "asked": True} for i in (1, 2)]
                    order = [int(x) for x in rng.permutation(n_passages)]
                    out.append({"id": f"{L // 1024}k-{task}-d{int(depth * 100)}-{t:02d}", "length": L, "task": task,
                                "depth": depth, "trial": t, "name": names[0], "needles": needles,
                                "asked_codes": [n["code"] for n in needles if n["asked"]],
                                "other_codes": [n["code"] for n in needles if not n["asked"]],
                                "question": QUESTION[task].format(name=names[0]), "order": order})
    return out


def haystack(trial, pool, chars):
    """The haystack of `chars` characters (cut at a sentence end) with the needles inserted."""
    text, i = "", 0
    while len(text) < chars:
        text += pool[trial["order"][i % len(pool)]] + "\n\n"
        i += 1
    cut = text[:chars]
    m = list(re.finditer(r"[.!?]\s", cut))
    if m and m[-1].end() > 0.9 * chars:
        cut = cut[: m[-1].end()]
    ends = [0] + [x.end() for x in re.finditer(r"(?<=[.!?])\s+|\n\n", cut)] + [len(cut)]
    inserts = []
    for n in trial["needles"]:
        target = n["depth"] * len(cut)
        pos = min(ends, key=lambda e: abs(e - target))
        inserts.append((pos, NEEDLE.format(name=n["name"], code=n["code"])))
    for pos, s in sorted(inserts, key=lambda x: -x[0]):
        cut = cut[:pos] + s + " " + cut[pos:]
    return cut


def prompt(trial, pool, chars):
    return PROMPT.format(haystack=haystack(trial, pool, chars), question=trial["question"])


def judge(answer, trial):
    """Right if every asked code appears as a 7-digit string and no other needle code does."""
    found = set(CODE_RE.findall(answer or ""))
    return all(c in found for c in trial["asked_codes"]) and not any(c in found for c in trial["other_codes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    pool = passages(a.parquet)
    tr = trials(len(pool))
    cdir = os.path.join(ROOT, "cases", "c7")
    os.makedirs(cdir, exist_ok=True)
    body = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tr)
    path = os.path.join(cdir, "trials.jsonl")
    if a.check:
        same = open(path, encoding="utf-8").read() == body
        print("trials.jsonl", "matches" if same else "DIFFERS")
        sys.exit(0 if same else 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    man = {"design": "plan/c7.md", "generator": "bench/gen-c7.py", "seed": SEED, "lengths": LENGTHS, "tasks": TASKS,
           "depths": DEPTHS, "trials_per_cell": TRIALS, "prompts": len(tr),
           "haystack_source": {"file": "deepset/germanquad plain_text/test/0000.parquet", "sha256": PARQUET_SHA,
                               "passages": len(pool), "rule": "distinct contexts in id order, without those containing seven digits in a row"},
           "trials_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
           "needle": NEEDLE, "questions": QUESTION, "prompt": PROMPT}
    with open(os.path.join(cdir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"prompts": len(tr), "passages": len(pool), "sha256": man["trials_sha256"]}))


if __name__ == "__main__":
    main()
