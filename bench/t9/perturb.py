"""Freezes the T9 inputs (plan/t9.md): perturbed MASSIVE de requests.

    python bench/t9/perturb.py

Sample: the 2,974 MASSIVE de requests of requests/route1/requests.jsonl in
file order; the first 1,000 positions of numpy default_rng(20261009).permutation(2974),
kept in file order. Every sampled utterance appears four times:

- clean: unchanged (the same request as in route1)
- typo: keyboard slips. Every word of four or more letters gets one edit
  with probability 0.15 (at least one word per utterance when any is long
  enough): a QWERTZ neighbour substitution, a dropped letter, two swapped
  neighbouring letters or a doubled letter, chosen uniformly
- translit: umlauts and ß as typed without a German layout (ä -> ae, ö -> oe,
  ü -> ue, ß -> ss)
- colloquial: the fixed spoken contractions below, applied where they match,
  and one opening filler from a fixed list, so every variant differs

Each perturbed request keeps the questions and options of its clean request,
so the gold labels of cases/route1/massive-de.jsonl stay valid. Randomness:
numpy default_rng([20261009, k]) per sampled request k. Writes
cases/t9/requests.jsonl and cases/t9/manifest.json.
"""
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SEED = 20261009
N = 1000
P_WORD = 0.15
ROWS = ["qwertzuiopü", "asdfghjklöä", "yxcvbnm"]
CONTRACTIONS = [
    (r"\bgibt es\b", "gibts"), (r"\bhabe ich\b", "hab ich"), (r"\bich habe\b", "ich hab"),
    (r"\bist es\b", "isses"), (r"\bnicht\b", "nich"), (r"\beinen\b", "nen"), (r"\beine\b", "ne"),
    (r"\bkannst du\b", "kannste"), (r"\bhast du\b", "haste"), (r"\bwie viel\b", "wieviel"),
    (r"\bheute\b", "heut"), (r"\bich möchte\b", "ich will"),
]
FILLERS = ["äh ", "also ", "hey ", "sag mal ", "du "]
TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}


def neighbours(c):
    out = []
    for r, row in enumerate(ROWS):
        i = row.find(c)
        if i < 0:
            continue
        out += [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS) and i < len(ROWS[rr]):
                out.append(ROWS[rr][i])
    return out


def edit(word, rng):
    kind = int(rng.integers(4))
    i = int(rng.integers(len(word)))
    if kind == 0:
        nb = neighbours(word[i])
        if nb:
            return word[:i] + nb[int(rng.integers(len(nb)))] + word[i + 1:]
        kind = 1
    if kind == 1:
        return word[:i] + word[i + 1:]
    if kind == 2:
        i = min(i, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def typo(text, rng):
    words = text.split(" ")
    long_ix = [i for i, w in enumerate(words) if sum(ch.isalpha() for ch in w) >= 4]
    if not long_ix:
        return text
    hit = [i for i in long_ix if rng.random() < P_WORD]
    if not hit:
        hit = [long_ix[int(rng.integers(len(long_ix)))]]
    for i in hit:
        words[i] = edit(words[i], rng)
    return " ".join(words)


def translit(text):
    return "".join(TRANSLIT.get(c, c) for c in text)


def colloquial(text, rng):
    for pat, rep in CONTRACTIONS:
        text = re.sub(pat, rep, text)
    return FILLERS[int(rng.integers(len(FILLERS)))] + text


def main():
    src = os.path.join(ROOT, "requests", "route1", "requests.jsonl")
    reqs = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
    de = [r for r in reqs if r["lang"] == "de" and r["line"].startswith("massive-")]
    assert len(de) == 2974, len(de)
    keep = sorted(np.random.default_rng(SEED).permutation(len(de))[:N])
    out, changed = [], {"typo": 0, "translit": 0, "colloquial": 0}
    for k, ix in enumerate(keep):
        r = de[ix]
        rng = np.random.default_rng([SEED, k])
        clean = r["text"]
        variants = {"clean": clean, "typo": typo(clean, rng), "translit": translit(clean), "colloquial": colloquial(clean, rng)}
        for v, t in variants.items():
            if v != "clean" and t != clean:
                changed[v] += 1
            req = json.loads(json.dumps(r))
            req.update({"line": f"{r['line']}~{v}", "base_line": r["line"], "variant": v, "text": t})
            req["request"]["state"]["utterance"] = t
            out.append(req)
    os.makedirs(os.path.join(ROOT, "cases", "t9"), exist_ok=True)
    p = os.path.join(ROOT, "cases", "t9", "requests.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"design": "plan/t9.md", "source": "requests/route1/requests.jsonl (MASSIVE 1.1 de, CC BY 4.0)",
           "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
           "sample": {"n": N, "seed": SEED, "rule": "MASSIVE de requests in file order; first 1,000 positions of default_rng(20261009).permutation(2974); kept in file order",
                      "lines": [de[i]["line"] for i in keep]},
           "variants": ["clean", "typo", "translit", "colloquial"], "changed": changed,
           "requests": len(out), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(ROOT, "cases", "t9", "manifest.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps({"requests": len(out), "changed": changed}))


if __name__ == "__main__":
    main()
