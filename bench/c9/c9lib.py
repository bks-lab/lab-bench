"""Shared pieces of the C9 runner and scorer (plan/c9.md, notes of 2026-10-12). Stdlib only.

- load(repo, data): the frozen documents (hash-checked against cases/c9/manifest.json) and the label file
- gold(doc, labels): the gold spans of one document mapped to the seven types
- norm(text, start, end): span boundaries after trimming whitespace,
  trailing punctuation and enclosing brackets or quotes, the boundary rule of
  the primary metric (brackets and quotes added after the sanity arm, see
  plan/c9.md, notes of 2026-10-12)
- chunks(text): word windows of 384 words with 50 words overlap, as
  (char_start, char_end) pairs, for the GLiNER arms
- regex_spans(text): the regex baseline (e-mail, IBAN with checksum, German
  phone numbers)
- find_all(text, needle): every non-overlapping occurrence, for LLM answers
"""
import hashlib
import json
import os
import re

LEADING = "[(\"'\u201e\u201c\u201a\u2018<"
TRAILING = ".,;:!?])\"'\u201c\u201d\u2018\u2019>"
CHUNK_WORDS = 384
CHUNK_OVERLAP = 50


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def docs_path(repo, data=None):
    """docs.jsonl from --data (the workstation), else cases/c9/ once committed, else .cache/c9/ (the Mac)."""
    for p in ([os.path.join(data, "docs.jsonl")] if data else []) + [
            os.path.join(repo, "cases", "c9", "docs.jsonl"), os.path.join(repo, ".cache", "c9", "docs.jsonl")]:
        if os.path.exists(p):
            return p
    raise SystemExit("docs.jsonl not found: run bench/c9/freeze.py or pass --data")


def load(repo, data=None):
    man = json.load(open(os.path.join(repo, "cases", "c9", "manifest.json"), encoding="utf-8"))
    p = docs_path(repo, data)
    if sha(p) != man["docs_sha256"]:
        raise SystemExit(f"{p} does not match cases/c9/manifest.json")
    docs = [json.loads(line) for line in open(p, encoding="utf-8")]
    labels = json.load(open(os.path.join(repo, "reference", "c9", "labels.json"), encoding="utf-8"))
    return man, docs, labels


def norm(text, start, end):
    """(start, end) after trimming whitespace, opening brackets and quotes at the start and
    punctuation, closing brackets and quotes at the end, repeated until stable; None if empty."""
    start, end = max(0, int(start)), min(len(text), int(end))
    while True:
        before = (start, end)
        while start < end and (text[start].isspace() or text[start] in LEADING):
            start += 1
        while end > start and (text[end - 1].isspace() or text[end - 1] in TRAILING):
            end -= 1
        if (start, end) == before:
            return (start, end) if end > start else None


def gold(doc, labels):
    """Gold spans of the seven types: sorted unique (start, end, type) after norm()."""
    out = set()
    for s in doc["spans"]:
        t = labels["map"].get(s["label"])
        if t is None:
            continue
        b = norm(doc["text"], s["start"], s["end"])
        if b:
            out.add((b[0], b[1], t))
    return sorted(out)


def chunks(text, words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    toks = [m.span() for m in re.finditer(r"\S+", text)]
    if not toks:
        return [(0, len(text))]
    out, i = [], 0
    while True:
        j = min(i + words, len(toks))
        out.append((toks[i][0], toks[j - 1][1]))
        if j == len(toks):
            return out
        i = j - overlap


EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
PHONE_DE = re.compile(r"(?<![\w+])(?:\+49|0049|\(0\d{2,5}\)|0)[ /-]?(?:\(0\)[ /-]?)?\d{2,5}(?:[ /-]?\d{2,}){1,4}(?![\w])")


def iban_ok(s):
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    r = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in r)) % 97 == 1
    except ValueError:
        return False


def regex_spans(text):
    out = []
    for m in EMAIL.finditer(text):
        out.append((m.start(), m.end(), "email"))
    for m in IBAN.finditer(text):
        if iban_ok(m.group()):
            out.append((m.start(), m.end(), "iban_account"))
    taken = [(a, b) for a, b, _ in out]
    for m in PHONE_DE.finditer(text):
        if sum(c.isdigit() for c in m.group()) >= 6 and not any(a < m.end() and m.start() < b for a, b in taken):
            out.append((m.start(), m.end(), "phone"))
    return out


def find_all(text, needle):
    if not needle:
        return []
    out, i = [], text.find(needle)
    while i >= 0:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + len(needle))
    return out
