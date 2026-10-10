"""SQuAD v1.1 answer scoring (the official evaluate-v1.1.py definitions), plus
the German-article variant that plan/c6.md names as a sensitivity check."""
import re
import string
from collections import Counter

ARTICLES = {"en": r"\b(a|an|the)\b", "de": r"\b(der|die|das|ein|eine)\b"}


def normalize(s, lang="en"):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(ARTICLES[lang], " ", s)
    return " ".join(s.split())


def f1(pred, gold, lang="en"):
    p, g = normalize(pred, lang).split(), normalize(gold, lang).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred, gold, lang="en"):
    return float(normalize(pred, lang) == normalize(gold, lang))


def best(fn, pred, golds, lang="en"):
    """max over the gold answers, as the official script does"""
    return max(fn(pred, g, lang) for g in golds)
