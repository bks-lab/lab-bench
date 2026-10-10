"""C3 sanity gate (plan/c3.md): every non-null gold value must be found, after
normalisation, in the pdftotext -layout output of its PDF. Gate is 100 %.
Usage: python3 -I bench/c3/render_check.py reference/c3/gold.jsonl cases/c3/text > results/c3/render-check.md"""
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import FIELDS, MONEY, DATES, norm_money, norm_date  # noqa: E402

NUM = re.compile(r"-?\d{1,3}(?:\.\d{3})*,\d+|-?\d+,\d+|-?\d+(?:\.\d+)?")
DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")


def found(field, gold, txt):
    flat = re.sub(r"\s+", " ", unicodedata.normalize("NFC", txt))
    if field in MONEY:
        return gold in {norm_money(m) for m in NUM.findall(txt)}
    if field in DATES:
        return gold in {norm_date(m) for m in DATE.findall(txt)}
    if field in ("seller_name", "buyer_name"):
        return gold in flat.casefold()
    if field in ("seller_vat_id", "iban", "currency"):
        return gold in re.sub(r"\s+", "", txt).upper()
    return gold in flat or gold in txt


def main(gold_path, text_dir):
    rows = [json.loads(l) for l in open(gold_path, encoding="utf-8")]
    total = hit = 0
    misses = []
    per = {f: [0, 0] for f in FIELDS}
    for r in rows:
        txt = open(os.path.join(text_dir, r["id"] + ".txt"), encoding="utf-8").read()
        for f in FIELDS:
            g = r[f]
            if g is None:
                continue
            total += 1
            per[f][1] += 1
            if found(f, g, txt):
                hit += 1
                per[f][0] += 1
            else:
                misses.append((r["id"], f, g))
    print("# C3 render check\n")
    print("Every non-null gold value of `reference/c3/gold.jsonl` searched in the")
    print("`pdftotext -layout` output of its rendered PDF (`cases/c3/text/`).\n")
    print(f"Result: {hit} of {total} non-null gold values found ({100.0 * hit / total:.1f} %). "
          f"Gate: 100 %, {'PASSED' if hit == total else 'FAILED'}.\n")
    print("| field | found | non-null |\n|---|---|---|")
    for f in FIELDS:
        print(f"| {f} | {per[f][0]} | {per[f][1]} |")
    if misses:
        print("\n## Misses\n\n| invoice | field | gold |\n|---|---|---|")
        for m in misses:
            print(f"| {m[0]} | {m[1]} | `{m[2]}` |")
    return 0 if hit == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
