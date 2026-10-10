"""C3 regex baseline over the PDF text layer (plan/c3.md, Baseline). No model.
Three rules, everything else null:
  iban:       first IBAN-shaped token whose ISO 13616 mod-97 checksum is valid
  issue_date: first date after the label "Rechnungsdatum"
  amount_due: the amount after "Fälliger Betrag" (the KoSIT layout's label for
              BT-115; the plan named "Gesamtbetrag", which this layout does not
              print, see the dated section of plan/c3.md)
Usage: python3 -I bench/c3/baseline_regex.py cases/c3/text reference/c3/gold.jsonl <out-dir>"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import FIELDS, score_row  # noqa: E402

IBAN = re.compile(r"\b([A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?)\b")
DATE_AFTER = re.compile(r"Rechnungsdatum:?\s+(\d{1,2}\.\d{1,2}\.\d{4})")
DUE_AMT = re.compile(r"Fälliger Betrag\s+(-?[\d.]+,\d{2})")


def iban_ok(s):
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    r = s[4:] + s[:4]
    n = "".join(str(int(c, 36)) for c in r)
    return int(n) % 97 == 1


def extract(txt):
    out = {f: None for f in FIELDS}
    for m in IBAN.finditer(txt):
        if iban_ok(m.group(1)):
            out["iban"] = m.group(1).replace(" ", "")
            break
    m = DATE_AFTER.search(txt)
    if m:
        out["issue_date"] = m.group(1)
    m = DUE_AMT.search(txt)
    if m:
        out["amount_due"] = m.group(1)
    return out


def main(text_dir, gold_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cells = right = 0
    with open(os.path.join(out_dir, "result.jsonl"), "w", encoding="utf-8") as fo:
        for line in open(gold_path, encoding="utf-8"):
            g = json.loads(line)
            pred = extract(open(os.path.join(text_dir, g["id"] + ".txt"), encoding="utf-8").read())
            sc = score_row(g, pred)
            cells += len(sc)
            right += sum(sc.values())
            fo.write(json.dumps({"id": g["id"], "arm": "regex-baseline", "pred": pred, "correct": sc},
                                ensure_ascii=False) + "\n")
    print(f"regex-baseline: {right}/{cells} = {right / cells:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
