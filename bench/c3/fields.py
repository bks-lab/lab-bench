"""C3 field definitions, normalisation and scoring (plan/c3.md). Stdlib only."""
import re
import unicodedata
from decimal import Decimal, InvalidOperation

FIELDS = ["invoice_number", "issue_date", "due_date", "seller_name", "seller_vat_id",
          "buyer_name", "iban", "net_total", "vat_total", "gross_total", "amount_due", "currency"]
MONEY = ["net_total", "vat_total", "gross_total", "amount_due"]
DATES = ["issue_date", "due_date"]

NS = {
    "rsm": "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100",
    "ram": "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100",
    "udt": "urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100",
}


def _ws(s):
    return re.sub(r"\s+", " ", s).strip()


def norm_money(v):
    """Decimal with 2 places. Accepts 1234.5, 1.234,56, 1234,56, -12,00, '1 234,56 EUR'."""
    if v is None:
        return None
    s = str(v).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in "-.,":
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "", s.count(".") - 1)
    try:
        return str(Decimal(s).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def norm_date(v):
    """ISO date. Accepts 2024-01-31, 20240131, 31.01.2024."""
    if v is None:
        return None
    s = str(v).strip()
    m = re.fullmatch(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"
    return s


def norm(field, v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        v = str(v)
    if not isinstance(v, str):
        return None
    if v.strip() == "" or v.strip().lower() == "null":
        return None
    if field in MONEY:
        return norm_money(v)
    if field in DATES:
        return norm_date(v)
    if field in ("seller_name", "buyer_name"):
        return unicodedata.normalize("NFC", _ws(v)).casefold()
    if field in ("seller_vat_id", "iban", "currency"):
        return re.sub(r"\s+", "", v).upper()
    return v.strip()


def score_row(gold, pred):
    """Per-field exact match after normalisation. pred may be None (call failed)."""
    out = {}
    for f in FIELDS:
        p = norm(f, (pred or {}).get(f)) if isinstance(pred, dict) else None
        out[f] = (p == gold[f]) if pred is not None else False
    return out
