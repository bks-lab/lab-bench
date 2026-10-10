"""Gold values for C3 from the CII XML by XPath (plan/c3.md, field table).
Usage: python3 -I bench/c3/gold.py cases/c3/xml > reference/c3/gold.jsonl"""
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import FIELDS, NS, norm  # noqa: E402

H = "rsm:SupplyChainTradeTransaction/"
AG = H + "ram:ApplicableHeaderTradeAgreement/"
ST = H + "ram:ApplicableHeaderTradeSettlement/"
SUM = ST + "ram:SpecifiedTradeSettlementHeaderMonetarySummation/"


def text(root, path):
    el = root.find(path, NS)
    return el.text if el is not None and el.text is not None and el.text.strip() else None


def raw(root):
    cur = text(root, ST + "ram:InvoiceCurrencyCode")
    vat = None
    for el in root.findall(SUM + "ram:TaxTotalAmount", NS):  # BT-110 is the one in invoice currency
        if el.get("currencyID") in (None, cur):
            vat = el.text
            break
    vat_ids = [e.text for e in root.findall(AG + "ram:SellerTradeParty/ram:SpecifiedTaxRegistration/ram:ID", NS)
               if e.get("schemeID") == "VA"]
    return {
        "invoice_number": text(root, "rsm:ExchangedDocument/ram:ID"),
        "issue_date": text(root, "rsm:ExchangedDocument/ram:IssueDateTime/udt:DateTimeString"),
        "due_date": text(root, ST + "ram:SpecifiedTradePaymentTerms/ram:DueDateDateTime/udt:DateTimeString"),
        "seller_name": text(root, AG + "ram:SellerTradeParty/ram:Name"),
        "seller_vat_id": vat_ids[0] if vat_ids else None,
        "buyer_name": text(root, AG + "ram:BuyerTradeParty/ram:Name"),
        "iban": text(root, ST + "ram:SpecifiedTradeSettlementPaymentMeans/ram:PayeePartyCreditorFinancialAccount/ram:IBANID"),
        "net_total": text(root, SUM + "ram:TaxBasisTotalAmount"),
        "vat_total": vat,
        "gross_total": text(root, SUM + "ram:GrandTotalAmount"),
        "amount_due": text(root, SUM + "ram:DuePayableAmount"),
        "currency": cur,
    }


def main(xml_dir):
    for name in sorted(os.listdir(xml_dir)):
        if not name.endswith(".xml"):
            continue
        root = ET.parse(os.path.join(xml_dir, name)).getroot()
        r = raw(root)
        row = {"id": name[:-4]}
        row.update({f: norm(f, r[f]) for f in FIELDS})
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
