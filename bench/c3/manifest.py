"""Write cases/c3/manifest.json: SHA-256 of XML, PDF, every PNG and text per invoice.
Usage: python3 -I bench/c3/manifest.py cases/c3"""
import hashlib
import json
import os
import sys


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(d):
    ids = sorted(n[:-4] for n in os.listdir(os.path.join(d, "xml")) if n.endswith(".xml"))
    pngs = sorted(os.listdir(os.path.join(d, "png")))
    inv = []
    for i in ids:
        pages = [p for p in pngs if p.startswith(i + "-") and p.endswith(".png")]
        pages.sort(key=lambda p: int(p[len(i) + 1:-4]))
        inv.append({
            "id": i,
            "source_file": f"{i}-INVOICE_uncefact.xml",
            "xml_sha256": sha(os.path.join(d, "xml", i + ".xml")),
            "pdf_sha256": sha(os.path.join(d, "pdf", i + ".pdf")),
            "text_sha256": sha(os.path.join(d, "text", i + ".txt")),
            "pages": len(pages),
            "png": [{"file": p, "sha256": sha(os.path.join(d, "png", p))} for p in pages],
        })
    out = {
        "test": "C3",
        "written": "2026-10-09",
        "testsuite": {"repo": "https://github.com/itplr-kosit/xrechnung-testsuite",
                      "commit": "1ce1dafffd035e5ce59a7ab0d3bf44ac14af16c9", "licence": "Apache-2.0"},
        "renderer": {"repo": "https://github.com/itplr-kosit/xrechnung-visualization", "tag": "v2026-08-31",
                     "commit": "e158048ec6774cd4837744b21ca34dc95ba14862", "licence": "Apache-2.0",
                     "steps": "cii-xr.xsl, xr-pdf.xsl (foengine=fop, default layout, lang de), conf/fop.xconf"},
        "tools": {"saxon": "Saxon-HE 12.8 (MPL-2.0, Maven Central net.sf.saxon:Saxon-HE:12.8)",
                  "xmlresolver": "5.3.3", "fop": "Apache FOP 2.11 (fop-2.11-bin.tar.gz)",
                  "java": "OpenJDK 27 (Homebrew)", "poppler": "pdftoppm/pdftotext 26.10.0 (Homebrew)",
                  "raster": "pdftoppm -r 200 -gray -png", "text": "pdftotext -layout -enc UTF-8"},
        "invoices": len(inv),
        "pages": sum(x["pages"] for x in inv),
        "items": inv,
    }
    json.dump(out, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(out["invoices"], out["pages"])


if __name__ == "__main__":
    main(sys.argv[1])
