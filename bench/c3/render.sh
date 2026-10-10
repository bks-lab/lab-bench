#!/bin/sh
# C3 rendering, CPU only, run on a Mac (plan/c3.md).
# Usage: bench/c3/render.sh <testsuite-dir@1ce1daf> <visualization-dir@v2026-08-31> <tools-dir> <out-dir>
# tools-dir holds Saxon-HE-12.8.jar, xmlresolver-5.3.3.jar and fop-2.11/ (Apache binaries).
# Steps per invoice: CII -> cii-xr.xsl -> XR XML -> xr-pdf.xsl (foengine=fop) -> FO -> FOP 2.11 with
# the visualisation's conf/fop.xconf -> PDF -> pdftoppm 200 dpi greyscale PNG per page, pdftotext -layout.
set -eu
mkdir -p "$4"
TS=$(cd "$1" && pwd); VIS=$(cd "$2" && pwd); TOOLS=$(cd "$3" && pwd); OUT=$(cd "$4" && pwd)
CP="$TOOLS/Saxon-HE-12.8.jar:$TOOLS/xmlresolver-5.3.3.jar"
FOP="$TOOLS/fop-2.11/fop/fop"
# The xconf names fonts relative to its own folder; FOP resolves them against <base>.
CONF="$OUT/fop.xconf"
sed "s#<fop version=\"1.0\">#<fop version=\"1.0\"><base>$VIS/conf/</base>#" "$VIS/conf/fop.xconf" > "$CONF"
mkdir -p "$OUT/xml" "$OUT/xr" "$OUT/fo" "$OUT/pdf" "$OUT/png" "$OUT/text"
LIST=$(find "$TS" -name '*a-INVOICE_uncefact.xml' | grep -E '/0[1-4]\.[0-9]{2}a-INVOICE_uncefact\.xml$' | sort)
for f in $LIST; do
  b=$(basename "$f" .xml); id=${b%%-INVOICE_uncefact}
  case "$id" in 01.0[1-9]a|01.1[0-5]a|01.1[7-9]a|01.2[01]a|02.0[1-6]a|03.0[1-7]a|04.05a) ;; *) continue;; esac
  cp "$f" "$OUT/xml/$id.xml"
  java -cp "$CP" net.sf.saxon.Transform -s:"$f" -xsl:"$VIS/src/xsl/cii-xr.xsl" -o:"$OUT/xr/$id.xml"
  java -cp "$CP" net.sf.saxon.Transform -s:"$OUT/xr/$id.xml" -xsl:"$VIS/src/xsl/xr-pdf.xsl" -o:"$OUT/fo/$id.fo" foengine=fop
  "$FOP" -q -c "$CONF" -fo "$OUT/fo/$id.fo" -pdf "$OUT/pdf/$id.pdf"
  pdftoppm -r 200 -gray -png "$OUT/pdf/$id.pdf" "$OUT/png/$id"
  pdftotext -layout -enc UTF-8 "$OUT/pdf/$id.pdf" "$OUT/text/$id.txt"
  echo "$id ok"
done
