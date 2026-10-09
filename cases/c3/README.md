# C3 inputs (frozen 2026-10-09)

34 KoSIT XRechnung test invoices (CII), rendered with the KoSIT XRechnung
visualisation. Built by `bench/c3/render.sh`; hashes in `manifest.json`
(`bench/c3/manifest.py`); design in `plan/c3.md`.

- `xml/`: source CII files, xrechnung-testsuite at `1ce1daf`
- `pdf/`: rendered PDFs (visualisation `v2026-08-31`, Saxon-HE 12.8, FOP 2.11)
- `png/`: every page at 200 dpi, greyscale, no text layer (`<id>-<page>.png`)
- `text/`: `pdftotext -layout` of each PDF (render check, regex baseline,
  reference arm)

Licences: the test suite and the visualisation are Apache-2.0
(`LICENSE-xrechnung-testsuite`, `LICENSE-xrechnung-visualization`). The
PDFs embed subsets of Source Serif Pro, SIL Open Font License 1.1
(`LICENSE-SourceSerifPro-OFL.txt`). All files here are derived from that
data; names and numbers are the test suite's placeholders.
