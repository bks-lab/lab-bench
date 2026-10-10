---
license: apache-2.0
language:
  - de
  - en
pretty_name: XRechnung invoice fields from page images
task_categories:
  - image-to-text
  - document-question-answering
tags:
  - invoices
  - e-invoicing
  - xrechnung
  - document-extraction
  - benchmark
  - local-models
size_categories:
  - n<1K
configs:
  - config_name: pages
    data_files:
      - split: test
        path: pages/*
    default: true
  - config_name: invoices
    data_files:
      - split: test
        path: invoices.jsonl
  - config_name: predictions
    data_files:
      - split: test
        path: predictions.jsonl
---

# XRechnung invoice fields from page images

{{N_INVOICES}} German test invoices of the official XRechnung test suite,
rendered to {{N_PAGES}} page images, with twelve fields per invoice that a
model has to read from the images alone: invoice number, issue and due date,
seller and buyer name, seller VAT id, IBAN, net, VAT and gross total, amount
due and currency. The correct value of every field comes from the invoice's
XML, so the gold answers are exact and nobody had to annotate them.

This is test C3 of [lab-bench]({{REPO}}), the bench BKS-Lab uses to decide
which open model can do a task on hardware we control. The recorded outputs
of every model we ran are included, so you can compare your own model with
ours invoice by invoice.

- Built from commit `{{COMMIT}}` of bks-lab/lab-bench.
- Method: [results/c3/method.md]({{REPO}}/blob/{{COMMIT}}/results/c3/method.md),
  findings: [results/c3/findings.md]({{REPO}}/blob/{{COMMIT}}/results/c3/findings.md).
- Browse the outputs: [bks-lab/xrechnung-invoice-fields-viewer](https://huggingface.co/spaces/bks-lab/xrechnung-invoice-fields-viewer).

## Configs

**pages** (default). One row per page image: 200 dpi, greyscale, no text
layer. Each row carries the invoice id, the page number and the gold fields
of its invoice.

**invoices.** One row per invoice: gold fields, number of pages, and the
paths of the CII XML and the rendered PDF in this repository.

**predictions.** One row per invoice and model arm, from our runs on one
RTX 4090 with 24 GB through Ollama: the predicted fields, a correct flag per
field (exact match after normalisation, see the method), the model's Hub id
where there is one, quantisation, licence and seconds per invoice. `kind`
says how the arm reads the invoice: `vision` from the page images, `ocr+text`
through docling OCR and then a text model. The arm with `reference_arm` set
reads the PDF text layer instead of an image. It shows what a text model can
do when the text is already clean and is not a scan result.

## Results so far

A regular expression over the PDF text layer gets {{REGEX}} of the 408
fields right. The best vision model, {{BEST_ARM}}, gets {{BEST}}. The
scores per model with 95 % bootstrap intervals are in the findings linked
above. One run per arm, temperature 0.

## Limits

- The invoices are test cases: names and numbers are the test suite's
  placeholders, layouts are the one official visualisation. Real scans with
  stamps, skew, handwriting or other layouts are harder.
- 34 invoices give wide intervals. Two models whose intervals overlap are not
  shown to differ.
- 17 of the 408 gold values are null because the invoice has no such field
  (no due date, no IBAN, no seller VAT id). A model gets those right by
  answering null. The render check in the repository found all 391 other
  gold values in the text of the rendered PDF.

## Licence

The test invoices are from the
[KoSIT XRechnung test suite](https://github.com/itplr-kosit/xrechnung-testsuite)
(commit 1ce1daf), rendered with the
[KoSIT XRechnung visualisation](https://github.com/itplr-kosit/xrechnung-visualization)
(v2026-08-31). Both are Apache-2.0, and so are the gold rows and outputs
derived from them. The PDFs embed subsets of Source Serif Pro under the SIL
Open Font License 1.1. The licence files are included.
