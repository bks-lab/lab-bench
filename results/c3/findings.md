# C3 findings: invoice fields from page images on one RTX 4090

Design `plan/c3.md` (committed 2026-10-09 before any arm), method
`results/c3/method.md`, shared rules `plan/c0-common.md`. Also roadmap row
T8. Runs: night of 2026-10-09 to 10, PC job queue, Ollama 0.35.0, every arm
with `num_ctx` 32768, temperature 0, seed 1. Data: 34 KoSIT test invoices
rendered with the KoSIT visualisation, 118 page images at 200 dpi, 408
cells. Scores: `results/c3/score.md`, chart data: `results/c3/summary.json`,
capability row: `results/capabilities.json`. Every cell is rescored by the
scorer and, with a separate normaliser, by `bench/check-cap.py` (7 checks
agree). Notes after the run, including the unfinished arm and the
qwen3-vl-8b rerun: the dated sections of 2026-10-10 at the end of
`plan/c3.md`.

## Sanity gate

Internal check before any arm (`results/c3/render-check.md`): 391 of 391
non-null gold values are in the PDF text layer. Passed.

## Result

| arm | kind | params | field accuracy | 95 % CI | money fields | IBAN | all 12 right | s per invoice | peak VRAM GB | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-26b | vision | 25.2 B (MoE) | **0.980** | 0.968 to 0.990 | 0.993 | 0.912 | 26 of 34 | 3.6 | 19.8 | yes |
| gemma4-12b | vision | 11.9 B | **0.978** | 0.963 to 0.990 | 0.993 | 0.912 | 26 of 34 | 2.1 | 9.8 | yes |
| mistral-small-3.2 | vision | 24.0 B | 0.909 | 0.880 to 0.936 | 0.941 | 0.912 | 10 of 34 | 4.6 | 19.7 | yes |
| docling+gemma4-12b | OCR + text | 11.9 B | 0.875 | 0.838 to 0.907 | 0.904 | 0.706 | 5 of 34 | 6.5 | 18.7 | yes |
| docling+qwen3.8-27b | OCR + text | 27.3 B | 0.591 | 0.561 to 0.618 | 0.000 | 0.765 | 0 | 26.1 | 22.8 | offloaded |
| qwen3-vl-8b (rerun, 180 min budget) | vision | 8.8 B | 0.571 | 0.466 to 0.672 | 0.279 | 0.765 | 3 of 34 | 39.0 | 11.9 | yes |
| pdftext+qwen3.8-27b (reference, text layer, not a scan) | text | 27.3 B | 0.657 | 0.647 to 0.664 | 0.000 | 0.912 | 0 | 2.1 | 19.7 | yes |
| qwen3-vl-32b | vision | 32 B | did not finish in 180 min | | | | | | 22.8 | offloaded |
| regex baseline over the text layer | rules | none | 0.270 | | | | | | | |

- **Gemma 4 reads invoices from images almost perfectly.** gemma4-12b gets
  399 of 408 cells, gemma4-26b 400; the paired bootstrap does not separate
  them (difference 0.003, p 0.86). Both get every gross total and amount
  due right. The 12B model needs half the memory (9.8 GB) and 2.1 s per
  invoice of 2 to 9 pages, at 614 GPU joules per invoice.
- **Every IBAN miss of the vision arms is the same trap.** The three
  invoices that print a direct-debit account although BT-84 is absent
  (03.01a, 03.04a, 03.05a, named in the plan before the run) are the only
  IBAN errors of gemma4-12b, gemma4-26b and mistral-small-3.2: each copied
  the printed IBAN where the gold is null. Without those three, IBAN is 31
  of 31. The remaining gemma errors are mostly the seller VAT id (5 of 9
  for gemma4-12b).
- **The image beats OCR plus text.** docling with EasyOCR in front of the
  same gemma4-12b loses 0.10 (0.875 against 0.978; against the best arm
  the paired test gives p < 0.001): OCR misreads
  invoice numbers, IBANs and amounts that the vision model reads right, and
  it costs 4.9 s of OCR per invoice on top.
- **qwen3.8-27b returns no amounts.** In both of its arms, even on the
  clean PDF text layer, it answered `null` for all four money fields of all
  34 invoices; its other cells are 268 of 272 right on the text layer. The
  instruction asks for amounts as numbers while the schema types every
  field as string or null; the other models wrote the amount as a string,
  qwen3.8 chose null. Scored as returned, as the method says; no prompt
  change was tried. In the docling arm it also ran with part of the model
  off the card (EasyOCR shares the GPU in that job), 19.7 s per call
  instead of 2.1 s.
- **mistral-small-3.2** is right on 0.909, weaker on dates and invoice
  numbers (due date 25 of 34).

## The two Qwen3-VL arms

Both stopped at their first queue timeout without writing results; the 8B
arm finished on one rerun. From the job logs and the 1 s GPU logs:

- **qwen3-vl-32b: CPU spill.** The model (21 GB weights) and the 32,768
  token context do not fit: memory sat at 24.0 of 24.0 GB for the whole
  run while the GPU drew 59 W on average (the fitting arms draw about 360 W
  during their calls). The 17 invoices it finished took 340 to 717 s each
  (median 494 s), so 34 would need about 4.7 h, against the 180 min budget.
  On those 17 its log shows 184 of 204 cells right; that is descriptive
  and not scored. The cheap fixes are not plan-conformant: `num_ctx` 8192
  (the plan's first setting) would cut multi-page prompts, which is why the
  plan moved every arm to 32,768 before the run, and a lower image
  resolution changes the method. Result: **did not finish within the
  preregistered budget on one 24 GB RTX 4090.**
- **qwen3-vl-8b: slow image prompts plus empty answers.** It fits (12.9 GB,
  no spill; 357 W average). The first run stopped at its 60 min queue
  timeout after 17 invoices. The rerun with the same code and settings and
  a 180 min queue timeout (`20261010-071202-c3-qwen3-vl-8b`; the timeout is
  not part of the design) finished in 101 GPU minutes and is the run
  scored: **0.571 [0.466, 0.672]**, far below the Gemma arms (paired test
  p < 0.001). Six of 34 invoices failed: five returned empty content after
  minutes of full GPU load (01.09a, 01.12a, 01.18a, 02.01a, 03.05a; the
  first three failed the same way in the first run), one got HTTP 400 from
  Ollama (02.05a, the nine-page invoice). That fits a generation that runs
  until the context is full without producing JSON; the runner keeps no
  raw answer of a failed call, so the cause is not proven. On the 28
  answered invoices it gets 233 of 336 cells (0.69), and on six of them it
  returned `null` for all four money fields. Good calls take 39 s per
  invoice (median, 11.8 s per page; gemma4-12b: 2.1 s), consistent with
  Qwen3-VL encoding each 1654 x 2339 page near native resolution, at
  62,110 GPU joules per invoice.

## Capability map

The roadmap rule for ties picks the arm with less peak VRAM:
**gemma4-12b** (Apache-2.0, open source), 0.978 [0.963, 0.990] against a
regex baseline of 0.270, 2.1 s per invoice (0.69 s per page), 9.8 GB peak,
614 GPU joules per invoice, fits on one card. The tie with gemma4-26b
(0.980) is stated in the row; gemma4-12b is also the smallest model within
the interval of the best.

## Limits

- Every invoice has the same clean KoSIT layout without logos, stamps or
  scan noise. The test measures reading and field mapping, not robustness
  to layouts or bad scans; real scans will score lower.
- The source XML is public since 2017; a model that saw it could know the
  values. The rendered pages are new.
- One run per arm; latencies are single calls at concurrency 1.
