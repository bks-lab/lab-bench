# einv1: yes/no questions over the free text of an e-invoice

Test design, written 2026-10-09 and committed before any einv1 run and
before any reference judgment. State in `results/toolmap.json`: row
`einv1-yesno-de`, all three cells `planned`. A change to this design after
the commit goes into a new dated section at the end, never into the text
above it.

## Question

Given only the free text of an e-invoice, does a tool answer ten fixed yes/no
questions about it as well as a careful reader? One cell is one invoice and
one question, 34 x 10 = 340 cells. The three tool families of the map
(Jev, GLiNER, local LLM) are scored against the same blind reference.

## Data

- Source: the KoSIT XRechnung test suite,
  https://github.com/itplr-kosit/xrechnung-testsuite at commit `1ce1daf`,
  licence Apache-2.0 (its LICENSE file is copied into `cases/einv1/`).
- Invoices: the 34 CII files `*_uncefact.xml` of that commit, the same 34
  that the BKS-Lab page "Jev reads e-invoices" uses
  (bks-web `src/data/labor/feld-text.kosit.json`, measured 2026-09-22):
  01.01a to 01.15a, 01.17a to 01.21a, 02.01a to 02.06a, 03.01a to 03.07a,
  04.05a.
- Input: free text only, no structured field. Per invoice:
  - `hinweise`: document notes (BT-22)
  - `zahlungsbedingungen`: payment terms text (BT-20)
  - `positionen`: per line, first 12 lines: item name (BT-153), item
    description (BT-154), line note (BT-127)
  - `steuerhinweis`: the sorted unique VAT exemption reasons (BT-120)

  Whitespace collapsed, empty keys dropped. The texts are German (hence
  `lang: de`), the questions English.
- Freezing: `cases/einv1/` gets one row per invoice with the file name, the
  sha256 of the source XML and the extracted state. Before the freeze is
  committed, the extracted state is checked to be identical to the `text`
  field of `feld-text.kosit.json` for all 34 invoices.
- Not part of einv1: the FeRD/ZUGFeRD samples (licence not cleared for
  publication) and the structured fields. The structured field answers a
  different question (is the detail machine-readable), so it is not the
  reference here.

## The ten questions, verbatim

| id | question | related EN 16931 field |
|---|---|---|
| faelligkeit | Does the text state a concrete payment due date, a calendar date by which payment must be made? | BT-9 |
| skonto | Does the text offer a cash discount (Skonto) for early payment? | BT-20, #SKONTO# |
| lastschrift | Does the text say the amount will be collected by SEPA direct debit? | BT-81 = 59 |
| anderer_empfaenger | Does the text say payment should go to a party other than the seller, such as a factoring company, parent company or collection agent? | BG-10 |
| vorauszahlung | Does the text say that part or all of the amount has already been paid, for example an advance payment or deposit that is deducted? | BT-113 |
| steuerfrei | Does the text say that VAT is not charged on part or all of the invoice, for example because of an exemption, reverse charge or an intra-community supply? | BT-118 / BT-151 |
| bestellbezug | Does the text refer to a purchase order of the buyer, such as an order number or order date? | BT-13 |
| vertragsbezug | Does the text refer to a contract, such as a contract number or insurance policy number? | BT-12 |
| leistungszeitraum | Does the text state a billing or service period, a date range during which the goods or services were delivered? | BG-14 / BG-26 |
| lieferdatum | Does the text state the date on which goods were delivered or the service was performed? | BT-72 |

Source of the wording: `fragen` in bks-web
`src/data/labor/feld-text.kosit.json`. No question is reworded for any arm.

## Arms

One call per cell for every arm (one invoice, one question), as in route1
since 2026-10-09. Run 1 per arm. Versions, digests and revisions are
recorded in every result row.

| family | arm | how |
|---|---|---|
| jev | `jev` | TypeSafe API, `jev-latest` (resolved version recorded), question type `noul`, the state as the JSON object above. p(yes) = `noul` |
| gliner | `gliner2.5-multi-decide`, `gliner2.5-multi-decide-intext` | multilingual Decide classifier, labels `yes` / `no`, question as task prompt (`prompt`) and appended to the text (`intext`). p(yes) = softmax probability of `yes`. Forced change: GLiNER2 rejects "(" and ")" in a task, so "(Skonto)" becomes "[Skonto]", recorded in `adapt` |
| gliner, extra | `gliner2.5-decide`, `gliner2.5-decide-1b` | English-only Decide models on German text: reported as extra, not eligible as the family's best arm (same rule as route1 and pv1) |
| local-llm | `winnow-12b`, `qwen3.8-27b` | Ollama, harness of `bench/run-local.mjs`: options A = Yes, B = No, assistant turn pre-filled, p(yes) from the letter probabilities renormalised over A and B, `mass` recorded. The system prompt is rewritten for invoices and recorded verbatim in the method file |

All local arms run on the same workstation as route1 and ex1 (RTX 4090).

The tool map takes the best eligible arm per family. That choice is made
after the fact, so the arm counts per family (Jev 1, GLiNER 2, local 2) are
stated next to the result.

## Reference

Built blind, for all 340 cells, and finished before any arm is scored.

1. **Judging rules**, written into `reference/einv1/rules.md` before the
   first judgment: a cell is `yes` only if the free text itself states the
   detail; general terms and conditions, a mere date of the invoice, or a
   detail present only in a structured field count as `no`. A month in which
   goods were delivered counts for `lieferdatum`, not for
   `leistungszeitraum`, which needs a billed period with a start and an end
   (rule carried over from the 2026-09-22 review).
2. **Two model judges**, independent of each other: `claude-opus-5-5` and
   `claude-sonnet-5-5`. Each sees the extracted free text, the question and
   the rules, and answers `yes` or `no` with the quoted words it relies on.
   Neither sees any arm's answer or probability, the structured field, or
   the earlier review.
3. **Adjudication**: every cell where the two disagree goes to a third,
   fresh judge from another family (`claude-fable-5-1`), with the same blind
   inputs plus the two verdicts and quotes. Its verdict is final for that
   cell, marked `"adjudicated": true`.
4. **Human sample**: Michael Boiman judges, blind to all verdicts, a seeded
   random sample of 50 cells (seed 1, five per question) plus every
   adjudicated cell. Where the human differs, the human verdict is the
   reference, marked `"human": true`.
5. **Gate**: if the human and the model reference agree on fewer than
   45 of the 50 sampled cells (90 %), the reference is not used. The rules
   are revised in a new dated section of this file and steps 2 to 5 are
   repeated before any scoring.

The 78 cells reviewed on 2026-09-22 (23 of them KoSIT) are not used as the
reference. They were selected by Jev's own probability (every cell Jev put
above 0.7 against an empty field or between 0.3 and 0.7), so they are not a
neutral sample and would favour or punish Jev by construction. They are
compared with the new reference afterwards, as a consistency check only.

A known limit: Jev already answered these 340 cells on 2026-09-22, and the
people who wrote this design have seen those answers. The questions and the
data are unchanged from that run; the reference judges never see them.

## Metric

- Primary: accuracy per arm, pooled over the 340 cells, with yes when
  p(yes) >= 0.5.
- Secondary: accuracy per question (n = 34 each, descriptive only), macro
  F1 over yes and no, AUC of p(yes) against the reference (threshold-free,
  because GLiNER probabilities are not calibrated), and the three-band view
  of the lab page (yes at p >= 0.7, no at p <= 0.3, unsure between) with
  the share of unsure cells.
- Speed: median ms per call. Cost: Jev median input tokens per call x the
  published input price.

## Baseline

Most frequent reference answer per question (expected `no` for most
questions), applied to all 34 invoices of that question and pooled.
skill = (accuracy - baseline) / (100 - baseline), as everywhere in the map.

## Thresholds fixed in advance

- Decision threshold 0.5 for the primary metric, for every arm. No arm gets
  a tuned threshold.
- A difference to Jev counts only with an exact two-sided McNemar test on
  the 340 paired cells at p < 0.05. Without it the result is "no difference
  shown", never "level".
- No claim per question: n = 34 is too small.
- A family counts as usable for this task at a pooled skill of 0.5 or more.
- The reference gate of 45 of 50 above.

## Outputs

- `cases/einv1/` frozen inputs with hashes and the KoSIT licence
- `reference/einv1/` rules, judge rows, adjudication, human sample
- `results/einv1/<arm>/` run rows, `results/einv1/score.md`,
  `results/einv1/findings.md`
- the row `einv1-yesno-de` in `results/toolmap.json` moves from `planned`
  to `measured`

## Revision 2026-10-09: after an independent review, before any run

Written the same day, before any einv1 run, before any reference judgment
and before the freeze. Where this section and the text above differ, this
section holds. The text above stays as committed.

### The human sample gates, it never overrides

Michael Boiman saw Jev's answers on these invoices in the 2026-09-22 review.
His verdicts can therefore not become reference values without carrying
Jev's influence into the reference. Steps 4 and 5 above are replaced:

4. **Human sample, check only.** Michael Boiman judges a seeded sample
   (below) blind to every model verdict and every arm's answer. His
   verdicts are stored in `reference/einv1/human-sample.jsonl` and never
   change a reference cell. There is no `"human": true` cell.
5. **Gate.** The model reference is used only if, on the sampled cells,
   - Cohen's kappa between the human and the reference is 0.70 or more, and
   - the positive specific agreement on `yes`, 2a / (2a + b + c) with a =
     both yes, b and c = only one says yes, is 0.80 or more.

   Raw agreement (the 45 of 50 above) is reported, but it is no longer the
   gate: with mostly `no` cells, "always no" can reach it. If the gate
   fails, the rules are revised in a new dated section, steps 2 to 5 are
   repeated with a fresh seed (2, then 3), and no arm is scored before a
   pass.

Adjudicated cells are also shown to the human, in the same blind form, and
their agreement is reported separately. They do not enter the gate.

### The sample, with its generator

Generator: `rng(1)` from `bench/lib/route-ex.mjs` (mulberry32, seed 1, the
same generator `bench/normalize.mjs` uses for shuffled runs). One generator
instance is used for all three steps, in this order, with nothing else
drawing from it:

1. Per question in the table order above (`faelligkeit` first), the 34 cells
   sorted by invoice file name, then Fisher-Yates from the end: for
   i = 33 down to 1, j = Math.floor(r() * (i + 1)), swap i and j. The first
   five cells are sampled. 50 cells.
2. The reference `yes` cells not already sampled, sorted by question order,
   then file name, shuffled the same way. The first 10 are added (all of
   them if there are fewer). This keeps enough `yes` cells for the positive
   agreement, which a uniform sample of a rare answer does not.
3. The combined list (and the adjudicated cells) shuffled the same way once
   more. That is the order the human sees, so the position says nothing
   about how a cell was drawn.

The script that draws the sample is committed before it runs and writes the
drawn cell ids with the generator name and seed into
`reference/einv1/human-sample.jsonl`.

### Arms tested against Jev, named in advance

The best arm per family is still chosen after the fact for the map. A
claim of a difference to Jev, in either direction, is made only for two
arms named here, before any run:

- `gliner2.5-multi-decide` (GLiNER2.5 multi-Decide, question as task
  prompt)
- `winnow-12b`

For these two the exact McNemar test against Jev on the 340 cells counts at
p < 0.05, without a correction, since there are only two. Every other arm,
and the best-of-family arm when it is a different one, is reported
descriptively. Its p value is printed and labelled as chosen after the fact.

### Where rule 1 comes from

The `lieferdatum` and `leistungszeitraum` rule in step 1 was found in the
2026-09-22 review, and the cells of that review were selected by Jev's own
probability. The rule is kept because it describes the questions, not Jev:
a month of delivery is a delivery date, a billed period needs a start and an
end. It is kept as a written rule only. No cell, quote or verdict of that
review is given to a judge as an example, and none enters the reference.
