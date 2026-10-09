# Test bench roadmap

Written 2026-10-09, before any of the tests below ran. A test moves from
"planned" to "measured" only through a dated design file in `plan/` that is
committed before its first run (see `plan/einv1.md` for the form).

## Principle: every tool family gets its home ground

A comparison that only asks what one tool is built for flatters that tool.
So every family in the tool map gets at least one test that plays to its
declared strength, named here in advance, next to the tests where it is a
guest. A family that loses on its own home ground is reported as such.

| Family | Declared strength | Home-ground test (status) |
|---|---|---|
| Jev | typed judgments with probabilities over a state | pv1 match (measured), route1 (measured), T1 automation rate (planned), T6 e-invoice yes/no (planned) |
| GLiNER Decide | fast classification into given labels | fast-decisions (measured), T4 trained on own labels (planned), T10 batched throughput (planned) |
| GLiNER extraction and PII | finding spans in text | ex1 slots (measured), T7 personal data (planned) |
| Local LLM | general purpose, runs in house, structured output | ex1 slots (measured), T8 invoice fields (planned), T11 long documents (planned) |
| Classic trained classifier (new) | cheap and strong when labelled data exists | T3 (planned) |
| Hosted LLM (new) | general purpose, no setup | T5 (planned) |

## Planned tests

| # | Test | Adds | Data | Effort |
|---|---|---|---|---|
| T1 | Automation rate: share of cases a tool can decide alone at 95 % accuracy, and calibration (ECE, Brier) of its probabilities | row | existing route1 and pv1 rows, no new runs | small |
| T2 | Stability: second run with shuffled options, flip rate | row | route1 | small |
| T3 | Classic baseline: multilingual sentence embeddings plus logistic regression trained on the MASSIVE train split | column | MASSIVE train/test | small |
| T4 | GLiNER Decide trained with 10, 50 and all examples per class | variant | MASSIVE train | medium |
| T5 | Hosted LLM on route1 and ex1 (not on pv1, where a Claude model is the judge) | column | route1, ex1 | small |
| T6 | E-invoice yes/no over free text | row | KoSIT, see `plan/einv1.md` | medium |
| T7 | Personal data detection in German text | row | public PII set, licence to check | medium |
| T8 | Invoice fields from rendered KoSIT invoices, XML fields as gold | row | KoSIT (Apache-2.0) | medium |
| T9 | Robustness: typos and colloquial wording in MASSIVE de | row | MASSIVE de, perturbed | small |
| T10 | Throughput with batching | row | route1 | small |
| T11 | Long documents: classify texts beyond the encoder context | row | to choose | medium |

Order: T1, T3 and T5 first (cheap, they close the largest gaps), then T4 and
T2, then T6 to T11.
