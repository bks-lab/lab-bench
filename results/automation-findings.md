# T1 findings: how much can a tool decide alone?

Written 2026-10-09 from the output of `node bench/score-automation.mjs`
(reports `results/route1/automation.md`, `results/pv1/automation.md`) and
the new toolmap rows `route1-automation95-de/en`, `pv1-automation95-de/en`.
Design, fixed before any number: `plan/t1.md`. Run 1 of every arm, no new
runs.

## Short answer

At a 95 % accuracy target, Jev and the best local LLM can each take over
roughly half of the cases on their own; GLiNER takes a small share.

- route1 (MASSIVE routing, plus fast-decisions in English): Jev 60.3 % (de)
  and 42.6 % (en) of items, Winnow-12B 58.0 % and 41.3 %, the best GLiNER
  arm 4.4 % and 10.8 %.
- pv1 (job requirement line questions): the best local LLM (Qwen3.8 27B)
  68.2 % (de) and 56.4 % (en), Jev 53.5 % and 51.9 %, the best GLiNER arm
  28.5 % and 24.6 %.
- The most-frequent-answer rule automates 0 % at 95 % everywhere, so the
  skill of each cell equals its rate.
- With a threshold chosen on the other half of the data (cross-fitted), the
  route1 rates hold almost exactly (n over 5,000 items). On pv1 (600 items
  per language) the rates of the best arms move by up to 6 points and the passed items land
  between 93.6 % and 95.0 % accurate, so a 95 % threshold on pv1 is not
  stable at this size.
- Calibration: on route1 Jev and Winnow are overconfident in their top bin
  (Jev says 0.98 and is right 93.0 % de, 88.8 % en; Winnow says 0.99 and is
  right 85.3 % de, 81.0 % en). GLiNER's probabilities are closer to its hit
  rate (lowest ECE) but rarely high, so they pass few items.

## Pooled automation rate per family, in-sample

Best arm per family on the pooled item set, chosen after the fact (as every
toolmap cell). English-only GLiNER models are not eligible on German rows.
`auto95` / `auto90`: share of items passed at 95 % / 90 % accuracy.
`cf95`: cross-fitted rate, in brackets the accuracy of the passed items.

### route1, de (MASSIVE scenario plus intent with 2+ options, 5,596 items)

| family | best arm | acc | auto95 | auto90 | cf95 (acc) | ECE | Brier |
|---|---|---|---|---|---|---|---|
| Jev | jev | 81.2 % | 60.3 % | 78.2 % | 60.3 % (95.2 %) | 0.082 | 0.285 |
| local LLM | winnow-12b | 79.8 % | 58.0 % | 75.4 % | 57.9 % (95.1 %) | 0.158 | 0.353 |
| GLiNER | gliner2.5-multi-decide | 53.4 % | 4.4 % | 12.7 % | 4.8 % (94.8 %) | 0.067 | 0.601 |
| baseline | most frequent answer | 30.7 % | 0.0 % | 0.0 % | 0.0 % | | |

### route1, en (the same plus fast-decisions, 8,196 items)

| family | best arm | acc | auto95 | auto90 | cf95 (acc) | ECE | Brier |
|---|---|---|---|---|---|---|---|
| Jev | jev | 76.3 % | 42.6 % | 64.7 % | 42.6 % (95.2 %) | 0.130 | 0.361 |
| local LLM | winnow-12b | 74.4 % | 41.3 % | 58.7 % | 41.9 % (94.9 %) | 0.197 | 0.435 |
| GLiNER | gliner2.5-decide-1b | 62.6 % | 10.8 % | 23.3 % | 9.2 % (94.7 %) | 0.061 | 0.496 |
| baseline | most frequent answer | 30.6 % | 0.0 % | 0.0 % | 0.0 % | | |

### pv1, de (five line questions against `adjudicated-a`, 600 items)

| family | best arm | acc | auto95 | auto90 | cf95 (acc) | ECE | Brier |
|---|---|---|---|---|---|---|---|
| local LLM | qwen3.8-27b | 85.2 % | 68.2 % | 90.5 % | 70.1 % (94.8 %) | 0.032 | 0.205 |
| Jev | jev | 87.0 % | 53.5 % | 90.8 % | 53.4 % (95.0 %) | 0.073 | 0.222 |
| GLiNER | gliner2.5-multi-decide-focus | 51.5 % | 28.5 % | 38.5 % | 24.0 % (94.5 %) | 0.188 | 0.636 |
| baseline | most frequent answer | 55.3 % | 0.0 % | 19.3 % | 9.7 % (91.7 %) | | |

### pv1, en (five line questions against `claude-c`, 615 items)

| family | best arm | acc | auto95 | auto90 | cf95 (acc) | ECE | Brier |
|---|---|---|---|---|---|---|---|
| local LLM | qwen3.8-27b | 82.3 % | 56.4 % | 85.7 % | 59.2 % (93.6 %) | 0.046 | 0.241 |
| Jev | jev | 84.7 % | 51.9 % | 79.3 % | 58.0 % (93.8 %) | 0.037 | 0.228 |
| GLiNER | gliner2.5-decide-1b-focus | 52.8 % | 24.6 % | 38.9 % | 25.9 % (94.3 %) | 0.132 | 0.576 |
| baseline | most frequent answer | 54.5 % | 0.0 % | 19.4 % | 0.0 % | | |

## Per item set (auto95, best arm per family)

| family, lang | set | n | Jev | GLiNER | local LLM |
|---|---|---|---|---|---|
| route1 de | MASSIVE scenario | 2,974 | 43.5 % | 0.2 % | 42.4 % |
| route1 de | MASSIVE intent, 2+ options | 2,622 | 86.5 % | 9.2 % | 81.0 % |
| route1 en | MASSIVE scenario | 2,974 | 50.4 % | 9.8 % | 42.2 % |
| route1 en | MASSIVE intent, 2+ options | 2,622 | 91.6 % | 42.3 % | 90.4 % |
| route1 en | fast-decisions, all single-label tasks | 2,600 | 0.0 % | 8.5 % | 0.0 % |
| pv1 de | is_req | 136 | 80.9 % | 48.5 % | 86.0 % |
| pv1 de | axis | 116 | 94.0 % | 5.2 % | 84.5 % |
| pv1 de | evidence | 116 | 62.9 % | 0.0 % | 73.3 % |
| pv1 de | level | 116 | 30.2 % | 1.7 % | 28.4 % |
| pv1 en | is_req | 139 | 77.7 % | 61.2 % | 87.1 % |
| pv1 en | axis | 119 | 87.4 % | 34.4 % | 84.0 % |
| pv1 en | evidence | 119 | 70.6 % | 0.8 % | 60.5 % |
| pv1 en | level | 119 | 37.8 % | 16.8 % | 21.9 % |

`must` is left out of this table: its majority answer reaches 90 % on its
own (baseline auto90 100 %), so a high rate there says little. The best arm
per set can differ from the pooled best arm; the reports list every arm.

## What it means in practice

- A confidence threshold is a usable lever for Jev and the local LLMs:
  about half of the routing and matching cases could go through without a
  person at 95 % accuracy, the rest to review. The rate depends far more on
  the task than on the tool: 81 to 92 % of MASSIVE intent decisions (given
  the scenario), 42 to 50 % of scenario decisions, 22 to 38 % of pv1 level
  scores (Jev and the best local arm).
- On fast-decisions neither Jev nor Winnow reaches 90 % at any threshold:
  their answers with a stated probability of 0.9 or more are right 75.6 %
  (Jev) and 69.2 % (Winnow) of the time. GLiNER Decide passes 8.5 % at 95 %, on a suite its maker built
  (see the toolmap note on home ground).
- GLiNER's probabilities are the most honest of the three in route1 (ECE
  0.06 to 0.07) but its accuracy is low and its probabilities rarely high,
  so a 95 % threshold passes little. On pv1 the best GLiNER arms never
  state 0.9 or more.
- Jev's and Winnow's high probabilities are optimistic on route1. A
  threshold has to be set from labelled data, not read off the probability:
  for Jev on route1 en the 95 % threshold is a stated probability of 1.0.
  The thresholds found are in `results/toolmap.json` (`threshold`).
- One global threshold over all questions was measured. A threshold per
  question would likely pass more, and was not part of the design.

## Caveats

- In-sample rates are optimistic by construction. The cross-fitted rates
  are the honest number for a new batch; on pv1 they show that 600 items
  do not pin a 95 % threshold down.
- The best arm per family is chosen after the fact, which favours GLiNER
  (up to nine arms) and the local family on pv1 (four arms).
- One run per arm; route1 options were not shuffled.
- pv1 accuracy is agreement with a model judge (German: adjudicated,
  English: a single Opus judge), not with a human sample.
- The probabilities are distributions over the offered options: a softmax
  over labels for GLiNER, next-token letter probabilities renormalised
  over the letters for the local LLMs (median letter mass before
  renormalising 0.998 to 1.0, so renormalising hides little here), and
  Jev's returned probabilities. None is a calibrated claim that an answer
  is right; ECE and Brier measure how far each is from one.
- Unanswered rows (Winnow on the 28-option fast-decisions task, one row
  without a letter) count as wrong and are never automated.
- No significance test on the automation rows.

## Addendum 2026-10-09: classic classifier (T3)

After T3 was merged, `bench/score-automation.mjs` was rerun so the classic
arms (`plan/t3.md`) appear as their own family in `results/route1/automation.md`.
They were not part of the T1 design, which predates them, so this is an
addition and the T1 numbers above are unchanged.

- Trained on the full MASSIVE train split, `classic-e5-lr` automates 76.4 %
  (de, pooled) at 95 % accuracy, against 60.3 % for Jev and 58.0 % for
  Winnow-12B. Its ECE is 0.174, so it is overconfident too, but its ranking
  of its own answers is good enough to pass more items.
- With 10 examples per class (`classic-e5-lr-10shot`) it automates 2.1 %:
  the probabilities of a classifier trained on so little carry almost no
  usable ranking.
- In `results/toolmap.json` the classic family is shown beside the zero-shot
  families and is not ranked with them (`top_family` is computed among Jev,
  GLiNER and the local LLM), because it needs labelled data the others do
  not. pv1 has no train split, so the classic cell there is not applicable.
