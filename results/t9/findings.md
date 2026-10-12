# T9 findings: robustness to typos and colloquial wording

Design `plan/t9.md` (committed 2026-10-12 before any T9 run), perturbation
rules `bench/t9/perturb.py` (frozen inputs `cases/t9/`), runner
`bench/t2/run_route.py --requests cases/t9/requests.jsonl`, scorer
`bench/t2/score_t2.py t9`. Runs: 2026-10-12, workstation queue. Data: 1,000
MASSIVE de requests, each in four variants (`clean`, `typo`, `translit`,
`colloquial`), 8,000 questions per arm, plain option order, the clean variant
re-run in the same job. Scores: `results/t9/score.md`, `results/t9/score.json`;
compact rows in `results/t9/<arm>/`.

## Sanity gate

Shared with T2: the port agreed with route1 run 1 on 100 % of 500 questions
for winnow-12b and gliner2.5-multi-decide. Passed.

## Result

Accuracy on the clean variant: winnow-12b 80.3 %, gemma4-12b 79.0 %,
qwen3.8-27b 82.0 %, gliner2.5-multi-decide 56.2 %. Drop against clean is
paired on the same utterances; the interval is the 95 % bootstrap over
utterances (all variants together), 1,000 resamples, seed 20261009. "Flip" is
the share of questions whose choice differs from clean.

| arm | variant | acc | drop vs clean (95 % CI) | flip vs clean |
|---|---|---|---|---|
| winnow-12b | typo | 77.3 % | 3.0 pt [1.9, 4.0] | 8.6 % |
| winnow-12b | translit | 81.1 % | -0.8 pt [-1.3, -0.2] | 2.0 % |
| winnow-12b | colloquial | 80.5 % | -0.2 pt [-1.1, 0.6] | 5.4 % |
| gemma4-12b | typo | 74.7 % | 4.2 pt [3.0, 5.4] | 9.2 % |
| gemma4-12b | translit | 78.8 % | 0.2 pt [-0.4, 0.7] | 1.9 % |
| gemma4-12b | colloquial | 77.5 % | 1.5 pt [0.3, 2.6] | 7.6 % |
| qwen3.8-27b | typo | 78.3 % | 3.6 pt [2.6, 4.7] | 7.1 % |
| qwen3.8-27b | translit | 81.5 % | 0.4 pt [0.1, 0.8] | 1.1 % |
| qwen3.8-27b | colloquial | 82.0 % | 0.0 pt [-0.8, 0.8] | 4.7 % |
| gliner2.5-multi-decide | typo | 52.8 % | 3.5 pt [2.1, 4.8] | 18.2 % |
| gliner2.5-multi-decide | translit | 54.5 % | 1.8 pt [0.9, 2.5] | 7.0 % |
| gliner2.5-multi-decide | colloquial | 55.6 % | 0.7 pt [-0.2, 1.5] | 7.2 % |

- **Typos cost every arm 3 to 4 points**, and the intervals exclude zero in
  all four arms: gemma4-12b 4.2, qwen3.8-27b 3.6, gliner2.5-multi-decide 3.5,
  winnow-12b 3.0. The four are not separated from each other at this sample
  size (intervals overlap). The rule changed 997 of 1,000 utterances (one
  edit in about every seventh long word, at least one per utterance), so this
  is a plain-typing-slip load, not a worst case.
- **GLiNER Decide's answers move the most**: 18.2 % of its questions change
  with typos, against 7.1 to 9.2 % for the LLMs, while its net accuracy loss
  (3.5 points) is similar. Many of its changed answers cancel (wrong to
  right and right to wrong), which also reflects a clean accuracy of only
  56 %.
- **Missing umlauts (ae, oe, ue, ss) cost almost nothing**: -0.8 to +1.8
  points. The rule changed 404 utterances; restricted to those, the drop is
  -1.9 (winnow-12b, an improvement), 0.4, 1.0 and 4.3 points (GLiNER).
  GLiNER has the largest loss of the four.
- **Colloquial wording** (contractions and one opening filler on every
  utterance) costs gemma4-12b 1.5 points [0.3, 2.6] and the rest within noise
  (-0.2, 0.0, 0.7), although 4.7 to 7.6 % of the answers flip.
- The ranking of the arms by accuracy is the clean ranking in every variant:
  qwen3.8-27b, winnow-12b, gemma4-12b, then GLiNER far behind. Robustness is a
  property of how much each loses, and no arm is clearly better or worse there
  except that GLiNER flips most.

## Anomalies

- Negative drops (winnow-12b translit -0.8 [-1.3, -0.2]) mean the perturbed
  variant scored higher than clean; with 1,000 utterances this is noise or
  a lucky change of a few answers, and is reported as measured.
- Errors: none in any arm. Questions with less than half of the probability
  mass on the option letters: winnow-12b 254, gemma4-12b 140, qwen3.8-27b 3,
  GLiNER 0 (of 8,000).
- The perturbations are rule-based: a QWERTZ neighbour, drop, swap or double
  per word, a fixed list of contractions, five fillers. They do not model
  real speech-to-text errors, dialect or code-switching, so the numbers are a
  floor for what real messy input costs, not an estimate of it.
- Jev was not run (hosted API, needs Michael Boiman's go; `bench/run-jev.mjs`
  would first need a `--requests` option).
