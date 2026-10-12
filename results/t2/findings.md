# T2 findings: stability under shuffled options

Design `plan/t2.md` (committed 2026-10-12 before any T2 run), runner
`bench/t2/run_route.py`, scorer `bench/t2/score_t2.py t2`, shared rules
`plan/c0-common.md`. Runs: 2026-10-12, workstation queue, Ollama 0.35.0 for
the LLM arms and gliner2 2.0.0 for GLiNER. Data: route1, MASSIVE de, 2,974
requests, 5,948 questions (scenario out of 18, intent given the gold scenario).
Two passes per job in one model load: options in definition order (`plain`)
and in the seed 1002 shuffle of `withShuffledOptions` (`shuffled`). Scores:
`results/t2/score.md`, `results/t2/score.json`; compact rows (line, question,
pass, choice) in `results/t2/<arm>/`, so the table can be recomputed with
`bench/t2/score_t2.py` from the repository.

## Sanity gate

The plain pass of the port on the first 250 requests (500 questions) agreed
with the committed route1 run 1 of the same arm on 100 % of the choices, for
winnow-12b and for gliner2.5-multi-decide (minimum 99 %). Passed.

## Result

| arm | flip, plain vs route1 run 1 | flip, shuffled vs plain (95 % CI) | acc plain | acc shuffled | McNemar p |
|---|---|---|---|---|---|
| winnow-12b | 0.0 % | 7.8 % [7.2, 8.5] | 81.0 % | 82.0 % | 0.001 |
| gemma4-12b | 0.0 % | 9.1 % [8.5, 9.8] | 79.0 % | 79.0 % | 1.000 |
| qwen3.8-27b | 0.0 % | 5.7 % [5.1, 6.2] | 82.0 % | 83.1 % | 0.0001 |
| gliner2.5-multi-decide | 0.0 % | 0.0 % [0.0, 0.0] | 56.1 % | 56.1 % | 1.000 |

Intervals: bootstrap over utterances (both questions together), 1,000
resamples, seed 20261009.

- **A local LLM changes its answer on 6 to 9 % of the questions when only the
  option order changes.** gemma4-12b flips 9.1 %, winnow-12b 7.8 %,
  qwen3.8-27b 5.7 %. qwen3.8-27b is clearly lowest; the intervals of
  winnow-12b (7.2 to 8.5) and gemma4-12b (8.5 to 9.8) only touch.
- **The flips sit in the scenario question.** Scenario (18 options): 13.9 %
  (winnow-12b), 14.4 % (gemma4-12b), 10.4 % (qwen3.8-27b). Intent (the
  options of one scenario): 1.7 %, 3.9 % and 1.0 %. More options, more order
  effect.
- **The flips are not neutral for accuracy in two of three arms.** The
  shuffled order scores 1.0 points higher for winnow-12b (190 questions right
  only when shuffled against 132 only when plain, p 0.001) and 1.1 points
  higher for qwen3.8-27b (165 against 101, p 0.0001); gemma4-12b is symmetric
  (182 against 182). The route1 definition order is therefore not the best
  order for two arms, and accuracy of run 1 depends partly on an arbitrary
  choice. The difference is about one point, small against the flip rate.
- **Run-to-run noise on another day is zero.** The plain pass reproduces the
  route1 run 1 of 2026-10-09 on every question for all four arms (flip 0.0 %).
  At temperature 0 with one token, a run three days later gave identical
  choices. So the flips above are caused by the order, not by the sampler.
- **GLiNER Decide shows no flip at all** (0 of 5,948), and its probabilities
  are identical to four decimals in both passes. See the anomalies.

## Anomalies

- GLiNER 0 flips: the runner hands the labels to the classifier in the
  shuffled order (the schema is built from the shuffled `criteria`), the rows
  store options in definition order, and every probability is identical to
  four decimals. That suggests the classifier scores each label independently
  of the others in its input, but we did not verify that in the library
  source or with a control; treat the 0.0 % as measured and the explanation as
  a guess.
- winnow-12b had one unanswered question per pass (error counted as a flip;
  it is in the 7.8 %). gemma4-12b and qwen3.8-27b had none. 201 and 196 winnow
  and 114 and 119 gemma questions had less than half of the probability mass
  on the option letters (qwen3.8-27b: 0); they are answered by the letters
  that did appear and are in the numbers.
- Jev (hosted API) was not run; it needs Michael Boiman's go.
- Only German MASSIVE, only the shuffle of one seed (1002); a different
  permutation gives a different set of flips, the rate is what is measured.
