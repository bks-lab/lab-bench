# route1: routing with Jev, GLiNER2.5 Decide and a local LLM (2026-10-09)

route1 asks every arm to put a short text into one of a fixed set of classes:
the 18 MASSIVE scenarios, the MASSIVE intents of the gold scenario, and the
single-label tasks of Fastino's fast-decisions dev set. The question: on the
task GLiNER2.5 Decide is built for, where does it stand against Jev and
against the best local LLM of pv1?

Short answer:

- **MASSIVE, both languages:** Jev is first in every group. Winnow-12B is
  close behind: scenario de 71.8 % against 72.8 % (p = 0.110), intent de
  89.0 % against 90.7 % (p = 0.002), scenario en 70.6 % against 74.1 %
  (p < 0.001), intent en 91.6 % against 92.0 % (p = 0.575). The best GLiNER
  arm is 13.2 to 31.8 points behind Jev, every difference p < 0.001.
- **fast-decisions (English, 26 labels or fewer):** the four English Decide
  arms reach 63.3 to 64.4 %, Jev 63.0 %, Winnow-12B 63.6 %. None of these
  differences is significant (p 0.168 to 0.774 against Jev). Only the
  multilingual Decide model falls behind (57.7 %, 58.0 %, p < 0.001).
- **Speed:** median per decision GLiNER 14 to 27 ms, Winnow-12B 87 to 120 ms,
  Jev 230 to 240 ms. These are three different clocks: GPU time in-process
  for GLiNER, an HTTP call over the LAN for Winnow-12B, an HTTP round trip
  to the USA for Jev (see Speed).
- **Cost:** Jev costs USD 0.0212 to 0.0228 per 1,000 decisions at list
  price (median input tokens; USD 0.0187 to 0.0235 from the mean). The
  local arms ran on one RTX 4090 workstation; their cost (hardware, power)
  was not estimated, which does not make them free.

Every arm: run 1 only, 2026-10-09. Method, prompts and label texts:
[method.md](method.md). Full report with macro-F1 and every fast-decisions
task: [score.md](score.md).

## The arms

| arm | model | version as run | language |
|---|---|---|---|
| `jev` | TypeSafe Jev, called as `jev-latest` | `jev-1.13.0` on all 14,496 rows | de, en |
| `winnow-12b` | Gemma 4 12B with the Winnow LoRA, Q8_0, Ollama 0.35.0 | digest `b5bedb246945` | de, en |
| `gliner2.5-multi-decide`, `-intext` | fastino/GLiNER2.5-multi-Decide | revision `e173bca1f0c4` | multilingual |
| `gliner2.5-decide`, `-intext` | fastino/GLiNER2.5-Decide | revision `9d1bfb848cd1` | English |
| `gliner2.5-decide-1b`, `-intext` | fastino/GLiNER2.5-Decide-1B | revision `67af949e17be` | English |

`-intext` puts the question into the passage instead of the task prompt
(see pv1 [gliner.md](../pv1/gliner.md)). Versions are read from the result
rows (`arm_version`, `arm_digest`, `arm_revision`).

## Results

Accuracy against the gold labels, unanswered rows count as wrong. skill =
(acc - base) / (1 - base). "jev only / arm only" are the discordant items of
the paired exact McNemar test against Jev; the report prints p values below
0.0005 as 0.000, written here as < 0.001. "Most frequent answer" is the most
frequent gold label per option set, not an arm with a results file.

### MASSIVE, German (n = 2,974 scenario, 2,622 intent with 2+ options)

Source: [score.md](score.md), sections `massive scenario, de` and
`massive intent, 2+ options, de`.

| arm | scenario acc | skill | jev only / arm only | p vs Jev | intent 2+ acc | skill | jev only / arm only | p vs Jev |
|---|---|---|---|---|---|---|---|---|
| most frequent answer | 13.5 % | 0.000 | | | 50.2 % | 0.000 | | |
| jev | 72.8 % | 0.686 | | | 90.7 % | 0.812 | | |
| winnow-12b | 71.8 % | 0.674 | 192 / 161 | 0.110 | 89.0 % | 0.779 | 115 / 71 | 0.002 |
| gliner2.5-multi-decide | 48.5 % | 0.404 | 912 / 188 | < 0.001 | 58.9 % | 0.175 | 899 / 67 | < 0.001 |
| gliner2.5-multi-decide-intext | 46.7 % | 0.384 | 957 / 181 | < 0.001 | 58.4 % | 0.165 | 916 / 71 | < 0.001 |
| gliner2.5-decide (English model, extra) | 49.0 % | 0.410 | 856 / 147 | < 0.001 | 65.9 % | 0.316 | 709 / 61 | < 0.001 |
| gliner2.5-decide-intext (English model, extra) | 35.9 % | 0.259 | 1182 / 85 | < 0.001 | 58.0 % | 0.156 | 931 / 74 | < 0.001 |
| gliner2.5-decide-1b (English model, extra) | 41.1 % | 0.319 | 1149 / 205 | < 0.001 | 56.4 % | 0.125 | 967 / 69 | < 0.001 |
| gliner2.5-decide-1b-intext (English model, extra) | 40.2 % | 0.308 | 1183 / 212 | < 0.001 | 51.0 % | 0.017 | 1112 / 73 | < 0.001 |

Winnow-12B answered 2,621 of the 2,622 intent questions (one row without an
option letter, see Caveats).

### MASSIVE, English (n = 2,974 scenario, 2,622 intent with 2+ options)

Source: [score.md](score.md), sections `massive scenario, en` and
`massive intent, 2+ options, en`.

| arm | scenario acc | skill | jev only / arm only | p vs Jev | intent 2+ acc | skill | jev only / arm only | p vs Jev |
|---|---|---|---|---|---|---|---|---|
| most frequent answer | 13.5 % | 0.000 | | | 50.2 % | 0.000 | | |
| jev | 74.1 % | 0.701 | | | 92.0 % | 0.838 | | |
| winnow-12b | 70.6 % | 0.661 | 237 / 134 | < 0.001 | 91.6 % | 0.832 | 82 / 74 | 0.575 |
| gliner2.5-decide | 60.9 % | 0.547 | 543 / 149 | < 0.001 | 78.3 % | 0.565 | 428 / 71 | < 0.001 |
| gliner2.5-decide-intext | 49.8 % | 0.420 | 813 / 91 | < 0.001 | 70.5 % | 0.407 | 631 / 68 | < 0.001 |
| gliner2.5-decide-1b | 53.3 % | 0.460 | 786 / 167 | < 0.001 | 72.4 % | 0.446 | 572 / 60 | < 0.001 |
| gliner2.5-decide-1b-intext | 52.3 % | 0.449 | 817 / 169 | < 0.001 | 67.4 % | 0.345 | 704 / 59 | < 0.001 |
| gliner2.5-multi-decide | 52.0 % | 0.445 | 844 / 186 | < 0.001 | 63.7 % | 0.270 | 812 / 70 | < 0.001 |
| gliner2.5-multi-decide-intext | 50.1 % | 0.423 | 886 / 172 | < 0.001 | 64.1 % | 0.279 | 791 / 60 | < 0.001 |

With the one-option intent questions included (n = 2,974 per language,
`massive intent, de` and `massive intent, en` in score.md): Jev 91.8 % and
92.9 %, Winnow-12B 90.3 % and 92.6 %, base 56.1 %. The McNemar counts and
p values are the same as above, because every arm is right on a one-option
question.

### fast-decisions, English (n = 2,500 with 26 labels or fewer, 2,600 all)

Source: [score.md](score.md), sections `fastdec, tasks with 26 options or
fewer, en` and `fastdec, all single-label tasks, en`.

| arm | 26 or fewer acc | skill | jev only / arm only | p vs Jev | all tasks acc | skill | jev only / arm only | p vs Jev |
|---|---|---|---|---|---|---|---|---|
| most frequent answer | 31.2 % | 0.000 | | | 30.4 % | 0.000 | | |
| jev | 63.0 % | 0.461 | | | 63.0 % | 0.469 | | |
| winnow-12b | 63.6 % | 0.471 | 115 / 132 | 0.309 | 61.2 % | 0.443 | 180 / 132 | 0.008 |
| gliner2.5-decide | 63.3 % | 0.466 | 293 / 301 | 0.774 | 63.3 % | 0.473 | 299 / 307 | 0.776 |
| gliner2.5-decide-intext | 63.4 % | 0.468 | 313 / 325 | 0.663 | 63.6 % | 0.477 | 317 / 332 | 0.583 |
| gliner2.5-decide-1b | 63.4 % | 0.468 | 322 / 334 | 0.668 | 63.5 % | 0.475 | 328 / 339 | 0.699 |
| gliner2.5-decide-1b-intext | 64.4 % | 0.482 | 305 / 341 | 0.168 | 64.6 % | 0.492 | 308 / 349 | 0.119 |
| gliner2.5-multi-decide | 57.7 % | 0.385 | 440 / 309 | < 0.001 | 58.0 % | 0.396 | 449 / 317 | < 0.001 |
| gliner2.5-multi-decide-intext | 58.0 % | 0.389 | 439 / 315 | < 0.001 | 58.2 % | 0.400 | 447 / 322 | < 0.001 |

The "all tasks" column includes `support_intent:intent` (28 labels), which
Winnow-12B cannot be asked (26 letters): its 100 rows are unanswered and
count as wrong, hence 2,500 answered and 0.0 % on that task. The
like-for-like column is "26 or fewer".

Per task (100 rows each, accuracy, not tested for significance, score.md
`fastdec per task, en`): the English Decide arms are ahead of Jev on
`ticket_route:queue` (51.0 to 55.0 % against 37.0 %, and the two
multilingual Decide arms 56.0 %) and
`product_feedback:feedback_type` (67.0 to 73.0 % against 57.0 %). Jev and
Winnow-12B are ahead on `sports_recap:sport` (75.0 % and 77.0 % against 54.0
to 63.0 %) and `sports_recap:upset` (70.0 % and 68.0 % against 51.0 to
64.0 %). Winnow-12B is first on `ticket_route:contains_pii` (80.0 %, Jev
69.0 %).

## Speed

Median latency per decision; every arm makes one call per question. Source:
a script over the result rows (`results/route1/<arm>/2026-10-09-run1.jsonl`,
fields `latency_ms` and `compute_ms`), per source and language. The `p50 ms`
column of [score.md](score.md) is the same median per question group.

| arm | MASSIVE de | MASSIVE en | fastdec en | all rows | what it measures |
|---|---|---|---|---|---|
| jev | 233 ms | 230 ms | 240 ms | 233 ms | HTTP round trip from the Mac to api.typesafe.ai, four requests in parallel |
| winnow-12b | 87 ms | 90 ms | 120 ms | 90 ms | HTTP call from the Mac to Ollama on the workstation, LAN included; Ollama's own compute time (`compute_ms`) 81, 81, 107 and 83 ms |
| gliner2.5-multi-decide | 14 ms | 14 ms | 14 ms | 14 ms | in-process on the GPU, `cuda.synchronize` around the call |
| gliner2.5-multi-decide-intext | 14 ms | 14 ms | 14 ms | 14 ms | same |
| gliner2.5-decide | 26 ms | 26 ms | 26 ms | 26 ms | same |
| gliner2.5-decide-intext | 26 ms | 26 ms | 26 ms | 26 ms | same |
| gliner2.5-decide-1b | 17 ms | 17 ms | 27 ms | 18 ms | same |
| gliner2.5-decide-1b-intext | 18 ms | 18 ms | 26 ms | 18 ms | same |

The three clocks are different measurements: Jev's includes the network to
the USA, GLiNER's includes no network at all. All local arms ran one
question at a time on one RTX 4090; batching was not tried.

Unlike pv1, where the long CV state made a local line 7 to 10 times slower
than Jev ([speed-cost.md](../pv1/speed-cost.md)), route1's input is a short
utterance, and the local LLM answers faster per decision than the Jev round
trip.

## Jev cost

Computed the pv1 way: median input tokens x USD 0.042 per million input
tokens, output free (docs.typesafe.ai/models, read 2026-10-02). A decision
is one call; a request is the sum of the calls of one source row (two for
MASSIVE, one to four for fast-decisions).

| source | lang | decisions | median input tokens per decision | USD per 1,000 decisions | median input tokens per request | USD per 1,000 requests |
|---|---|---|---|---|---|---|
| massive | de | 5,948 | 510 | 0.0214 | 894 | 0.0375 |
| massive | en | 5,948 | 504 | 0.0212 | 888 | 0.0373 |
| fastdec | en | 2,600 | 544 | 0.0228 | 633 | 0.0266 |

Spend over many decisions follows the mean, not the median. Mean input
tokens per decision are 448.5, 445.8 and 560.0, which gives USD 0.0188,
0.0187 and 0.0235 per 1,000 decisions. On MASSIVE the mean is below the
median because the shorter intent calls pull it down.

Per request: the `Jev cost` table of [score.md](score.md). Per decision: a
script over `results/route1/jev/2026-10-09-run1.jsonl` (private, not part of this export) (`tokens_in` per
row). For comparison, a pv1 line (five questions over a CV) costs USD 0.074
per 1,000 lines ([speed-cost.md](../pv1/speed-cost.md)). No call needed a
retry (`attempts` is 1 on every row). The local arms have no cost line: the
workstation's hardware and power were not estimated.

## Reading it

- **Where GLiNER is strong:** English fast-decisions, the dataset Fastino
  built and calls "the classification suite behind GLiNER2.5-Decide". There the English Decide models are
  level with Jev and Winnow-12B at about a tenth of Jev's latency per
  decision (a different clock, see Speed), with no text leaving the
  machine. Fastino's own table puts Decide ahead of "JevK5" on this suite
  (60.2 % against 57.6 %, on the unpublished test split, quoted in
  [gliner.md](../pv1/gliner.md)). Whether JevK5 is the `jev-1.13.0` run
  here is not known; on the dev split here they are level. It is home ground
  either way (see Caveats).
- **Where GLiNER is weak:** MASSIVE, a dataset of the same kind (short
  commands into a fixed label set) that Fastino does not name as a Decide
  source (whether its training suite contains MASSIVE is not known, see
  [method.md](method.md)). Every GLiNER arm is 13.2 to 31.8 points behind Jev, and on German
  intent the best multilingual arm has a skill of 0.175 over always
  answering the most frequent intent. Moving the question into the passage
  (`-intext`) mostly hurts. It gains 1.0 point for the 1B model on
  fast-decisions (63.4 to 64.4 %) and less than 0.5 points in three other
  cells (multi-Decide English intent and fast-decisions, Decide
  fast-decisions); none of these gains was tested for significance.
- **Where Jev is strong:** first in all four MASSIVE groups (German and
  English) and level with the best on fast-decisions (English only). No arm
  was tuned per task; the same label texts went to every arm.
  Its lead over Winnow-12B is significant on English scenario and German
  intent, not on German scenario and English intent.
- **Where the local LLM is strong:** Winnow-12B, a 12B model on one consumer
  GPU, stays within 0.3 to 3.5 points of Jev on MASSIVE and level with it on
  fast-decisions, at under 100 ms per MASSIVE decision on the workstation.
  Its limits are the letter scheme (26 options at most) and that it
  sometimes starts with prose instead of a letter (Caveats).

## Caveats

- **Oracle scenario for intent.** The intent question offers only the
  intents of the gold scenario. It measures intent given the right scenario,
  not end-to-end routing, and it hands every arm the scenario indirectly.
- **One-option intent questions excluded.** Cooking, news and weather have a
  single intent: 704 of the 5,948 intent questions (352 per language) are
  right by construction. The tables above use the `2+ options` group; the
  full group is quoted for reference.
- **support_intent has 28 labels.** The local arm's letter scheme stops at
  26, so `support_intent:intent` is unanswered for Winnow-12B (100 rows,
  `too many options (28)`). Compare on the "26 or fewer" group.
- **fast-decisions may be close to the Decide training data.** Fastino built
  the dataset, and its card calls it "the classification suite behind
  GLiNER2.5-Decide". Whether these dev rows were used in training is not
  stated. A Decide lead or tie there is not a neutral result.
- **English-only Decide models on German rows** are a labelled extra
  ("English model, extra" in score.md), not a fair German result, and are
  not counted as the best GLiNER arm for German.
- **MASSIVE is public** (since 2022), and so are the fast-decisions dev rows.
  No arm was checked for exposure to them.
- **Single run.** Run 1 of each arm only; no shuffled run (option order) and
  no repeat runs (flip rates).
- **Winnow-12B letter mass.** Beyond the 100 unaskable rows there is one real
  failure: `massive-11650-de` intent, no option letter in the top 20 logprobs
  (first token "None"). On 449 further rows the option letters together had
  less than 0.5 probability: 422 MASSIVE intent rows (201 de, 221 en, 41 of
  them one-option questions) and 27 fast-decisions rows (sports_recap 15,
  document_type 7, clinic_request 3, paper_field 1, support_topic 1). The
  most frequent first tokens of these 450 rows are "The" (357), "None" (66)
  and "C" (13): the model wanted to start with prose. The answer is still
  the most probable letter, renormalised from a small mass. Counted by a
  script over the result rows.
- **Latency is not one measurement** across arms (see Speed).

## Addendum 2026-10-09: Classic baseline

Test T3, design committed before the run: [plan/t3.md](../../plan/t3.md).
A new tool family `classic`: `intfloat/multilingual-e5-base` sentence
embeddings (revision `d1287505`, prefix `query: `, L2-normalised) plus a
scikit-learn `LogisticRegression(C=1.0, max_iter=2000)`, multinomial. One
classifier for scenario and one for intent per language, trained on the
MASSIVE 1.1 train split of the same locale (11,514 rows each, not
committed; tarball SHA-256 and a hash of the training rows are in every
result row). No tuning, no dev split. Unlike every other arm, this one
has seen labelled examples of exactly these classes: it is the home ground
of the family, and the numbers measure what labelled data buys.

| arm | training rows |
|---|---|
| `classic-e5-lr` | full train split: 11,514 per classifier |
| `classic-e5-lr-10shot` | 10 per class, seeded: 180 for scenario, 594 for intent (`cooking_query` has only 4 train rows) |

Run on a Mac (Apple silicon, MPS) with Python 3.14.8, torch 2.14.1,
sentence-transformers 6.1.0, scikit-learn 1.9.1. Fitting took about a
minute per language on the full split. Intent is scored exactly like the
other arms: the classifier's 60 intent probabilities are cut to the
options of the gold scenario and renormalised.

Source: [score.md](score.md). McNemar against Jev, exact, two-sided.

| group | jev | classic-e5-lr | jev only / arm only | p vs Jev | classic-e5-lr-10shot | jev only / arm only | p vs Jev |
|---|---|---|---|---|---|---|---|
| scenario, de | 72.8 % | **86.9 %** | 141 / 559 | < 0.001 | 70.8 % | 398 / 339 | 0.033 |
| scenario, en | 74.1 % | **89.8 %** | 92 / 560 | < 0.001 | 73.2 % | 346 / 318 | 0.295 |
| intent 2+ options, de | **90.7 %** | 86.9 % | 243 / 144 | < 0.001 | 74.0 % | 548 / 110 | < 0.001 |
| intent 2+ options, en | 92.0 % | 91.2 % | 159 / 140 | 0.298 | 77.7 % | 466 / 93 | < 0.001 |

Flat 60-way intent (argmax over all intents, no option list, an extra that
no other arm was asked): 77.0 % German and 83.5 % English for the full
arm, 63.0 % and 67.4 % for the 10-shot arm. This is the end-to-end number
a router would see; it is not comparable to the intent rows above.

Median latency per question: 24 ms (full) and 25 ms (10-shot), one item at
a time on the Mac, embedding plus predict_proba. A third clock next to the
ones in Speed.

Reading it:

- **Scenario:** with the train split the classic arm is 14 to 16 points
  ahead of Jev in both languages (p < 0.001), and ahead of every zero-shot
  arm. The 18 scenarios are coarse topics that a linear probe on good
  embeddings separates well once it has seen examples.
- **Intent given the scenario:** the classic arm does not catch up. It is
  3.8 points behind Jev in German (significant) and level in English (0.8
  points, p = 0.298). Its macro-F1 on intent (72.5 % de, 79.6 % en against
  Jev's 86.5 % and 88.4 %) shows where it loses: the rarer intents, where
  the label texts help a zero-shot model and about 190 training rows per
  class do not suffice for a linear probe.
- **Data hunger:** with 10 examples per class the classic arm falls to
  Jev's level on scenario (German 2.0 points behind, p = 0.033; English
  level, p = 0.295) and 14 to 17 points behind on intent. The full-data lead
  on scenario is the data, not the model.
- In the tool map the classic family is reported as its own cell. The top
  family of a row is still chosen among the three zero-shot families, so
  no existing cell changed; the classic cell carries its test against Jev
  and against that top (`classic_note` in `results/toolmap.json`).
- Not run: fast-decisions (no train split published), pv1, einv1 (no
  training labels), ex1 (a slot tagger is a different model, `open`).

Self-check: scenario de accuracy (86.9 %), English intent 2+ accuracy
(91.2 %) and the German scenario McNemar counts (141 / 559) were recounted
by a separate Python script straight from the result rows and the gold
files, and match the report.
