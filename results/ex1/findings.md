# ex1: slot extraction with GLiNER and a local LLM (2026-10-09)

ex1 asks every arm to find the slots of a short spoken command (dates,
times, places, names and the other MASSIVE slot types) with the words of
the utterance that fill them. The question: can a GLiNER entity extractor,
a general span extractor that takes the slot types as labels, keep up with
the local LLM of route1? Jev is not in
this test: it has no extraction primitive.

Short answer: no. Winnow-12B reaches a slot F1 of 45.1 % (de) and 48.9 %
(en). The best GLiNER arm reaches 22.5 % in German (GLiNER2.5-multi) and
38.1 % in English (GLiNER2-large, an English model). On whole utterances
only Winnow-12B beats predicting nothing (exact frame 45.1 % and 44.9 %
against 33.5 %); every GLiNER arm is below that floor. GLiNER is 13 to 21
times faster per utterance, measured on different clocks (GPU time
in-process for GLiNER, an HTTP call over the LAN for Winnow-12B, see Speed).

Every arm: run 1 only, 2026-10-09, 1,000 MASSIVE 1.1 test utterances per
language. Method, prompt and label texts: [method.md](method.md). Full
report with every slot type: [score.md](score.md).

## The arms

| arm | model | version as run | language |
|---|---|---|---|
| `winnow-12b` | Gemma 4 12B with the Winnow LoRA, Q8_0, Ollama 0.35.0, JSON output | digest `b5bedb246945` | de, en |
| `gliner2.5-multi` | fastino/gliner2.5-multi-v1 entity extractor | revision `cf5593a5d45e` | de, en |
| `gliner2-large` | fastino/gliner2-large-v1 entity extractor | revision `d7aa8a2850d1` | en only |
| `jev` | not applicable: TypeSafe's API offers choice, score and noul and returns no text spans | | |

Versions are read from the result rows (`arm_version`, `arm_digest`,
`arm_revision`). All arms get the same 53 slot label texts, made by one
mechanical rule from the MASSIVE names (`place_name` becomes `place name`).

## Results

A slot is the pair (slot type, normalised text); P, R and F1 are micro over
all utterances. Exact frame is the share of utterances whose predicted slots
equal the gold slots, an utterance without gold slots counting when the arm
predicts nothing. "No extraction" predicts nothing: F1 0, exact frame 33.5 %
(the share of utterances without a slot), not an arm with a results file.
skill over the baseline: for F1 it equals F1, because the baseline F1 is 0;
for exact frame it is (frame - 33.5) / (100 - 33.5), computed from the
report values. GLiNER at the library default threshold 0.5, fixed before
the run.

### German (1,000 utterances, 941 gold slots)

Source: [score.md](score.md), section `de`.

| arm | predicted | P | R | F1 | exact frame | exact frame skill | errors | p50 ms |
|---|---|---|---|---|---|---|---|---|
| no extraction (baseline) | 0 | n/a | 0.0 % | 0.0 % | 33.5 % | 0.000 | | |
| winnow-12b | 714 | 52.2 % | 39.6 % | 45.1 % | 45.1 % | 0.174 | 0 | 389 |
| gliner2.5-multi | 604 | 28.8 % | 18.5 % | 22.5 % | 27.5 % | -0.090 | 0 | 19 |

### English (1,000 utterances, 937 gold slots)

Source: [score.md](score.md), section `en`.

| arm | predicted | P | R | F1 | exact frame | exact frame skill | errors | p50 ms |
|---|---|---|---|---|---|---|---|---|
| no extraction (baseline) | 0 | n/a | 0.0 % | 0.0 % | 33.5 % | 0.000 | | |
| winnow-12b | 857 | 51.2 % | 46.9 % | 48.9 % | 44.9 % | 0.171 | 0 | 401 |
| gliner2-large | 1491 | 31.0 % | 49.3 % | 38.1 % | 22.4 % | -0.167 | 0 | 30 |
| gliner2.5-multi | 664 | 28.8 % | 20.4 % | 23.9 % | 25.6 % | -0.119 | 0 | 19 |

No significance test: F1 over slots has no paired per-item test in the
scorer, and with one run per arm the gaps above are point values.

### Threshold sensitivity (GLiNER only)

Source: [score.md](score.md), section `GLiNER threshold sensitivity`.
Rescored from the stored candidates of the same run; read on the test data
itself, so it shows how much the GLiNER result depends on the threshold and
is not tuning.

| lang | arm | F1 at 0.3 | 0.4 | 0.5 (primary) | 0.6 |
|---|---|---|---|---|---|
| de | gliner2.5-multi | 23.3 % | 23.1 % | 22.5 % | 21.4 % |
| en | gliner2-large | 35.7 % | 37.3 % | 38.1 % | 39.3 % |
| en | gliner2.5-multi | 23.2 % | 23.7 % | 23.9 % | 23.8 % |

In this range no GLiNER arm comes within 9 points F1 of Winnow-12B (best:
GLiNER2-large 39.3 % at 0.6 against 48.9 %). GLiNER2-large is still rising
at 0.6; thresholds of 0.7 and above were not scored.

### F1 per slot type, types with 20 or more gold slots

Source: [score.md](score.md), sections `F1 per slot type, de` and `F1 per
slot type, en`; every type is listed there.

| slot type | gold de | gliner2.5-multi de | winnow-12b de | gold en | gliner2-large en | gliner2.5-multi en | winnow-12b en |
|---|---|---|---|---|---|---|---|
| date | 137 | 30.5 % | 42.4 % | 137 | 73.5 % | 48.3 % | 46.7 % |
| event_name | 87 | 52.2 % | 31.9 % | 84 | 53.1 % | 43.5 % | 47.2 % |
| place_name | 86 | 0.0 % | 73.8 % | 86 | 42.0 % | 0.0 % | 80.7 % |
| person | 77 | 2.4 % | 79.5 % | 77 | 45.1 % | 4.8 % | 72.4 % |
| time | 64 | 35.0 % | 61.5 % | 64 | 38.5 % | 11.1 % | 39.1 % |
| media_type | 42 | 0.0 % | 0.0 % | 42 | 10.0 % | 0.0 % | 0.0 % |
| business_name | 27 | 37.8 % | 54.2 % | 27 | 43.8 % | 33.7 % | 59.3 % |
| weather_descriptor | 25 | 13.6 % | 28.6 % | 27 | 28.2 % | 11.8 % | 26.4 % |
| food_type | 25 | 10.8 % | 42.6 % | 25 | 54.2 % | 0.0 % | 64.3 % |
| house_place | 23 | 37.5 % | 26.7 % | 23 | 35.1 % | 31.3 % | 34.3 % |
| timeofday | 22 | 0.0 % | 40.0 % | 22 | 44.4 % | 0.0 % | 41.4 % |
| artist_name | 21 | 41.9 % | 74.4 % | 21 | 43.8 % | 45.9 % | 69.8 % |
| list_name | 20 | 0.0 % | 6.7 % | 20 | 4.8 % | 0.0 % | 12.5 % |
| relation | 20 | 0.0 % | 16.7 % | 20 | 27.6 % | 0.0 % | 51.9 % |
| transport_type | 20 | 0.0 % | 33.3 % | 20 | 35.3 % | 0.0 % | 66.7 % |

## Speed

Median latency per utterance (one call per utterance). Source: the `p50 ms`
column of [score.md](score.md); `compute_ms` by a script over the result
rows.

| arm | de | en | what it measures |
|---|---|---|---|
| winnow-12b | 389 ms | 401 ms | HTTP call from the Mac to Ollama on the workstation, LAN included, generating JSON; Ollama's own compute time (`compute_ms`) 382 and 392 ms |
| gliner2.5-multi | 19 ms | 19 ms | in-process on the GPU (RTX 4090) |
| gliner2-large | n/a | 30 ms | same |

The local LLM generates the slots as text, so it is slower here than in
route1 (one letter per call, 87 to 120 ms). One utterance at a time, no
batching. Jev has no cost row because it has no arm.

## Reading it

- **Where the local LLM is strong:** Winnow-12B is first in both languages,
  by 22.6 points F1 in German and 10.8 in English over the best GLiNER arm,
  and it is the only arm that beats predicting nothing on exact frame. It
  is strongest on names and places: `place_name` 73.8 % and 80.7 %,
  `person` 79.5 % and 72.4 %. It stays quiet on most utterances without a
  slot: it predicts nothing on 269 of 335 (de) and 230 of 335 (en).
- **Where GLiNER is strong:** speed (19 to 30 ms per utterance), and some
  types: GLiNER2-large on English `date` (73.5 % against 46.7 %),
  GLiNER2.5-multi on German `event_name` (52.2 % against 31.9 %). Below
  the 20-gold line, and so noisy (see Caveats), `currency_name` points the
  same way (multi 60.0 % and 57.8 %, Winnow-12B 36.4 % and 38.1 %, 17 gold
  each). GLiNER2-large finds the most gold slots
  in English (recall 49.3 %) but predicts 1,491 slots for 937 gold ones, and
  predicts nothing on only 92 of the 335 utterances without a slot.
- **GLiNER2.5-multi never says place and rarely says person.** It has F1
  0.0 % on `place_name` in both languages (86 gold each) and 2.4 % and
  4.8 % on `person` (77 gold each). On the stored rows it returns `place
  name` for no utterance, not even among its candidates down to confidence
  0.1. It predicts `person` 5 times (de) and 6 times (en), 1 and 2 of them
  right. Of the 77 gold person names it finds 23 (de) and 21 (en) under any
  label, mostly as `artist name` (17 and 12); the rest it misses. The
  English GLiNER2-large finds both types (42.0 % and 45.1 %). This is the
  smoke-run finding, now at full scale. Whether a different label text
  would fix it was not tested: the label texts are mechanical by design.
- **Jev:** not applicable. A Jev arm would need a construction of ours
  (choice questions over candidate spans), which would test the
  construction, not Jev.

## Caveats

- **Text matching, not the official MASSIVE metric.** A slot counts when
  type and normalised text match; position is ignored. The official MASSIVE
  slot F1 scores word-level BIO tags with seqeval. Numbers here are not
  comparable with published MASSIVE slot results.
- **The setup may favour the LLM.** A span with slightly different
  boundaries from the gold text counts as a miss, with no partial credit.
  An extractor that returns spans may lose more to this than an LLM that
  writes its slots as text and can copy the annotation style. The 53 label
  texts are mechanical (`place_name` becomes `place name`), so no arm got
  wording tuned to it; a GLiNER arm might gain more from tuned label texts
  than the LLM. Neither effect was measured.
- **English-only GLiNER2-large** runs on English only; German has one GLiNER
  arm.
- **Threshold.** 0.5 is the library default and was fixed before the run;
  the sensitivity table is read on the test data and chooses nothing.
- **MASSIVE is public** (since 2022). No arm was checked for exposure to
  these utterances.
- **Sample.** 1,000 utterances per language, the same ids in de and en,
  stratified by scenario, seed 20261009. Rare slot types have few gold
  slots (one to five for many), so per-type F1 below 20 gold slots is noisy.
- **Single run.** Run 1 of each arm only.
- **Latency is not one measurement** across arms (see Speed).

The counts of silent predictions on utterances without a slot, and the
`place name` and `person` check, come from a script over
`results/ex1/<arm>/2026-10-09-run1*.jsonl` and `reference/ex1/gold.<lang>.jsonl`.
