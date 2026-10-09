# ex1: how the arms are measured

ex1 asks every arm to find the slots of a short spoken command: dates,
times, places, names and the other MASSIVE slot types, each with the words
of the utterance that fill it. The reference is the MASSIVE annotation.
Written 2026-10-09, before the full runs. Only smoke runs of 20 utterances
per language exist so far; they test the pipeline and are not results
(files `*-smoke*`, not committed).

## Data

Frozen in `cases/ex1/` (sample, seed and quota: its README; source, hash and
licence: `cases/route1/README.md`).

- 1,000 MASSIVE 1.1 test utterances per language, the same ids in de and
  en, seed 20261009, stratified by scenario.
- Gold slots parsed from `annot_utt` with `\[([^\]:]+?) : ([^\]]+?)\]`:
  941 in de, 937 in en. 335 utterances per language have no slot.
- 53 slot types, every type that occurs in the test split of either
  language (`cases/ex1/slot-types.json`). Every arm gets the same label text
  per type, without descriptions. The label text is made by one mechanical
  rule: every underscore of the MASSIVE name becomes a space (`place_name`
  becomes `place name`, `timeofday` stays `timeofday`). 47 of the 53 names
  change, nothing is written by hand. `bench/build-route-ex.mjs` writes the
  mapping to `cases/ex1/slot-labels.json`, and both arms read it from there.
  Each arm maps the returned label back to the MASSIVE type before the row
  is stored, so scoring compares MASSIVE types with the gold. Names glued
  together in MASSIVE (`timeofday`) stay glued; splitting them would need a
  word list, which is a hand-written step.
- MASSIVE test has been public since 2022. No arm was checked for exposure
  to these utterances.
- Gold per utterance: `reference/ex1/gold.<lang>.jsonl`.

## Arms

| arm | what | weights | lang |
|---|---|---|---|
| `winnow-12b` | local LLM over Ollama, generation mode | `hf.co/EldanRing/Winnow-12B:Q8_0`, Ollama digest per row (`arm_digest`) | de, en |
| `gliner2.5-multi` | GLiNER2.5 entity extractor, multilingual | `fastino/gliner2.5-multi-v1` | de, en |
| `gliner2-large` | GLiNER2 entity extractor, English | `fastino/gliner2-large-v1` | en |

The GLiNER rows store the commit sha of the Hugging Face snapshot they were
loaded from (`arm_revision`).

Winnow-12B is the one local model of route1 and ex1, chosen as the best pv1
local arm (see `results/route1/method.md`).

**Jev: not applicable.** TypeSafe's API offers three primitives, choice,
score and noul, and returns typed values and probabilities, no text spans
(docs.typesafe.ai/introduction, read 2026-10-09). There is no Jev arm in ex1.
Building one out of choice questions over candidate n-grams would test a
construction of ours, not Jev.

## How each arm is called

**Local** (`bench/run-local-extract.mjs`): one call per utterance, raw
prompt in the Gemma 4 template with an empty thought channel (thinking off,
as in pv1), temperature 0, `num_predict` 512, Ollama `format: "json"`. One
untimed warm-up call first. System line: "You extract slots from a short
text. Answer with JSON only." User turn:

```
Slot types: <the 53 label texts, comma separated>

Text: <utterance>

List every slot in the text as JSON: {"slots": [{"type": "<one of the slot types>", "text": "<the words, copied verbatim from the text>"}]}. Use only the slot types listed. If the text has no slot, answer {"slots": []}.
```

The list sits inside an object because Ollama's JSON mode expects an object
at the top level. Parsing is lenient: a bare list, a list under another key,
and items written as `slot`/`value` are all read. Output that is not JSON,
or items without type and text, give an empty prediction plus an `error`
field, and the row stays in the score. The answered label is mapped back to
its MASSIVE type (`type`, the answer itself stays in `label`); an answer
that already is a MASSIVE name is kept as that type, any other text is kept
as answered and counts as wrong. The raw output is stored per row (`raw`).

**GLiNER** (`bench/run-gliner-extract.py`): `gliner2.AutoExtractor`, one
`extract_entities(utterance, label_texts, threshold=0.1,
include_confidence=True, include_spans=True)` call per utterance, overlap
policy at the library default, one warm-up call first. Every returned entity
is stored with its confidence, its label and its MASSIVE type in
`candidates`. The primary prediction `slots` is the candidates with a
confidence of at least 0.5, the library default, fixed before any run. Both
thresholds are stored per row (`threshold`, `call_threshold`). The low call
threshold only exists so the scorer can show a sensitivity table from the
same run (see Metrics). `--check-threshold` makes a second call at 0.5 per
utterance and compares it with the filtered candidates: in the smoke runs
they were identical in 40 of 40 utterances (gliner2.5-multi) and 20 of 20
(gliner2-large), so filtering the 0.1 call gives the same slots as calling
at 0.5. The scored value is the returned text. Spans are stored for
inspection: in the smoke runs `text == utterance[start:end]` held for all 77
returned candidate spans (character offsets, end exclusive), checked with
`--check-spans`.

## Metrics (`bench/score-extract.mjs`)

A slot is the pair (slot type, normalised text). Normalised means
lowercase, trimmed, inner spaces collapsed, trailing punctuation removed.
Gold and prediction of an utterance are multisets of such pairs.

- **P, R, F1**: micro over all utterances of an arm and language.
- **F1 per slot type**, with the gold count per type.
- **exact frame**: share of utterances whose predicted multiset equals the
  gold multiset. An utterance without gold slots counts when the arm
  predicts nothing.
- **errors**: rows with a call or parse error.
- **p50 ms**: median latency per utterance. For the local arm this is the
  HTTP call to Ollama on the workstation, LAN included; Ollama's own
  `total_duration` is stored per row as `compute_ms` (smoke: median 234 ms
  against about 230 ms compute). GLiNER is timed in-process on the GPU.
- **GLiNER threshold sensitivity**: the GLiNER rows are rescored from
  their stored candidates at 0.3, 0.4, 0.5 and 0.6. 0.5 is the primary
  result and equals the main table. The other three are read on the test
  data itself: they show how much the GLiNER numbers depend on the
  threshold, they are not tuning, and no threshold is chosen from them.
  There is no separate tuning split in ex1, so any threshold other than the
  default would be fitted to the test data.
- **Baseline "no extraction"**: predicts nothing, so recall and F1 are 0
  and precision is undefined. Its exact-frame rate is the share of
  utterances without slots, 33.5 % in both languages, which is the floor
  any arm has to beat on that measure.

This is text matching, not the official MASSIVE slot F1. The official
evaluation scores word-level BIO tags with seqeval, so it counts position
as well; here a value found at the wrong place, or twice, still matches as
text.

## Full run

The full run is part of `bench/run-route-ex.sh` (Mac side) and
`bench/run-route-ex.bat` (workstation side), see `results/route1/method.md`.
ex1 commands:

```bash
node bench/run-local-extract.mjs --model hf.co/EldanRing/Winnow-12B:Q8_0 --arm winnow-12b --template gemma4 --host-label local-rtx4090:11434
python bench/run-gliner-extract.py --model fastino/gliner2.5-multi-v1 --arm gliner2.5-multi --host local-rtx4090
python bench/run-gliner-extract.py --model fastino/gliner2-large-v1 --arm gliner2-large --lang en --host local-rtx4090
node bench/score-extract.mjs --md results/ex1/score.md
```

Estimated durations from the smoke latencies: Winnow-12B about 8 to 10 minutes
for 2,000 utterances (smoke median 0.22 s each), GLiNER about 1 minute per arm
plus model load.

## Smoke runs

Every arm went through utterances, result file and scorer report on
2026-10-09 with `--limit 20 --tag smoke` (the first 20 utterances of each
language). Reproduce with those flags and
`node bench/score-extract.mjs --tag smoke`.
