# route1: how the arms are measured

route1 asks every arm to put a short text into one of a fixed set of classes,
the job of a router in front of a service. Unlike pv1 the reference is not a
judge: both sources ship gold labels. Written 2026-10-09, before the full
runs. Only smoke runs of 20 requests per source and language exist so far;
they test the pipeline and are not results (files `*-smoke*`, not committed).

## Data

Frozen in `cases/route1/` (sources, hashes, licences: its README).

| source | lang | rows | questions per row |
|---|---|---|---|
| MASSIVE 1.1 test | de | 2,974 | `scenario`, `intent` |
| MASSIVE 1.1 test | en | 2,974 (same ids) | `scenario`, `intent` |
| fast-decisions dev | en | 1,700 | one per single-label task, 1 to 4 (2,600 in total) |

**Public test data.** MASSIVE test has been public since 2022, and the
fast-decisions dev rows are public on Hugging Face. No arm (Jev, Gemma 4 and
the Winnow LoRA, the GLiNER models) was checked for exposure to these rows,
and Fastino's Decide training suite may contain MASSIVE as well.

`bench/build-route-ex.mjs` turns them into 7,648 requests
(`requests/route1/requests.jsonl`, the pv1 row format with `unit: "utt"`,
`line: "<source>-<id>-<lang>"`) and 14,496 gold rows
(`reference/route1/gold.<lang>.jsonl`, the pv1 reference row format with
`arm: "gold"`).

## The questions

Every arm gets the same question ids, instructions, label names and label
texts. No label has a hand-written description: the label text is the label
name made readable (`alarm_set` becomes `alarm: set`), and fast-decisions
labels are used as given.

| question | state | instruction | options |
|---|---|---|---|
| `scenario` | `utterance` | Which scenario does `` `utterance` `` belong to? | the 18 MASSIVE scenarios |
| `intent` | `utterance` | Which intent does `` `utterance` `` express? | the intents of the **gold** scenario only |
| `<domain>:<task>` | `input` | Which label fits `` `input` `` for the task \<task\>? | the task's labels, 2 to 28 |

**The intent question is an oracle setup.** It offers only the intents of
the gold scenario (1 to 9 options), so it measures intent given the right
scenario, not end-to-end routing, and it tells every arm the scenario
indirectly. That hint stays inside the intent question: every arm, Jev
included, asks the scenario question in a call of its own, so the scenario
answer never sees the intent options (see "How each arm is called"). Three scenarios (cooking, news, weather) have a single intent:
704 of the 5,948 intent questions have one option and are right by
construction for every arm. The scorer therefore also reports `massive
intent, 2+ options`.

fast-decisions keeps only single-label tasks; the 300 multi-label tasks are
dropped. Fastino built this dataset, and its card calls it "the
classification suite behind GLiNER2.5-Decide", so these dev rows may be
close to the Decide training data (the card does not say whether they were
used in training).
It is GLiNER's home ground on purpose, and a Decide lead there is not a
neutral result.

## Arms

| arm | what | weights | runtime |
|---|---|---|---|
| `jev` | TypeSafe's hosted decision model | `jev-latest` (smoke: `jev-1.13.0`) | api.typesafe.ai |
| `winnow-12b` | Gemma 4 12B with a LoRA for Jev-style typed decisions (see pv1 method) | `hf.co/EldanRing/Winnow-12B:Q8_0`, Ollama digest `b5bedb246945...` (full digest per row, `arm_digest`) | Ollama 0.35.0, local workstation (RTX 4090) |
| `gliner2.5-multi-decide`, `-intext` | GLiNER2.5 Decide, multilingual | `fastino/GLiNER2.5-multi-Decide` | gliner2 2.0.0, torch 2.11.0+cu128, same workstation |
| `gliner2.5-decide`, `-intext` | GLiNER2.5 Decide, English | `fastino/GLiNER2.5-Decide` | same |
| `gliner2.5-decide-1b`, `-intext` | GLiNER2.5 Decide 1B, English | `fastino/GLiNER2.5-Decide-1B` | same |

**Why Winnow-12B as the one local model.** It is the best pv1 local arm by
the mean of the five line questions (is_req, must, axis, evidence, level)
against the adjudicated German reference (`results/pv1/score-de-adjudicated.md`):
Winnow-12B 86.2 %, Qwen3.8-27B 85.2 %, Shisa DE-1 78.4 %, Qwen3-32B 77.4 %.
The lead over Qwen3.8 is one point and not tested for significance.

Weights: Jev is called as `jev-latest`, and each row stores the version
that answered (`arm_version`). The local rows store the Ollama digest
(`arm_digest`), the GLiNER rows the commit sha of the Hugging Face snapshot
they were loaded from (`arm_revision`, runner change of 2026-10-09).

The English Decide models also run on the German MASSIVE rows. Those rows
are a labelled extra (`English model, extra` in the score report), not a
fair German result.

## How each arm is called

**Jev** (`bench/run-jev.mjs`): one POST per question, four requests in
parallel (the calls of a request run one after the other), no system
prompt. The request file holds both MASSIVE questions, or all tasks of a
fast-decisions row, in one request, but outside pv1 the runner sends each
question alone, as the local arm and GLiNER do. A joint call would let the
scenario question see the intent options, which name the gold scenario, and
would let correlated fast-decisions tasks (`email_triage` asks category,
action, needs_reply and is_phishing of the same message) be answered
together while the other arms answer them one by one. Rows carry
`call_scope: "question"` and `attempts` (HTTP attempts of the call, more
than 1 after a 429 or 5xx). Probe of 2026-10-09: Jev accepted a choice with
1, 28 and 60 options and question ids with a colon.

**Local** (`bench/run-local.mjs`): as in pv1, one call per question, options
as letters, the probability of an option from the logprob of its letter,
`--template gemma4 --prefix ''` (empty thought channel, so thinking off),
temperature 0. Only the system line is new: "You classify a short text.
Answer each multiple-choice question with the letter of exactly one
option." When no option letter is among the top 20 logprobs (letter mass
0), the row is an error with no answer, not option A. The 26 letters are a
hard limit: `support_intent:intent` has 28
labels, so its 100 questions cannot be asked. The runner refuses a request
set with more than 26 options unless `--skip-over-26` is given, and then
writes those questions as rows with an error and no answer. The scorer
counts them as wrong and shows them in the `answered` column. The group
`fastdec, tasks with 26 options or fewer` leaves this task out and is the
like-for-like fast-decisions comparison of the local arm with the others.

**GLiNER** (`bench/run-gliner.py`): one single-label classification task per
question, as in pv1. The passage is the state as a named section
(`` `utterance` ``: text). Mode `prompt` puts the instruction into the task
prompt, mode `intext` appends it to the passage as `Question: ...`. The
label is the option text (`alarm: set`), the label description is the same
text, and the answer is mapped back to the option key. No label contains a
parenthesis, so no row carries an `adapt` field. GLiNER accepted the colon
in the labels in a probe of 2026-10-09. `--focus` is not run: the state has a single
section, so it would change nothing. Probabilities are the classifier's
softmax over the task's labels.

## Metrics (`bench/score-route.mjs`)

Per arm, source, language and question group, on run 1:

- **acc**: share of answers equal to gold; an unanswered row counts as wrong.
- **macro-F1**: mean F1 over the labels seen in gold or prediction.
- **base**, "most frequent answer": for every option set (the scenario
  question, the intent question per gold scenario, every fast-decisions
  task) the most frequent gold label among the same rows.
- **skill** = (acc - base) / (1 - base). n/a when base is 100 %.
- **McNemar** against Jev, exact and two-sided (`bench/lib/stats.mjs`, the
  test of `bench/significance.mjs`), on the items both arms have a row for.
  An unanswered row counts as wrong, as in acc, so acc and the test cover
  the same items.
- **Groups**: `massive scenario`, `massive intent`, `massive intent, 2+
  options`, `fastdec, all single-label tasks` and `fastdec, tasks with 26
  options or fewer` (without `support_intent:intent`), each per language,
  plus accuracy per fast-decisions task.
- **p50 ms**: median latency per call; every arm makes one call per
  question. The numbers are not the same measurement across arms. Jev's is
  the HTTP round trip to api.typesafe.ai. The local arm's is the HTTP call
  from the Mac to Ollama on the workstation, LAN included; Ollama's own
  `total_duration` is stored per row as `compute_ms` (smoke: median 93 ms
  per call against 88 ms compute). GLiNER's is measured in-process on the
  GPU, with `cuda.synchronize` around the call.
- **Jev cost** the pv1 way: median input tokens per request, summed over
  its calls, x USD 0.042 per million input tokens (docs.typesafe.ai/models,
  read 2026-10-02), output free. Smoke with one call per question: about
  880 input tokens per MASSIVE utterance (two calls) and 490 per
  fast-decisions row of one task, about USD 0.037 and 0.021 per 1,000.

## Full run

Mac side (Jev, then the local model over Ollama), one command:

```bash
TYPESAFE_API_KEY=... OLLAMA_HOST=http://<workstation>:11434 HOST_LABEL=local-rtx4090:11434 bash bench/run-route-ex.sh
# without HOST_LABEL the local rows say host "local"
```

Workstation side (GLiNER, from a copy of the repo with `bench/`,
`requests/route1/` and `cases/ex1/`), one command:

```bat
set PYTHON=<venv>\Scripts\python.exe
set HOST_LABEL=local-rtx4090
bench\run-route-ex.bat
```

Then copy `results/route1/` and `results/ex1/` of the workstation back and
score:

```bash
bash bench/run-route-ex.sh score
```

which runs `node bench/score-route.mjs --md results/route1/score.md` and
`node bench/score-extract.mjs --md results/ex1/score.md`.

Commands inside the scripts, route1 part:

```bash
node bench/run-jev.mjs --pv route1 --runs 1
node bench/run-local.mjs --pv route1 --model hf.co/EldanRing/Winnow-12B:Q8_0 --arm winnow-12b --template gemma4 --prefix '' --host-label local-rtx4090:11434 --runs 1 --skip-over-26
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-multi-Decide --arm gliner2.5-multi-decide --mode prompt --host local-rtx4090
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-multi-Decide --arm gliner2.5-multi-decide-intext --mode intext --host local-rtx4090
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide --arm gliner2.5-decide --mode prompt --host local-rtx4090
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide --arm gliner2.5-decide-intext --mode intext --host local-rtx4090
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide-1B --arm gliner2.5-decide-1b --mode prompt --host local-rtx4090
python bench/run-gliner.py --pv route1 --model fastino/GLiNER2.5-Decide-1B --arm gliner2.5-decide-1b-intext --mode intext --host local-rtx4090
```

Estimated durations from the smoke latencies (2026-10-09): Jev about 15 to
20 minutes for 14,496 calls at four in parallel (smoke median about 240 ms
per call), if no rate limit appears;
Winnow-12B about 25 to 35 minutes for 14,496 calls (smoke median 0.09 s
per call, fast-decisions inputs are longer) plus up to 2 minutes model load; GLiNER 5 to 10 minutes per arm, about 45 minutes for
the six arms. The local model and GLiNER share one GPU, so the two machine
scripts should not run at the same time, or the latencies are not clean.

Not in the full run: a shuffled run (`--runs 5 --run-ids 5` for Jev and the
local arm) to test option order, and repeated runs for flip rates. Both are
cheap to add later.

## Smoke runs

Every arm went through request, result file and scorer report on
2026-10-09 with `--per-group 20 --tag smoke` (20 requests per source and
language; for fast-decisions that is 20 rows of `agent_handoff`). Jev and
the local arm were run again after the review fixes of the same day (Jev
one call per question, the local arm with `compute_ms` and the letter mass
rule); the GLiNER smoke rows are unchanged, its runner only gained
`arm_revision`. Smoke
files match `*-smoke*` and are not committed. Reproduce with
`--per-group 20 --tag smoke` on any runner (plus `--runs 1` for the Node
runners) and `node bench/score-route.mjs --tag smoke`.
