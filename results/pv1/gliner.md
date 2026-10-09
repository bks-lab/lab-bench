# pv1: GLiNER2.5 Decide as a further arm (2026-10-08)

Fastino published the GLiNER2.5 Decide classifiers in September 2026 and lists
"JevK5" in its own benchmark table (fast-decisions, English: Decide 60.2 %,
JevK5 57.6 %, multi-Decide 56.7 %). The question here: do they answer the
match questions of pv1 as well as Jev or the local LLMs?

Short answer: no. The best of nine configurations reaches a mean of 53.6 %
(de) and 52.3 % (en) over the five line questions. Giving the most frequent
reference answer to every line reaches 54.3 % and 53.4 %. Jev reaches 86.9 %
and 84.6 %.

## The models

| arm prefix | model | encoder | parameters | language | licence |
|---|---|---|---|---|---|
| `gliner2.5-multi-decide` | fastino/GLiNER2.5-multi-Decide | mDeBERTa-v3-base | 287M | multilingual | Apache 2.0 |
| `gliner2.5-decide` | fastino/GLiNER2.5-Decide | DeBERTa-v3-large | 340M | English | Apache 2.0 |
| `gliner2.5-decide-1b` | fastino/GLiNER2.5-Decide-1B | Ettin encoder-from-decoder 1B | about 1B | English | Apache 2.0 |

Source: the model cards on huggingface.co/fastino, read 2026-10-08. None of
them generates text. A call gets a passage and a label set, optionally with a
description per label and a task prompt, and returns a score per label.

Run with `bench/run-gliner.py` on a local workstation (RTX 4090), gliner2 2.0.0,
torch 2.11.0+cu128, transformers 5.19.0, one question per call, run 1 only.
The classifier generates nothing and has no sampling, so a repeat run is not
expected to differ and was not made.

## How the questions were asked

The passage is the state exactly as `run-local.mjs` writes it. Each question
is one single-label task, labels are the option keys, descriptions are Jev's
option texts, probabilities are the classifier's softmax over the labels.

Two adaptations are forced by the model and recorded per row (`adapt`):

- GLiNER refuses "(" and ")" in prompts, labels and descriptions (they are
  prompt markers). They became "[" and "]".
- The evidence options are references like `` `cv.s-x` ``. An encoder cannot
  look a reference up, so each evidence label got the text of its CV entry as
  description.

Three ways to hand over the question, each with all three models:

| mode | question | passage |
|---|---|---|
| `prompt` | Jev's instruction as the task prompt, the form the model card shows for "question over a passage" | full state |
| `intext` | appended to the passage as `Question: ...`, the way the local LLMs get it; no task prompt | full state |
| `focus` (`prompt` + `--focus`) | as `prompt` | only the state sections the question names (`requirement`, `cv`, `project`), so a requirement question does not carry the whole CV |

`intext` and `focus` are our additions. They exist because a probe before the
run showed that the prompt is barely read: on the model card's own example
("The treaty was signed in Paris in 1992. It entered into force the following
year"), multi-Decide gave 0.993 for yes to both "Did the treaty enter into
force in 1992?" and "...in 1993?". Decide and Decide-1B behaved the same.
The model card shows "no" for 1992 as a potential output.

## Results

Against `adjudicated-a` (de) and `claude-c` (en), run 1, same lines and
definitions as `score-<lang>.md`. Mean = mean of the five line questions.

| arm | lang | is_req | must | axis | evidence | level | mean |
|---|---|---|---|---|---|---|---|
| jev | de | 88.2 % | 93.1 % | 93.1 % | 81.0 % | 79.3 % | 86.9 % |
| most frequent answer | de | 85.3 % | 94.0 % | 16.4 % | 41.4 % | 34.5 % | 54.3 % |
| gliner2.5-decide-1b-focus | de | 72.1 % | 94.0 % | 51.7 % | 13.8 % | 36.2 % | 53.6 % |
| gliner2.5-decide | de | 84.6 % | 92.2 % | 27.6 % | 15.5 % | 38.8 % | 51.7 % |
| gliner2.5-multi-decide-focus | de | 83.8 % | 94.0 % | 32.8 % | 19.0 % | 22.4 % | 50.4 % |
| gliner2.5-multi-decide | de | 85.3 % | 94.0 % | 15.5 % | 19.0 % | 22.4 % | 47.2 % |
| jev | en | 88.5 % | 89.1 % | 91.6 % | 79.0 % | 74.8 % | 84.6 % |
| most frequent answer | en | 85.6 % | 94.1 % | 16.0 % | 37.0 % | 34.5 % | 53.4 % |
| gliner2.5-decide-1b-focus | en | 67.6 % | 94.1 % | 62.2 % | 1.7 % | 36.1 % | 52.3 % |
| gliner2.5-decide | en | 85.6 % | 90.8 % | 31.1 % | 14.3 % | 38.7 % | 52.1 % |

All nine arms are in `score-de.md`, `score-de-adjudicated.md`, `score-en.md`,
the paired tests in `significance-<lang>.md`. "Most frequent answer" is the
reference's own most frequent answer per question (evidence: always none),
computed from `reference/pv1/`, not an arm with a results file.

What the numbers mean:

- **is_req and must** look close to Jev only because the reference says yes
  on 85 % and 94 % of the lines. multi-Decide answers yes on all 275 lines
  for both questions in `prompt` mode and no on all 275 for is_req in
  `intext` mode.
- **axis** is the one question GLiNER is built for: sort one short text into
  topics. With `focus` the English models reach 52 to 62 %, still far below
  Jev (93 %, 92 %) and every local LLM (71 to 88 %). The difference to Jev is
  significant (de: Jev alone right on 50 lines, Decide-1B-focus alone on 2,
  p < 0.001).
- **evidence**: no arm in any mode ever answers none. Every line without
  evidence gets an invented entry (invented evidence 100 %). Over all
  requirements the evidence answer is right on 2 to 19 %, always answering
  none would be right on 41 % (de) and 37 % (en).
- **level** stays at or near the most-frequent-answer line (22 to 39 %).
- **Total score**: the pairs built to not fit (p03, p06, p09) get 31 to 91
  points from the GLiNER arms, 12 to 34 from Jev (`report.md`). The score does
  not separate fit from no fit.

## Speed

Median time per German line (sum of the five question latencies, one question
per call, GPU synchronised before and after): multi-Decide 0.13 to 0.16 s,
Decide 0.28 to 0.36 s, Decide-1B 0.26 to 0.41 s. Jev answers a whole line in
one request in 0.25 s (`speed-cost.md`), the local LLMs need 1.6 to 2.5 s.
Fast, but fast at the wrong answer.

## Reading it

GLiNER2.5 Decide is what its card says it is: a router for one text and a
label set (intent, topic, sentiment). The match asks something else: does a
second text show what the first asks for. That needs the two texts weighed
against each other, and none of the three models does it in any of the three
ways we tried. A fine-tuned GLiNER (Fastino offers training) is a different
question and was not tested.

Fastino's table puts Decide ahead of JevK5 on its own English routing set.
That is a different task from ours and a version of Jev we do not know
(`jev-1.13.0` here). Both can be true.
