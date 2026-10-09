# pv1: how the arms were measured

How Jev and the four local models were asked the same questions in bench
version pv1, what came back, and every way the two sides were not treated
alike. All numbers are recomputed from the run 1 files of 2026-10-01 in this
directory unless a line names another source. Written 2026-10-02.

## What the arms are

None of the local arms is Jev, and none is a copy of it. Two kinds of model
took part.

| arm | what it is | weights we ran | runtime |
|---|---|---|---|
| `jev` | TypeSafe's hosted decision model (System One family), called over its API. TypeSafe does not publish how Jev works inside, so we only see what the API returns | `jev-latest`, resolved to `jev-1.13.0` (field `arm_version`) | api.typesafe.ai, USA |
| `qwen3-32b` | Qwen3 32B, a general chat model. No training for typed questions is described on its model card | Ollama tag `qwen3:32b` | Ollama 0.30.6 on a local workstation (RTX 4090) |
| `qwen3.8-27b` | Qwen3.8 27B, a general chat model. No training for typed questions is described on its model card | Ollama tag `qwen3.8:27b` | Ollama 0.35.0, same machine |
| `winnow-12b` | Gemma 4 12B IT with a merged LoRA, trained by EldanRing for "Jev-style" typed decisions (gold-label plus teacher-distribution cross-entropy). Ships its own llama.cpp server `winnow-inference` with a `/v1/systemone` endpoint. The card says it is not affiliated with TypeSafe | `hf.co/EldanRing/Winnow-12B:Q8_0` | Ollama 0.30.6, same machine |
| `shisa-de-1` | Gemma 4 26B-A4B IT, trained by shisa-ai as a "decision model" with softmax cross-entropy over the option letters at the answer boundary, so its answer is read from the next-token distribution | `hf.co/mradermacher/shisa-de-1-GGUF:Q4_K_M` | Ollama 0.30.6, same machine |

Sources: model cards at huggingface.co/EldanRing/Winnow-12B and
huggingface.co/shisa-ai/shisa-de-1, read 2026-10-02. Ollama versions are not
in the run 1 rows (`runtime` is null, the field was added in eb57238 after
these runs). They come from the work log: the machine ran 0.30.6 until the
update to 0.35.0 at 19:37 UTC on 2026-10-01. The row timestamps fit that
order (qwen3-32b 18:38 to 18:46, shisa 19:05 to 19:17, winnow 19:17 to
19:30, qwen3.8 19:47 to 19:55 UTC). The quantisation behind the two Qwen
tags was not recorded, and the machine was not reachable on 2026-10-02 to
read it back.

Winnow and Shisa were not run in their native format. Both went through our
own Ollama harness and our prompt (below), not through `winnow-inference` or
Shisa's own template. A result for them is a result for "this model in our
harness".

## The questions

`bench/build-requests.mjs` turns the cases into 299 requests
(`requests/pv1/requests.jsonl`), exactly as the service would send them to
Jev (`bench/match.ts`, vendored from the service):

- 8 `project` requests (4 postings, en and de), 9 questions each: `demand:<area>` (score 0 to 3) and `role` (choice)
- 16 `profile` requests (8 CVs, en and de), 8 questions each: `depth:<area>` (score 0 to 3)
- 275 `line` requests (one per requirement line, pair and language), 5 questions each: `is_req` and `must` (noul, yes or no), `axis` (choice of 9 areas), `evidence` (choice of the CV entries plus `none`), `level` (score 0 to 3)

That is 8 × 9 + 16 × 8 + 275 × 5 = 1575 questions per run, sent to Jev as
299 requests and to a local model as 1575 calls.

## How Jev is called (`bench/run-jev.mjs`)

- One POST to `https://api.typesafe.ai/v1/systemone` per request, so all questions of a unit go out together: five for a line, nine for a project, eight for a profile. Four requests run in parallel.
- The body is the service's request unchanged: `model: "jev-latest"`, `state` as a JSON object (`requirement` as text, `cv` as an object of entry id to entry text), `questions` as an object of typed questions with Jev's own `instructions` and `criteria`.
- No system prompt, no temperature or other sampling setting from our side.
- What comes back per question type, and how `bench/normalize.mjs` stores it:
  - `noul`: one number, the probability of yes. Stored as `probs = [1 - noul, noul]` over `["no", "yes"]`, `choice = yes` when `noul >= 0.5`.
  - `choice`: a `choice` key and `probabilities` per option key. Stored as the probabilities in the order the question defined the options, `choice` as returned. In all 558 choice rows of run 1 the returned choice is also the most probable option.
  - `score`: a `score` (the expected level) and `probabilities` per level. Stored as `probs` over `"0"` to `"3"`, `choice` = the most probable level, `expected` = Jev's `score`. In 39 of 467 score rows, rounding Jev's expected level gives a different level than the most probable one. The service itself works with the expected value, the per-question comparison in this bench with the most probable level. The total score in `bench/analyze.mjs` recomputes the expected value from `probs` for every arm alike.
- Jev returns its probabilities to two decimals. In run 1 the largest yes or no probability Jev gave on any of 550 noul questions is 0.99.

## How a local model is called (`bench/run-local.mjs`)

A chat model does not return a probability per option, so the harness makes
it answer with one letter and reads the probability of that letter.

- **One call per question**, not per request. The state comes first and the question last, so Ollama can reuse the cached prompt prefix across the questions of a unit.
- **System prompt** (ours, Jev gets none from us): "You judge a project posting and a candidate CV. Answer each multiple-choice question with the letter of exactly one option."
- **State as text**: every state key becomes a named section, the CV as one line per entry written `cv.<id>: <text>`, so the backtick references in Jev's instructions (`` `requirement` ``, `` `cv` ``) and the evidence options (`` `cv.s-x` ``) point at something the model can see.
- **Instruction**: Jev's wording, unchanged.
- **Options as letters**: noul as `A) Yes`, `B) No`. Score as its levels 0 to 3 in order (`A` = level 0). Choice as its criteria in definition order, label only (for `evidence` the label is `` `cv.<id>` ``, for `axis` the area description).
- **Template and prefix**:
  - Qwen arms: ChatML (`--template chatml`), an empty `<think>` block to switch reasoning off, then the assistant turn pre-filled with `Answer: **`. With only `Answer:` as prefix, qwen3:32b writes the letter in bold, and 28 of 29 first tokens were `**` instead of a letter (probe of 2026-10-01 on qwen3:32b only, run-local header comment and work log).
  - Winnow and Shisa: Gemma 4 turns (`--template gemma4`), an empty thought channel, and no prefix (`--prefix ''`). Shisa DE-1 is trained to put the letter first. Winnow-12B's own server (winnow-inference, native/protocol.h) ends the model turn with `Answer:\n` before the letter and lists the yes or no options as false, then true. Our template leaves both out, so for Winnow the empty prefix and our letter order depart from its native format. Commands as run: `node bench/run-local.mjs --model <tag> --arm <arm> --template gemma4 --prefix '' --runs 5 --run-ids 1`.
- **Sampling**: `raw: true`, `temperature: 0`, `num_predict: 1`, `top_logprobs: 20`, `num_ctx: 16384`. The largest prompt in run 1 was 1403 tokens, so nothing was cut.
- **Probability**: among the 20 most likely first tokens, every token that is a single option letter after trimming spaces adds its probability to that letter. The letter probabilities are then divided by their sum (renormalised over the options). A letter outside the top 20 counts as 0. Values are rounded to four decimals.
- **`mass`**: the sum before renormalising. It says how much of the model's first-token probability went to any option letter at all. A low mass means the model wanted to write something else, and the renormalised probabilities then rest on a small remainder.
- **`first_token`**: the token the model would actually have written.

## Run 1 recomputed

Files: `results/pv1/<arm>/2026-10-01-run1.jsonl`. For `qwen3-32b` run 1 is
split in two: the project and line rows in `2026-10-01-run1.jsonl`, the 128
profile rows in `2026-10-01-run1-profile.jsonl` (a separate profile run at
18:48 UTC, after the check run below). Both are counted here, as
`analyze.mjs`, `score.mjs` and `significance.mjs` do. Every arm: 1575 rows,
0 errors.

| arm | median top probability | top probability >= 0.99 | noul | choice | score | median mass | mass < 0.9 | mass < 0.5 |
|---|---|---|---|---|---|---|---|---|
| jev | 0.910 | 359 (22.8 %) | 10 of 550 | 237 of 558 | 112 of 467 | n/a | n/a | n/a |
| qwen3-32b | 0.997 | 987 (62.7 %) | 467 of 550 | 306 of 558 | 214 of 467 | 1.0000 | 0 | 0 |
| qwen3.8-27b | 0.884 | 235 (14.9 %) | 1 of 550 | 171 of 558 | 63 of 467 | 0.9979 | 0 | 0 |
| shisa-de-1 | 0.956 | 378 (24.0 %) | 255 of 550 | 88 of 558 | 35 of 467 | 0.9968 | 1 | 0 |
| winnow-12b | 0.995 | 902 (57.3 %) | 320 of 550 | 379 of 558 | 203 of 467 | 0.9996 | 167 | 59 |

Lowest mass per arm: qwen3-32b 0.9931, qwen3.8-27b 0.9767, shisa-de-1
0.8819, winnow-12b 0.0039.

**The share at >= 0.99 does not compare cleanly between Jev and the local
arms.** Jev rounds to two decimals and in this run never went above 0.99 on
a yes or no question, so its 10 noul rows at >= 0.99 are all exactly 0.99.
The local arms are rounded by us to four decimals. The column says how often
an arm is close to certain in its own output format, not who is better
calibrated.

**First tokens** (run 1, all 1575 questions):

| arm | first tokens |
|---|---|
| qwen3-32b | A 740, C 256, B 254, D 105, H 66, F 50, E 39, I 32, G 16, K 15, J 2 |
| qwen3.8-27b | A 744, B 227, C 223, D 147, H 65, F 57, I 42, E 31, K 23, G 14, J 2 |
| shisa-de-1 | A 636, B 347, C 203, D 199, F 66, H 42, E 30, I 28, K 13, G 11 |
| winnow-12b | A 713, B 230, C 177, D 144, The 113, F 50, H 48, I 40, E 30, K 16, G 14 |

Every arm except Winnow put a letter first on every question. Winnow started
113 answers with "The". All 59 Winnow answers with mass < 0.5 are among
them: 31 `depth` questions (profile), 18 `evidence`, 7 `axis`, one each of
`is_req`, `must` and `level`, 37 German and 22 English. Their median mass is
0.347. Those 59 answers are a renormalised remainder and should be read with
that caveat. Winnow's own server reads answers in its native format, which we
did not use, so this says nothing about Winnow in its own harness.

A for "Yes" on every noul question and A for level 0 on every score question
mean the letter A carries two very different meanings across question types.
The counts above are not a position-bias measure.

## The prefix is not neutral (Shisa probe)

On 2026-10-01 at 18:56 UTC Shisa DE-1 answered the first three lines of p01
(15 questions) once with prefix `''` and once with `Answer: **`, everything
else equal. **6 of 15 choices changed**: `must` on line 1 (no to yes), `axis`
on line 1 (testauto to ai), `is_req` and `must` on line 2 (yes to no) and
`level` on lines 2 and 3 (2 to 3). Mass stayed high in both (min 0.9885 and
0.9865). The README said 5 of 15. That was a miscount, corrected with this
file. The probe output was written to an arm `shisa-probe` that was not kept,
so the source is the command output in the session transcript, not a file in
this repository.

The pv1 Shisa and Winnow runs use `''`. For Shisa that is its trained
format. For Winnow it is not, since its own format expects `Answer:\n` before
the letter. The Qwen runs use `Answer: **`. So the prefix differs between arms by design, and
the probe shows the choice of prefix alone can move answers.

## Run to run: what repeats locally and what does not

`results/pv1/qwen3-32b/checks/2026-10-01-run2-p01-p04.jsonl` is a second run
of qwen3-32b on pairs p01 and p04 (348 questions, 18:46 to 18:48 UTC),
compared here with run 1 question by question.

| unit | questions | same choice | probabilities differ | largest difference |
|---|---|---|---|---|
| line | 280 | 280 | 8 | 0.189 (p01-en-l07, evidence) |
| project | 36 | 36 | 7 | 0.053 |
| profile | 32 | 32 | 9 | 0.100 |

Rows are matched on unit, line, question and language. So at temperature 0
all 348 choices repeat exactly, and the probabilities move by up to 0.19 on
line questions and up to 0.10 on project and profile questions. An earlier
version of this file said 3 of 68 project and profile choices changed. That
count matched rows without the language, so it compared an English answer
with a German one. Project and profile rows carry no language in `line`. The
cause of the probability spread was not measured. Candidates are the reuse of
the cached prompt prefix and GPU arithmetic that is not bit-exact.

For comparison Jev, sent the same requests five times: run 2 against run 1
changed 24 of 1575 choices (3 of the same 348 questions), runs 3 and 4 changed
27 each, 78 of 4725 or 1.7 % over runs 2 to 4 (the flip column in report.md).
Median largest probability difference per question 0.01, largest 0.19 in each
of runs 2 to 4. The shuffled run (run 5, options of choice questions in a
seeded random order) changed 32 of 1575 Jev choices (2.0 %) and 89 of 1575
qwen3-32b choices (5.7 %, run 1 including the 128 profile rows). Shisa, Winnow and Qwen3.8 have one run each, so their
run-to-run spread is not known.

## Every difference in treatment, and what it can do

| # | Jev | local arms | possible effect |
|---|---|---|---|
| 1 | all questions of a unit in one request | one call per question | TypeSafe documents that each question is evaluated in parallel and in isolation against the same state (docs.typesafe.ai/introduction.md, read 2026-10-02). Local answers are independent as well. Not verified by us. If that holds, the difference is mainly time and cost |
| 2 | state as JSON, no system prompt from us | state rendered as text sections plus our system prompt | the rendering and the system prompt are our choices and can move local answers. Jev may add its own framing, which we cannot see |
| 3 | options passed as keys with descriptions, noul without options | options as letters with labels only, noul as A) Yes B) No | letter position can bias a chat model. Yes is always A. The model never sees the option keys, only labels |
| 4 | probabilities produced by Jev itself | next-token probability of the letter, top 20 only, renormalised | a letter outside the top 20 counts as 0. With low mass the probabilities rest on a small remainder (Winnow, 59 questions) |
| 5 | probabilities rounded to two decimals, noul at most 0.99 here | rounded to four decimals by us | certainty shares and calibration curves are not on the same scale |
| 6 | score as expected level plus distribution | distribution only | per-question comparison uses the most probable level for both. Jev's own expected level differs from it in 39 of 467 score rows |
| 7 | `jev-latest`, sampling unknown | temperature 0, one token | Jev's run-to-run spread includes whatever sampling TypeSafe uses. Local spread comes from runtime arithmetic only |
| 8 | one model version for all runs | Ollama 0.30.6 for three arms, 0.35.0 for Qwen3.8 | runtime version can change numbers. Not recorded in the run 1 rows |
| 9 | full-precision weights on TypeSafe's side (assumed, not published) | quantised GGUF: Winnow Q8_0, Shisa Q4_K_M, Qwen tags unrecorded | quantisation can shift probabilities, more at Q4 than at Q8 |
| 10 | native interface | Winnow and Shisa outside their native format, Qwen as a chat model forced into letters | results for Winnow and Shisa may differ, in either direction, from what their own servers return |
| 11 | prompt prefix n/a | `Answer: **` for Qwen, `''` for Winnow and Shisa (Winnow's own format would be `Answer:\n`) | the probe shows the prefix alone changed 6 of 15 Shisa answers |
| 12 | 5 runs, one of them shuffled | qwen3-32b run 1 plus shuffled run 5 and a p01/p04 check, the others run 1 only | stability is measured for Jev and partly for qwen3-32b, not for the other three |
| 13 | hosted in the USA, network latency included (median 250 ms per request of 5 to 9 questions) | one RTX 4090, median 260 to 461 ms per single question | latency compares a whole request with a single question. Not a throughput comparison |
| 14 | token counts as TypeSafe bills them (1080 to 2293 input tokens per request) | Ollama prompt tokens per question (259 to 1403) | token counts are not comparable between the two |

Same for both sides: the cases, the request builder, the service's own
instructions and criteria, the CV entries cut at 600 characters by the
service, the line splitter that drops one-word lines, the scoring code, and
the blind reference the arms are scored against.

## What this means for reading the results

Agreement with Jev and accuracy against the reference compare answers, and
both sides answered the same questions on the same state. The probability
values are produced in two different ways and are best compared within an
arm, not across. A claim that a local model "matches Jev" holds for this
harness, these quantisations and these runtime versions on 2026-10-01.
