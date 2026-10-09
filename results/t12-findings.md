# T12 findings: eleven local open models on one RTX 4090

Runs of 2026-10-09, design `plan/t12.md` (merged as PR #15 before the
first run). Full per-model table: `results/t12/series.md`; chart-ready
aggregate: `results/t12/summary.json`; probes and template checks:
`results/t12/probes.md`; machine record: `results/t12/machine.json`.
Score reports: `results/route1/score.md`, `results/ex1/score.md`.

Setup: Ollama 0.35.0 on the workstation (RTX 4090, 24 GB, Windows),
called from a Mac over the LAN, one request at a time, temperature 0.
Prompts, options and parameters identical to the Winnow-12B rows of the
same day (`bench/run-local.mjs`, `bench/run-local-extract.mjs`), the raw
template per family from `bench/lib/templates.mjs`.

## Headline

- **Qwen3.8 27B is the most accurate local model on both tasks.** On
  MASSIVE it is level with Jev within about a point (scenario de 72.4 %
  against Jev 72.8 %, en 73.8 % against 74.1 %; intent 2+ de 90.5 %
  against 90.7 %, en 92.0 % against 92.0 %), and it has the best slot F1
  (54.3 % de, 55.6 % en, Winnow-12B 45.1 % and 48.9 %). It is also the
  slowest model that fits: 207 ms per decision, about 16,700 decisions per
  hour, 18.7 GiB VRAM. Not tested against Jev (descriptive only, chosen
  after the fact).
- **Gemma 4 E4B is the efficiency point.** At 5.2 GiB VRAM and 52 ms per
  decision it reaches 69.7 % / 72.9 % on MASSIVE scenario and 86.0 % /
  88.8 % on intent, more than Qwen3 32B (67.3 % / 69.8 %, 84.2 % /
  86.7 %), which needs 21.3 GiB and spills to the CPU.
- **Size helps inside one family, maker matters more across families.**
  The pre-named Qwen3 line rises from 4B to 14B to 32B on MASSIVE
  (scenario de 57.5, 65.6, 67.3 %); the preregistered large against small
  test is significant on four of five route1 rows (McNemar p < 0.001) and
  not on fast-decisions (192 against 168 discordant items, p = 0.225).
  But a 4B-effective Gemma beats the 32B Qwen, and the 3.4B Granite is the
  weakest model on MASSIVE scenario (47.4 % / 51.9 %).
- **Fast-decisions does not move with size.** Every model lands between
  54.9 % and 63.6 % on the tasks with 26 labels or fewer (Jev 63.0 %,
  Winnow-12B 63.6 %, best new model Mistral Small 3.2 at 62.8 %). No local
  model reaches the usable bar (skill 0.5, that is 65.6 %) there.
- **No format failures.** Every model kept a median letter mass of 0.5 or
  more over its full route1 run. Three models did so with a large share of
  low-mass answers (see Caveats).

## Series table

route1 accuracy per score-report group; ex1 micro slot F1 and exact frame;
speed from the Mac (LAN included); VRAM = peak nvidia-smi during the runs
minus the idle value before loading.

| class | model | quant | scen de | scen en | int2 de | int2 en | fd26 en | slot F1 de | slot F1 en | frame de | frame en | p50 ms route1 | decisions per h | p50 ms ex1 | utterances per h | VRAM GiB | on GPU only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | granite4-micro (3.4B) | Q4_K_M | 47.4 | 51.9 | 77.2 | 81.5 | 54.9 | 19.5 | 28.4 | 35.1 | 33.9 | 39 | 91,861 | 105 | 32,012 | 3.8 | yes |
| small | qwen3-4b * | Q4_K_M | 57.5 | 63.0 | 77.0 | 82.6 | 59.7 | 26.3 | 36.1 | 32.4 | 32.8 | 36 | 94,847 | 112 | 28,550 | 5.0 | yes |
| small | gemma4-e4b (7.5B, 4B eff.) | Q4_K_M | 69.7 | 72.9 | 86.0 | 88.8 | 61.1 | 39.3 | 42.2 | 23.2 | 26.3 | 52 | 66,827 | 204 | 16,478 | 5.2 | yes |
| medium | gemma4-12b | Q4_K_M | 72.2 | 71.9 | 83.8 | 87.6 | 61.6 | 47.9 | 51.9 | 45.6 | 46.1 | 82 | 38,218 | 196 | 18,020 | 9.3 | yes |
| medium | winnow-12b (Gemma 4 12B + LoRA) | Q8_0 | 71.8 | 70.6 | 89.0 | 91.6 | 63.6 | 45.1 | 48.9 | 45.1 | 44.9 | 90 | 35,874 | 397 | 9,368 | about 14 (loaded) | yes |
| medium | qwen3-14b * | Q4_K_M | 65.6 | 69.0 | 79.9 | 83.5 | 60.9 | 35.1 | 43.9 | 28.5 | 32.6 | 49 | 69,462 | 279 | 11,283 | 11.6 | yes |
| medium | phi4-14b | Q4_K_M | 67.8 | 70.3 | 85.9 | 89.0 | 59.7 | 37.8 | 43.6 | 31.5 | 35.0 | 42 | 81,939 | 260 | 12,707 | 12.0 | yes |
| large | mistral-small3.2-24b | Q4_K_M | 62.9 | 66.6 | 83.6 | 86.1 | 62.8 | 36.4 | 39.5 | 20.8 | 20.3 | 52 | 60,067 | 643 | 5,635 | 17.0 | yes |
| large | qwen3.8-27b | Q4_K_M | **72.4** | **73.8** | **90.5** | **92.0** | 62.0 | **54.3** | **55.6** | **49.9** | **49.1** | 207 | 16,739 | 351 | 9,997 | 18.7 | yes |
| large | qwen3-32b * | Q4_K_M | 67.3 | 69.8 | 84.2 | 86.7 | 60.6 | 39.8 | 44.3 | 28.9 | 29.9 | 168 | 18,463 | 563 | 5,481 | 21.3 | **no**, 1.9 GiB on CPU |
| MoE | qwen3-30b-a3b * | Q4_K_M | 59.8 | 61.5 | 70.8 | 77.0 | 56.2 | 33.9 | 39.0 | 24.4 | 29.2 | 60 | 51,935 | 228 | 15,160 | 19.2 | yes |
| German, MoE | shisa-de-1 (Gemma 4 26B-A4B) | Q4_K_M | 69.1 | 68.8 | 86.6 | 88.9 | 61.0 | 46.3 | 49.3 | 36.6 | 38.6 | 73 | 43,490 | 206 | 15,609 | 17.7 | yes |
| | Jev (hosted, for reference) | | 72.8 | 74.1 | 90.7 | 92.0 | 63.0 | n/a | n/a | n/a | n/a | 232 | | | | | |

`*` = pre-named size-class representative for the tool map. Bold = best
local model in the column. Winnow-12B was measured earlier the same day
with the same harness; its VRAM is the value after loading (no run peak).

**The 100 errors per model in route1 are by design.** The
`support_intent:intent` task has 28 labels, more than the 26 answer
letters, so its 100 questions are written as unanswered rows for every
local model (as for Winnow-12B before). The score report counts them as
wrong in "fastdec, all single-label tasks"; the group "26 options or
fewer" and `summary.json` (`route1.accuracy`, n = 14,396) leave them out.

Load time (median of three loads after an unload, file already in the OS
cache): 1.7 s (Granite) to 6.0 s (Qwen3 32B); the first, colder load of
each model took 2.9 to 22.6 s (`machine.json`, `load_wall_ms`).

## Size against accuracy

Inside the Qwen3 release (same generation, same recipe), the pre-named
columns of the tool map:

| row | small 4B | medium 14B | large 32B | MoE 30B-A3B | large vs small (McNemar, preregistered) |
|---|---|---|---|---|---|
| scenario de | 57.5 | 65.6 | 67.3 | 59.8 | 459 vs 168, p < 0.001 |
| scenario en | 63.0 | 69.0 | 69.8 | 61.5 | 368 vs 166, p < 0.001 |
| intent 2+ de | 77.0 | 79.9 | 84.2 | 70.8 | 316 vs 129, p < 0.001 |
| intent 2+ en | 82.6 | 83.5 | 86.7 | 77.0 | 229 vs 122, p < 0.001 |
| fast-decisions <= 26 | 59.7 | 60.9 | 60.6 | 56.2 | 192 vs 168, p = 0.225 |
| slot F1 de / en | 26.3 / 36.1 | 35.1 / 43.9 | 39.8 / 44.3 | 33.9 / 39.0 | no test (ex1) |

Most of the gain comes from 4B to 14B; 14B to 32B adds 1 to 4 points at
three times the latency (the 32B does not fit fully at 16k context). The
MoE model is below the 4B on intent; see the letter-mass caveat.

Across makers, size predicts little. Gemma 4 E4B (4B effective) is level
with or ahead of every 14B to 32B model on route1 except Qwen3.8 27B, and
Mistral Small 3.2 (24B) is below Phi-4 (14B) on MASSIVE.

German against English: on MASSIVE scenario most models lose 1 to 6
points in German. Only the Gemma 4 based models (Gemma 4 12B, Winnow-12B,
shisa-de-1) are level or slightly better in German. The German-tuned
shisa-de-1 is not consistently better in German than the general Gemma 4
12B: lower on scenario (69.1 against 72.2 %) and slot F1 (46.3 against
47.9 %), higher on intent (86.6 against 83.8 %).

## Best choice per task on one 24 GB card

Read off the same test data, so chosen after the fact; no test between
the candidates.

| need | choice | why |
|---|---|---|
| routing, accuracy first | qwen3.8-27b | best on every MASSIVE row, level with Jev; 16,700 decisions per hour on one stream |
| routing, throughput first | gemma4-e4b | 1 to 5 points below Qwen3.8 on MASSIVE at 4x the throughput and a quarter of the memory; leaves room for a second model |
| slot extraction | qwen3.8-27b | only model with slot F1 above 50 % in both languages (54.3 / 55.6) |
| slot extraction, faster | gemma4-12b | 47.9 / 51.9 F1, best exact frame after Qwen3.8, 18,000 utterances per hour, 9.3 GiB |
| German only | qwen3.8-27b, or gemma4-12b when speed matters | the German-tuned shisa-de-1 does not beat them overall |
| avoid | qwen3-32b at 16k context (spills to CPU, slow), granite4-micro (weak on scenario), qwen3-30b-a3b (letter-format problem) | |

Against the usable bar of the map (skill 0.5): MASSIVE scenario (56.8 %
needed) is reached by every model except Granite 4 micro; intent 2+
(75.1 % needed) by every model except Qwen3 30B-A3B in German;
fast-decisions (65.6 % needed) by none; slot extraction (F1 50 %) only by
Qwen3.8 27B (both languages) and Gemma 4 12B (English).

## Tool map

`results/toolmap.json`: route1 and ex1 rows gained `local_size_classes`
(the four Qwen3 representatives named in `plan/t12.md`) and route1 rows
`local_size_contrast` (the preregistered test). The `local-llm` family
cell stays Winnow-12B, so no existing cell changed: after removing the
three new keys and the generation date, the map is equal to the one on
main (checked with a JSON comparison). In the text diff one line per
route1 row with a `note` shows as changed only because a trailing comma
was added. `results/route1/score.md` and `results/ex1/score.md` gained the
eleven arms as rows; the per-task and per-slot-type tables gained columns;
no existing number changed.

## Independent recompute

`bench/check-t12.py` (Python, no bench library) recomputes five numbers
from the raw rows and gold files: qwen3-4b scenario de 57.5 %, qwen3-32b
intent 2+ en 86.7 %, qwen3-14b fast-decisions 26 or fewer 60.9 %,
qwen3-30b-a3b slot F1 de 33.9 %, and the McNemar large against small on
scenario de (459 and 168 discordant, p = 3.7e-32). All five equal the
reports and the map.

## Caveats

- **Letter mass.** Three models passed the 0.5 median but answered a large
  share of questions with something other than a letter: Mistral Small
  3.2 (48 % of asked questions below 0.5), Qwen3 30B-A3B (44 %, the first
  token was "The" in about half of the rows) and Phi-4 (32 %). Their
  answers on those rows come from the residual letter probabilities and
  are likely worse than a model that answers in format. Their accuracies
  are lower bounds of what a different prompt might give; no second
  prompt was run (the design fixed one probe ladder).
- **Prefix.** The probe chose `Answer: **` for Qwen3 4B, Qwen3 32B and
  Gemma 4 E4B (median mass on the empty prefix 0.005, 0.25, 0.0001) and
  the empty prefix for the others, as fixed in the design. So the prompts
  are identical except for that prefix.
- **Probe size.** 20 questions were too few to predict the full run:
  Mistral and Qwen3 30B-A3B passed the probe (0.69, 0.61) and ended with
  half of their rows below 0.5.
- **Qwen3 32B spills to the CPU** at 16,384 context (Ollama size 23.1 GiB,
  21.2 on the GPU), which explains much of its 168 ms. A smaller context
  (prompts are at most 456 tokens) would fit it; that was not part of the
  design.
- **Ollama `/api/ps` is wrong for the Gemma 4 builds** under 0.35.0 (0.3
  to 1.3 GiB reported); their fit is judged by nvidia-smi only.
- **shisa-de-1 ex1 errors.** 36 of 2,000 utterances failed: 18 times Ollama
  aborted with "token repeat limit reached", 18 times the JSON had items
  without type and text. They count as empty predictions.
- **Latency is one stream from a Mac over the LAN**, not a throughput
  limit of the card; batching or parallel requests (T10, C8) will give
  more decisions per hour. VRAM includes the KV cache for 16k context.
- **One run per model**, test data public (MASSIVE since 2022,
  fast-decisions on Hugging Face), no check for training exposure.
- **The machine is shared.** It hosts GitHub runners and the desktop; the
  queue was held during the runs (HOLD-t12) and no other GPU job ran, but
  CPU-side load was not controlled.

## Deviations from the design

- Ollama returns no `load_duration` for an empty-prompt load, so the
  driver's load time was empty. Load time was measured afterwards with
  `bench/run-t12.mjs --load-only`: three loads after an unload, wall time
  from the Mac, recorded in `machine.json`. For Winnow-12B the same pass
  recorded the VRAM after loading.
- The plan named `results/t12/probes.md` and `machine.json`; in addition
  `results/t12/series.md` and `results/t12/summary.json` (chart data, on
  request of Michael Boiman) are written by `bench/score-t12.mjs`.

Total PC time of the series: 5 h 50 min (13:15 to 19:05 UTC including
probes and template checks), plus 4 min for the load pass.
