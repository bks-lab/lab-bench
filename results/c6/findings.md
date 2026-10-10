# C6 findings: German reading comprehension on one RTX 4090

Design `plan/c6.md` (committed 2026-10-09 before any arm), method
`results/c6/method.md`, shared rules `plan/c0-common.md`. Runs: 2026-10-10,
PC job queue, Ollama 0.35.0 for the six LLM arms, transformers 4.57.6 for
the trained reader. Data: GermanQuAD test (deepset, CC BY 4.0), the reader
on all 2,204 questions for the gate, every arm scored on the same 1,000
sampled questions (`cases/c6/`). Scores: `results/c6/score.md`, chart data:
`results/c6/summary.json`, capability row: `results/capabilities.json`.
F1 is recomputed from the answers by the scorer and, with its own code, by
`bench/check-cap.py` (7 checks agree). Notes during the run (the gate miss
and its cause, the Winnow-12B template): the dated sections of 2026-10-10
at the end of `plan/c6.md`.

## Sanity gate

GELECTRA-large on all 2,204 test questions: F1 87.10 against the card's
88.1, tolerance 2 points. Passed on the second run. The first run, with the
answer cap of 30 tokens the design named, reached only 84.31: 9.7 % of the
gold answers are longer than that, up to 187 tokens. With the cap at 200
the gap closes; exact match is 73.0 there (card 68.6).

## Result

| arm | params | F1 | 95 % CI | exact match | answer verbatim in the passage | "keine Antwort" | ms per question | peak VRAM GB |
|---|---|---|---|---|---|---|---|---|
| gelectra-large (trained reader) | 0.34 B | **0.872** | 0.855 to 0.888 | 0.719 | 1.000 | 0 | 28 | 1.7 |
| gemma4-12b | 11.9 B | 0.849 | 0.832 to 0.866 | 0.656 | 0.910 | 27 | 302 | 9.2 |
| winnow-12b | 11.9 B | 0.848 | 0.832 to 0.866 | 0.652 | 0.914 | 26 | 448 | 13.5 |
| mistral-small-3.2 | 24.0 B | 0.801 | 0.782 to 0.819 | 0.549 | 0.901 | 15 | 416 | 16.0 |
| gemma4-26b | 25.2 B (MoE) | 0.778 | 0.759 to 0.798 | 0.525 | 0.914 | 25 | 264 | 19.0 |
| qwen3.8-27b | 27.3 B | 0.751 | 0.729 to 0.772 | 0.541 | 0.838 | 104 | 541 | 18.1 |
| qwen3-8b | 8.2 B | 0.676 | 0.654 to 0.696 | 0.345 | 0.861 | 1 | 248 | 6.3 |
| best-overlap sentence (baseline) | none | 0.221 | 0.204 to 0.239 | | | | | |

- **The small trained reader beats every LLM.** GELECTRA-large, 335 million
  parameters trained on GermanQuAD, reaches F1 0.872 on the 1,000 sampled
  questions; the best LLM, gemma4-12b, 0.849. The paired bootstrap
  separates the two (difference 0.023, p 0.024). The reader needs 1.7 GB of
  GPU memory and 28 ms per question, against 9.2 GB and 302 ms, and about
  11 GPU joules per question against 79. It was trained on this dataset's
  training split, so the comparison is a reader on its own ground against
  general models without examples.
- **LLMs answer longer and less literally.** Gold answers have a median of
  3 words. Exact match drops from 0.719 for the reader to 0.656 for
  gemma4-12b and 0.345 for qwen3-8b, whose median answer has 10 words; in
  52 % of its answers it wrote more words than the longest gold answer.
  Between 8.6 % and 16.2 % of the LLM answers are not a verbatim span of the
  passage although the prompt asks for one (the reader copies by
  construction).
- **Size does not help here.** gemma4-26b (0.778) is below gemma4-12b
  (0.849), and qwen3.8-27b (0.751) answered "keine Antwort" to 104
  questions whose answer is in the passage (GermanQuAD's test questions
  all have one). gemma4-12b and its German fine-tune winnow-12b lie 0.001
  apart (0.849 and 0.848; no test between the two was planned).
- **The roadmap rule shows the reader on the capability map**: best value,
  separated from every other arm. Among the LLMs, gemma4-12b is the
  recommendation at 9.2 GB.
- Removing German instead of English articles changes F1 by 0.002 to 0.007
  per arm and no order.
- Every model fits the card; nothing was offloaded.

## Caveats

- One run per arm, temperature 0. GermanQuAD passages come from German
  Wikipedia, which nearly every model has seen, and GermanQuAD itself is
  public since 2021.
- Winnow-12B runs through the bench's raw `gemma4` template with an empty
  thought channel, the other LLM arms through Ollama's chat template with
  thinking off (`plan/c6.md`, notes of 2026-10-10).
- F1 counts tokens after the SQuAD normalisation; a correct answer phrased
  differently scores below 1.
