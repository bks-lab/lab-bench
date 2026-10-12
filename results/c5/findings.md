# C5 findings: translation German to English and English to German

Design `plan/c5.md` (committed 2026-10-09 before any arm, preparation notes
of 2026-10-12), runner `bench/c5/run_c5.py`, scorer `bench/c5/score_c5.py`,
shared rules `plan/c0-common.md`. Runs: 2026-10-12 on the workstation queue
(one RTX 4090, Ollama 0.35.0 for the LLM arms, torch 2.11.0 with CUDA 12.8
for the dedicated MT arms). Data: FLORES+ devtest, 1,012 sentence pairs, both
directions, 2,024 segments per arm. The text is gated and ShareAlike, so
neither the sentences nor the translations are in this repository; only the
scores are (`results/c5/score.json`, `results/c5/score.md`), together with the
job meta and queue status of every arm (`results/c5/<arm>/`). Capability row:
`results/capabilities.json`, rebuilt by `bench/score-cap.py score`
(`bench/cap_c5.py`).

## Sanity gate

OPUS-MT, plain chrF against the published leaderboard value, tolerance 1.5
points: de to en 66.66 against 66.70, en to de 63.60 against 63.63. Passed,
and all other arms ran after it.

## Result

chrF++ (sacrebleu 2.6.0, primary), BLEU (13a) and mean COMET
(`Unbabel/wmt22-comet-da`, a learned metric, not an LLM judge) per direction.
The interval is the 95 % bootstrap half width over sentences (1,000
resamples, seed 20261009); p is the paired bootstrap against the best arm of
that direction. Copying the source gives chrF++ 20.8 (de to en) and 18.9 (en
to de).

| arm | chrF++ de-en | p | chrF++ en-de | p | BLEU de-en / en-de | COMET de-en / en-de |
|---|---|---|---|---|---|---|
| gemma4-26b | **68.1** (±0.7) | best | **65.5** (±0.8) | best | 44.3 / 42.1 | 0.898 / 0.892 |
| gemma4-12b | 67.9 (±0.8) | 0.060 | 65.1 (±0.9) | 0.053 | 43.8 / 41.7 | 0.898 / 0.888 |
| eurollm-9b | 67.3 (±0.8) | 0.001 | 64.8 (±0.9) | 0.008 | 43.5 / 41.5 | 0.895 / 0.887 |
| madlad-3b | 67.5 (±0.8) | 0.010 | 63.6 (±0.9) | 0.001 | 45.0 / 40.0 | 0.894 / 0.881 |
| mistral-small-3.2 | 67.5 (±0.8) | 0.003 | 63.6 (±0.8) | 0.001 | 43.8 / 39.5 | 0.897 / 0.886 |
| qwen3.8-27b | 66.7 (±0.8) | 0.001 | 62.5 (±0.8) | 0.001 | 42.8 / 37.6 | 0.898 / 0.888 |
| opus-mt | 64.9 (±0.8) | 0.001 | 61.1 (±0.8) | 0.001 | 40.6 / 36.1 | 0.883 / 0.847 |

Machine (GPU only, whole job; peak VRAM over idle; segments per second at
batch 1, and at batch 32 for the dedicated MT models):

| arm | seg/s b1 (de-en / en-de) | seg/s b32 | J per 1,000 segments | peak VRAM GB |
|---|---|---|---|---|
| opus-mt | 6.15 / 5.82 | 85.6 / 83.3 | 14,500 | 1.9 |
| gemma4-26b | 5.53 / 4.25 | | 47,700 | 18.5 |
| gemma4-12b | 4.84 / 3.53 | | 73,800 | 8.7 |
| eurollm-9b | 2.62 / 2.27 | | 111,500 | 9.4 |
| madlad-3b | 0.61 / 0.47 | 9.6 / 8.7 | 131,400 | 9.4 |
| qwen3.8-27b | 2.30 / 1.70 | | 166,700 | 17.6 |
| mistral-small-3.2 | 1.93 / 1.56 | | 202,000 | 14.8 |

- **The two Gemma 4 models lead in both directions, and the paired test does
  not separate them.** gemma4-26b is first on chrF++, gemma4-12b is 0.2 to 0.4
  points behind (p 0.060 and 0.053, just above 0.05). The capability row
  therefore shows gemma4-12b by the roadmap rule (less peak VRAM, 8.7 GB
  against 18.5 GB); the mean chrF++ over both directions is 66.50 against
  66.82.
- **A general LLM beats the dedicated models, and the small one costs a
  fraction.** The dedicated opus-mt (0.3 GB per direction) is last on every
  metric, 3 to 4 chrF++ points behind the Gemma models, but it runs at 6
  segments per second at batch 1 and 85 at batch 32 on 1.9 GB and 14.5 kJ per
  1,000 segments, against 8.7 GB and 73.8 kJ for gemma4-12b. For a bulk
  translation job the question is whether three to four points matter.
- **MADLAD-400 3B is the slowest arm** (0.5 to 0.6 segments per second at
  batch 1, 69 minutes of GPU for the job) and its quality sits in the middle:
  the best BLEU in de to en (45.0) but chrF++ 63.6 in en to de. At batch 32
  it is 15 times faster, but 5 of 1,012 outputs per direction differ from the
  batch 1 output.
- **The metrics do not give one ranking.** qwen3.8-27b is last of the LLMs on
  chrF++ and BLEU (en to de 62.5 and 37.6) but ties for the best COMET in de
  to en (0.898) and is second in en to de (0.888). The reference uses mostly
  straight quotes; qwen3.8-27b, eurollm-9b and madlad-3b write typographic
  quotes in 105, 145 and 105 of the 1,012 de to en outputs (gemma4-12b: 1).
  This costs chrF++ and is invisible to COMET; we did not diagnose the rest
  of the gap, so the qwen3.8-27b rank is not a statement about meaning.
- **COMET separates opus-mt, little else.** opus-mt has 0.883 and 0.847, the
  other six arms lie within 0.894 to 0.898 (de to en) and 0.881 to 0.892 (en
  to de). COMET was computed once per arm and not bootstrapped; no
  significance claim is made between the LLMs on it.

## Anomalies

- gemma4-12b: one of 2,024 calls failed (segment 214, en to de, HTTP 500
  "prediction aborted, token repeat limit reached"). It is scored as an empty
  translation, which lowers its en to de chrF++ slightly; the other 2,023
  calls ended normally. No other arm had a failed or empty output, none was
  truncated (no output over 600 characters, `done_reason` `stop` for every
  Ollama call), and no arm copied the source except one segment of
  mistral-small-3.2 (en to de).
- `think` was off where the model has the setting; qwen3.8-27b shows no
  thinking text in its outputs.
- EuroLLM-9B ran as a community GGUF (bartowski, Q8_0), not the original
  weights; the other LLM arms are Ollama library builds at Q4_K_M. The
  quantisation differs between arms, so the LLM comparison includes it.
- Energy covers the whole job including model load; for opus-mt and madlad-3b
  it includes the extra batch 32 pass.
- FLORES is the standard MT evaluation set since 2021 and may be in the
  training data of some arms; the dataset's gating limits, but does not
  remove, that risk.
- One sentence per call, German and English only, news and encyclopedic text:
  no statement about contracts, mail threads or long documents.
