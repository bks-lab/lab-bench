# C2 findings: German speech to text on one RTX 4090

Design `plan/c2.md` (committed 2026-10-09 before any run), shared rules
`plan/c0-common.md`. Runs: 2026-10-09, PC job queue. Data: FLEURS de_de
test, 862 utterances, 347 sentences, 3.15 h of read Wikipedia sentences.
Scores: `results/c2/score.md`, chart data: `results/c2/summary.json`,
capability row: `results/capabilities.json`. The WER of every arm is
recomputed from the raw reference and hypothesis with a separate normaliser
and edit distance in `bench/check-cap.py` (6 checks agree). Notes after the
run: the dated section of 2026-10-10 at the end of `plan/c2.md`.

## Sanity gate

Passed. canary-1b-v2: WER 4.52 against the card's 4.40 (tolerance 1.5
points). parakeet-v3, reported without gating: 5.16 against the card's
5.04.

## Result

| arm | params | WER % | 95 % CI | CER % | x real time (batch 1) | audio h per hour (batch 16) | peak VRAM GB | GPU J per audio h |
|---|---|---|---|---|---|---|---|---|
| canary-1b-v2 | 979 M | **4.52** | 3.95 to 5.16 | 1.50 | 22 | 145 | 22.9 (torch 11.6) | 11,322 |
| whisper-large-v3 | 1.55 B | 4.70 | 3.95 to 5.50 | 2.02 | 23 | 94 | 8.9 | 22,235 |
| parakeet-v3 | 627 M | 5.16 | 4.56 to 5.75 | 1.51 | 182 | 593 | 16.3 (torch 7.9) | 2,153 |
| whisper-large-v3-turbo | 809 M | 5.63 | 4.55 to 6.88 | 2.37 | 69 | 140 | 4.5 | 8,706 |
| whisper-turbo-de | 809 M | 5.85 | 5.20 to 6.51 | 2.14 | 64 | 158 | 4.5 | 8,815 |
| distil-whisper-de | 756 M | 13.32 | 11.83 to 14.94 | 7.38 | 90 | 162 | 4.4 | 7,285 |

- **canary-1b-v2 and whisper-large-v3 are not separated.** 4.52 against 4.70, a
  difference of 0.18 points with p 0.62 in the paired cluster bootstrap.
  canary-1b-v2 is separated from every other arm (parakeet-v3 +0.64,
  p 0.004; whisper-large-v3-turbo +1.11, p 0.026).
- **parakeet-v3 is the throughput model.** It transcribes 182 times faster
  than real time at batch 1 and 593 audio hours per wall hour at batch 16,
  at 2,153 GPU joules per audio hour, a tenth of whisper-large-v3. Its WER
  of 5.16 is 0.46 points above whisper-large-v3, a difference the test does
  not establish (p 0.18).
- **Long utterances split the Whisper arms.** On the 841 utterances up to
  30 s whisper-large-v3 has the lowest WER of all (4.13, canary-1b-v2
  4.58). On the 21 utterances over 30 s it reaches 17.6 and
  whisper-large-v3-turbo 32.3, while canary-1b-v2 stays at 3.2 and
  parakeet-v3 at 4.9. faster-whisper ran without VAD, as the plan says; its
  long-form path cuts at 30 s windows and loses or repeats text at the
  seams. These splits are descriptive, not preregistered, and the long
  group is small. Meeting audio is long-form, so this matters more for the
  use case than the headline.
- **The German fine-tunes did not beat their base.** primeline's
  whisper-large-v3-turbo-german scores 5.85 against 5.63 for the original
  turbo (worse on short utterances, better on the long ones), and the
  distilled German model 13.32, with 64 % WER on the long utterances.
- **Digits.** 20 % of the utterances contain digits; mapping them with
  num2words changes the WER by at most 0.7 points per arm (column in
  `score.md`) and changes no order at the top.

## Capability map

The roadmap rule for ties picks the arm with less peak VRAM:
**whisper-large-v3** (Apache-2.0 weights, open source), 4.70 % WER
[3.95, 5.50], real-time factor 0.047 at batch 1 (23 times faster than real
time), 94 audio hours per wall hour at batch 16, 8.9 GB, fits on one card.
The tie with canary-1b-v2 (4.52) is stated in the row. canary's nvidia-smi
peak of 22.9 GB is mostly memory NeMo reserves (float32 weights and the
batch-16 pass); torch counted 11.6 GB, still more than whisper-large-v3.
For volume, parakeet-v3 does the 3.15 h in 62 s at batch 1.

## Limits

- Read speech, one sentence per utterance, studio-like audio. The test
  measures the floor of the error, not a meeting with crosstalk.
- One run per arm; the batch-16 throughput was measured once, not three
  times as `plan/c0-common.md` asks.
- NeMo arms ran in float32, the NeMo default; the plan named no precision
  for them.
- Exposure: FLEURS is public since 2022; the Whisper training data is not
  published, NVIDIA lists FLEURS among its evaluation sets.
