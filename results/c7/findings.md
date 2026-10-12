# C7 findings: long context on one RTX 4090

Design `plan/c7.md` (committed 2026-10-09 before any arm), generator
`bench/gen-c7.py`, trials `cases/c7/trials.jsonl`, shared rules
`plan/c0-common.md`. Runs: 2026-10-11 22:35 to 2026-10-12 04:57, PC job
queue, each arm in its own private `ollama serve` (Ollama 0.35.0, one
request at a time, f16 KV cache, flash attention as Ollama decides, host
prompt cache off), temperature 0, `num_ctx` = target plus 512, thinking off
(gpt-oss: `think: "low"`). 540 prompts per model: German GermanQuAD filler
(CC BY 4.0) at 4k, 16k and 32k tokens, three tasks (`single`, `multikey`,
`multivalue`), needle depth 10 %, 50 % and 90 %, 20 trials per cell. Scores:
`results/c7/score.md`, chart data: `results/c7/summary.json`, capability
row: `results/capabilities.json`. The scorer judges every answer again with
the generator's rule, and `bench/check-cap.py` does so with its own code
(21 C7 checks agree). Notes: the dated sections at the end of `plan/c7.md`.

## Sanity gate

Internal check, no published reference applies: at least 95 % right on the
60 `single` prompts at 4k, run first in each job. All seven passed:
qwen3-8b, qwen3-14b, gpt-oss-20b and qwen3.8-27b 60 of 60, gemma4-12b,
gemma4-26b and mistral-small-3.2 59 of 60.

## Result

Accuracy pooled over tasks and depths, 180 prompts per length. "Without
cut prompts" leaves out the prompts Ollama cut to about half (below), a
descriptive number decided after seeing the rows.

| arm | 4k | 16k | 32k | 95 % CI at 32k | 32k without cut prompts | prompt processing at 32k, s | peak VRAM at 32k GB | J per 32k prompt |
|---|---|---|---|---|---|---|---|---|
| qwen3-14b | 1.000 | 1.000 | **1.000** | 1.000 to 1.000 | 1.000 | 8.1 | 14.0 | 6,412 |
| qwen3.8-27b | 1.000 | 0.989 | **1.000** | 1.000 to 1.000 | 1.000 | 13.5 | 19.7 | 9,373 |
| gpt-oss-20b | 1.000 | 0.994 | **1.000** | 1.000 to 1.000 | 1.000 | 2.7 | 13.2 | 1,923 |
| mistral-small-3.2 | 0.994 | 0.922 | 0.989 | 0.972 to 1.000 | 1.000 | 9.3 | 19.6 | 6,094 |
| qwen3-8b | 1.000 | 1.000 | 0.983 | 0.961 to 1.000 | 0.983 | 4.1 | 10.0 | 3,395 |
| gemma4-12b | 0.989 | 0.928 | 0.944 | 0.911 to 0.972 | 1.000 | 6.0 | 10.5 | 4,391 |
| gemma4-26b | 0.989 | 0.928 | 0.944 | 0.911 to 0.972 | 1.000 | 4.2 | 19.2 | 2,998 |

- **Retrieval holds up to 32k tokens for every model tested.** Every arm is
  at or above 90 % at every length, so the longest length at 90 % is 32k for
  all seven. qwen3-14b, qwen3.8-27b and gpt-oss-20b answer all 180 prompts
  at 32k. mistral-small-3.2 (p 0.292) and qwen3-8b (p 0.112) are not
  separated from them by the paired bootstrap; gemma4-12b and gemma4-26b
  are (difference 0.056, 95 % CI 0.028 to 0.089). Lengths above 32k were
  not tested.
- **The Gemma and Mistral misses are prompts Ollama cut, not lost needles.**
  97 of the 3,780 prompts record a prompt length of 51 % to 56 % of the
  target: gemma4-12b and gemma4-26b 34 each, mistral-small-3.2 25,
  qwen3.8-27b 4. Every one with the asked needle at depth 10 % or 50 % is
  wrong (22 of the 25 wrong Gemma answers say the text holds no such code): the
  text tokenized longer than `num_ctx` and Ollama dropped the start of the
  prompt. Without these prompts every arm is at 0.983 or above at every
  length; Gemma 4 and Mistral answer every uncut prompt right. The two Gemma
  arms share a tokenizer, so the same prompts were cut, which is why their
  rows are identical. For an SME this is the practical lesson: Ollama cuts
  an over-long prompt without an error, so `num_ctx` needs headroom over
  the longest input in the model's own tokens.
- **The real misses are few.** qwen3-8b got 3 prompts wrong at 32k (all
  needles at depth 50 %, `multikey` and `multivalue`), gpt-oss-20b 1 at 16k
  (an empty answer). No other arm missed an uncut prompt.
- **Everything fits on the card at 32k.** No arm was offloaded at any
  length (layer counts from the private server's log: all layers on the
  GPU). Peak VRAM over idle at 32k runs from 10.0 GB (qwen3-8b) to 19.7 GB
  (qwen3.8-27b), so the `q8_0` second pass the design reserved for models
  that do not fit did not run.
- **Prompt processing is the cost of length.** At 32k tokens the model reads
  the prompt in 2.7 s (gpt-oss-20b), 4.1 s (qwen3-8b), 4.2 s (gemma4-26b),
  6.0 s (gemma4-12b), 8.1 s (qwen3-14b), 9.3 s (mistral-small-3.2) and
  13.5 s (qwen3.8-27b), median per prompt, before the first answer token.
  GPU energy per 32k prompt ranges from 1.9 kJ (gpt-oss-20b) to 9.4 kJ
  (qwen3.8-27b).
- **The roadmap rule shows qwen3-14b on the capability map**: three arms are
  at 1.000 and not separated, and qwen3-14b needs the least memory of the
  three (14.0 GB at 32k). The smallest arm not separated from it is qwen3-8b
  (0.983, 10.0 GB).

## Caveats

- One run per arm, temperature 0. The needles are generated per trial, so
  no model has seen them; the filler is German Wikipedia text, which most
  models have seen.
- The task is retrieval of exact codes from unrelated filler (in the style
  of RULER). It says nothing about reasoning over a long contract, and a
  perfect score here is a lower bar than understanding the document.
- Prompt processing time is Ollama's prompt evaluation time per call with
  one request at a time and the host prompt cache off.
