# C8 findings: throughput and energy per model size on one RTX 4090

Design `plan/c8.md` (committed 2026-10-09 before any arm), generator
`bench/gen-c8.py`, shared rules `plan/c0-common.md`. Runs: 2026-10-10 and
2026-10-11, PC job queue. Ollama arms: Ollama 0.35.0, one private
`ollama serve` per concurrency level with `OLLAMA_NUM_PARALLEL` = the level,
host prompt cache off, `num_ctx` 5120. vLLM arms: vLLM 0.31.0 in WSL2
Ubuntu 24.04, one `vllm serve` per level with `--max-num-seqs` = the level,
prefix caching off, AWQ INT4 weights (`Qwen/Qwen3-14B-AWQ`, official; and
`cyankiwi/gemma-4-12B-it-AWQ-INT4`, a community build). Workloads:
`chat-short` (512 prompt tokens, 256 generated), `chat-long` (4,096 prompt
tokens, 256 generated), `decide` (500 one-token route1 decisions), each at
1, 4, 8 and 16 parallel clients and repeated three times. Scores:
`results/c8/score.md`, chart data: `results/c8/summary.json`, capability
row: `results/capabilities.json`. `bench/check-cap.py` recomputes the
aggregate rates from the rows and the section wall times (33 C8 checks
agree). Notes: the dated sections at the end of `plan/c8.md`.

## Sanity gate

Internal check: the three repetitions of `chat-short` at concurrency 1 agree
within a coefficient of variation of 5 %, and the GPU is idle before every
configuration. All eleven arms passed, with CVs from 0.01 % (qwen3-8b) to
0.23 % (vllm-qwen3-14b).

## Result

Aggregate generation tokens per second for `chat-short` and decisions per
hour, mean of three repetitions (ranges in `results/c8/score.md`). Power is
the mean GPU draw during the timed `chat-short` sections. "Split" marks a
level at which the model ran partly on the CPU (llama-server log).

| arm | engine, weights | tok/s at 1 | tok/s at 16 | decisions/h at 16 | J per 1,000 tokens at 16 | W at 1 | W at 16 | peak VRAM GB at 16 | at 16 |
|---|---|---|---|---|---|---|---|---|---|
| granite4-3b | Ollama Q4_K_M | 238.2 | **1,378.2** | **260,731** | **165** | 269 | 225 | 8.9 | on card |
| qwen3-8b | Ollama Q4_K_M | 141.1 | 875.8 | 190,414 | 305 | 325 | 272 | 16.4 | on card |
| gemma4-12b | Ollama Q4_K_M | 147.7 | 146.2 | 35,463 | 858 | 344 | 126 | 22.3 | split, 44 of 49 layers |
| winnow-12b | Ollama Q8_0 | 58.7 | 122.7 | 30,595 | 746 | 282 | 92 | 21.8 | split, 44 of 49 layers |
| qwen3-14b | Ollama Q4_K_M | 85.6 | 564.0 | 114,981 | 546 | 345 | 309 | 21.3 | on card |
| gpt-oss-20b | Ollama MXFP4 | 173.5 | 737.6 | 120,469 | 393 | 290 | 293 | 14.7 | on card |
| mistral-small-3.2 | Ollama Q4_K_M | 54.9 | 44.2 | 19,345 | 1,706 | 366 | 76 | 21.4 | split, 32 of 41 layers |
| gemma4-26b | Ollama Q4_K_M | 178.3 | 231.9 | 31,390 | 662 | 255 | 154 | 22.3 | split (weights moved off the card) |
| qwen3.8-27b | Ollama Q4_K_M | 73.4 | 73.9 | 26,816 | 5,551 | 410 | 414 | 18.1 | on card, one slot |
| vllm-qwen3-14b | vLLM AWQ INT4 | 79.6 | 789.5 | 138,916 | 411 | 272 | 331 | 19.7 | on card |
| vllm-gemma4-12b | vLLM AWQ INT4 | 76.8 | 708.8 | 156,645 | 430 | 261 | 307 | 20.9 | on card |

- **One user: a mixture of experts and the small models are fastest.** At
  concurrency 1, gemma4-26b (about 4 B active parameters) generates 178.3
  tokens per second, gpt-oss-20b 173.5, gemma4-12b 147.7, qwen3-8b 141.1 and
  granite4-3b 238.2. The dense 24 B to 27 B models stay at 54.9
  (mistral-small-3.2) and 73.4 (qwen3.8-27b).
- **Many users: the models that keep 16 caches on the card scale, the
  others collapse.** From 1 to 16 parallel clients `chat-short` throughput
  grows 5.8 times for granite4-3b, 6.2 times for qwen3-8b, 6.6 times for
  qwen3-14b and 4.3 times for gpt-oss-20b. gemma4-12b, winnow-12b,
  mistral-small-3.2 and gemma4-26b peak at 8 clients and fall at 16, where
  their 16 caches of 5,120 tokens no longer fit next to the weights and
  llama-server moves layers or weights to the CPU: gemma4-12b drops from
  397.0 tokens per second at 8 to 146.2 at 16, mistral-small-3.2 from 227.2
  to 44.2. For gemma4-12b and winnow-12b `/api/ps` reported the model fully
  on the card; only the server log shows the split (`plan/c8.md`, note of
  2026-10-11).
- **qwen3.8-27b does not run in parallel at all.** llama-server started it
  with one slot at every level (layers.json), so 16 clients queue behind one
  another: 73.9 tokens per second at 16, a median time to first token of
  1.7 s for `chat-long` at 1 client and a 95th percentile of 70.3 s at 16.
- **Ollama against vLLM (engine plus 4-bit build, not engine alone).** With
  one user Ollama's Q4_K_M is faster: qwen3-14b 85.6 against 79.6 tokens per
  second, gemma4-12b 147.7 against 76.8. With 16 users vLLM's AWQ builds are
  ahead: qwen3-14b 789.5 against 564.0 tokens per second (1.4 times) and
  138,916 against 114,981 decisions per hour; gemma4-12b 708.8 against
  146.2 tokens per second and 156,645 against 35,463 decisions per hour,
  because Ollama splits gemma4-12b at 16 clients and vLLM does not. vLLM
  reserves 85 % of the card at start, so its VRAM column says nothing about
  what the model needs, and each level took 58 to 146 s to start.
- **Energy.** At 16 clients granite4-3b needs 165 GPU joules per 1,000
  generated tokens, qwen3-8b 305, gpt-oss-20b 393, the vLLM arms 411 and
  430, qwen3-14b 546; qwen3.8-27b, with one slot, 5,551. Mean GPU draw stays
  between 255 W and 410 W for every arm at one client; the split
  configurations draw less (76 W to 154 W) while delivering less, which fits
  a GPU that waits for the CPU; low wattage there is not efficiency.
- **Capability map.** C8 has no quality metric; the headline is the arm with
  the most decisions per hour at 16 clients among the Ollama arms that stay
  on the card: granite4-3b, 260,731 per hour at 3,169 GPU joules per 1,000
  decisions. That ranks throughput only; C4, C6 and C7 say what the models
  answer.

## Caveats

- One card, one power limit (450 W), one Ollama version. Throughput depends
  on prompt and answer lengths; the workloads are fixed and synthetic.
- Time to first token is measured by the client and includes waiting for a
  free slot above concurrency 1.
- vLLM rows run AWQ INT4 builds, the Ollama rows Q4_K_M (Winnow: Q8_0). The
  gemma4-12b AWQ build is a community conversion. vLLM reports no
  per-request prompt durations, so its prompt-processing rate is missing.
- vllm-gemma4-12b refused one `chat-long` prompt (index 58 of 64) with HTTP
  400 in all 12 timed passes; its `chat-long` rates count 63 prompts
  (`plan/c8.md`, note of 2026-10-12).
