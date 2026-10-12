# T10 findings: throughput with batching (GLiNER Decide)

Design `plan/t10.md` (committed 2026-10-12 before the run), runner
`bench/t10/run_t10.py`. Run: 2026-10-12, workstation queue, one RTX 4090,
gliner2 2.0.0, torch 2.11.0+cu128, `fastino/GLiNER2.5-multi-Decide` revision
`e173bca1f0c4`. Data: route1 MASSIVE de, 5,948 questions in file order. Batch
sizes 1, 8, 32 and 128, three timed passes each, one warm-up batch before
every pass; wall time includes tokenisation and schema handling, with a CUDA
synchronise at the end. Model load 12.6 s, reported separately. Job meta and
status: `results/t10/gliner2.5-multi-decide/`; the choices of the first
pass per batch size are in the compact rows next to them.

## Sanity gate

The batch 1 choices agree with the committed route1 run 1 on 100 % of the
5,948 questions (minimum 99 %). Passed.

## Result

| batch | decisions per second (mean, range of 3) | decisions per hour | agreement with batch 1 | peak allocated GPU memory |
|---|---|---|---|---|
| 1 | 69.8 (69.3 to 70.1) | 251,000 | 100 % | 1.1 GB |
| 8 | **399.5** (399.4 to 399.6) | 1,438,000 | 100 % | 1.4 GB |
| 32 | 380.6 (378.3 to 383.4) | 1,370,000 | 100 % | 2.2 GB |
| 128 | 354.4 (353.0 to 356.5) | 1,276,000 | 100 % | 5.5 GB |

- **Batching gives 5.7 times the throughput at batch 8**, 399.5 decisions per
  second against 69.8 at batch 1; the answers do not change (all four batch
  sizes agree with batch 1 on all 5,948 questions, so padding moved no near
  tie).
- **Bigger batches are slower, not faster.** Batch 32 is 5 % and batch 128
  11 % below batch 8, with 1.6 times and 4 times the memory of batch 8. On this card
  and this text length the peak is at a small batch; the likely reason is
  padding to the longest item and the per-question schema handling, which we
  did not profile. A bulk job should use batch 8.
- **Batch 1 matches route1.** 69.8 decisions per second is 14.3 ms per
  question, the route1 median of 14 ms.
- **Energy.** The job drew 76.0 kJ GPU energy in 6.9 GPU minutes (mean 183.5
  W), covering load and all 12 passes (71,376 decisions) and the warm-ups,
  so about 1.06 J per decision on average; per batch size it was not
  separated (one job, one sampler). The GPU log peaks at 12.6 GB total, 10.6
  GB over idle, higher than the allocated 5.5 GB because the allocator
  caches freed blocks.
- For comparison, C8 measured route1 decisions per hour for local LLMs at
  Ollama concurrency 16: up to 260,731 (granite4-3b), 138,916 (vLLM
  qwen3-14b), 26,816 (qwen3.8-27b). GLiNER at batch 8 is 1,438,000, five
  times the fastest of those, but with a 56 % accuracy on MASSIVE de against
  81 to 82 % for the LLMs (T2), and a different, shorter prompt per question.

## Anomalies

- One job, one model load, one machine; the range is over three passes within
  that load. The queue sampler logs once per second, so no per-batch energy.
- Text length is that of MASSIVE utterances (short); the optimum batch size
  may move for long documents.
- The local LLMs were not re-measured here (plan/t10.md), so the comparison
  with C8 mixes different loads and the question wording.
