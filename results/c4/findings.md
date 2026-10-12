# C4 findings: structured output and tool calling on one RTX 4090

Design `plan/c4.md` (committed 2026-10-09 before any arm), method
`results/c4/method.md`, shared rules `plan/c0-common.md`. Runs: 2026-10-10
and 2026-10-11, PC job queue, Ollama 0.35.0, temperature 0, `num_ctx` 8192,
thinking off (gpt-oss: `think: "low"`). Data: BFCL v3 non-live AST
categories, all 1,000 items (gorilla `6ea57973c7a6`, Apache-2.0), and 600
JSONSchemaBench schemas (MIT). Scores: `results/c4/score.md`, chart data:
`results/c4/summary.json`, capability row: `results/capabilities.json`.
Part A is checked by BFCL's own AST checker (`bench/c4/check_ast.py`,
per-item verdicts next to each run); `bench/check-cap.py` recounts the Part A
totals from the verdicts and validates every Part B answer again with
`jsonschema` on its own (24 C4 checks agree). Notes during the run (gate, run
time, the Ollama outage in the qwen3.8-27b arm): the dated sections at the
end of `plan/c4.md`.

## Sanity gate

qwen3-14b in `fc` mode, run alone first: `simple_python` 93.50 % against
the BFCL leaderboard's 95.25 % for Qwen3-14B (FC), `multiple` 94.00 %
against 93.00 %, tolerance 5 points. Passed. The leaderboard row ran with
thinking on and full precision, this run with thinking off and Q4_K_M.

## Result

AST accuracy over the 1,000 BFCL items; schema validity over the 600
schemas (`constrained`: the 592 schemas Ollama accepts as `format`).

| arm | params | fc (native tools) | 95 % CI | prompt mode | free JSON | constrained JSON | ms per item fc | J per 1,000 items fc | peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-14b (gate arm) | 14.8 B | **0.919** | 0.901 to 0.935 | 0.946 | 0.810 | 0.907 | 606 | 288,452 | 10.3 |
| qwen3.8-27b | 27.3 B | 0.906 | 0.888 to 0.923 | not measured | 0.905 | **0.934** | 886 | 377,113 | 22.3 * |
| qwen3-8b | 8.2 B | 0.892 | 0.872 to 0.911 | 0.920 | 0.762 | 0.907 | 374 | 161,298 | 6.3 |
| gemma4-26b | 25.2 B (MoE) | 0.881 | 0.861 to 0.900 | **0.947** | 0.873 | 0.926 | 252 | 70,208 | 19.0 |
| gemma4-12b | 11.9 B | 0.877 | 0.857 to 0.897 | 0.913 | 0.805 | 0.897 | 300 | 103,827 | 9.2 |
| granite4-3b | 3.4 B | 0.869 | 0.849 to 0.889 | 0.770 | 0.733 | 0.895 | 223 | 75,992 | 3.1 |
| mistral-small-3.2 | 24.0 B | 0.816 | 0.791 to 0.840 | 0.934 | 0.813 | 0.927 | 726 | 399,689 | 16.2 |
| gpt-oss-20b | 20.9 B | 0.508 | 0.479 to 0.539 | 0.012 | 0.793 | 0.900 | 334 | 108,954 | 12.6 |

\* qwen3.8-27b: `fc` comes from the retry job (peak 18.1 GB over idle,
19.9 GB in total), the JSON modes from the first job, in which the card
reached 24.0 GB after the Ollama outage; the arm is marked `offloaded` by
the C0 rule (caveats below, and `plan/c4.md`, note of 2026-10-12).

- **Native tool calling: qwen3-14b and qwen3.8-27b are not separated.**
  qwen3-14b reaches 0.919, qwen3.8-27b 0.906; the paired bootstrap gives a
  difference of 0.013 (95 % CI -0.008 to 0.033, p 0.222). Every other arm
  is separated from qwen3-14b (p 0.010 or lower). By the roadmap rule the
  capability map shows qwen3-14b, which needs 10.3 GB against 22.3 GB.
- **Small models are close.** granite4-3b (3.4 B parameters, 3.1 GB) is
  0.050 behind the best in `fc` mode, qwen3-8b 0.027. In `simple_python`
  and `multiple` all arms except gpt-oss lie between 0.830 and 0.968; the
  gap opens in `parallel` and `parallel_multiple`, where one answer must
  hold several calls.
- **Prompt mode beats native tool calling for five of the seven arms with
  both modes.** With the functions as JSON in the system prompt and the
  answer parsed from text, qwen3-14b reaches 0.946, gemma4-26b 0.947 and
  mistral-small-3.2 0.934, against 0.919, 0.881 and 0.816 in `fc` mode
  (exact McNemar p 0.005 or lower for all five: these three, qwen3-8b and
  gemma4-12b). The gain sits mostly in the parallel categories (gemma4-26b
  `parallel` 0.800 in `fc`, 0.950 in prompt mode). granite4-3b goes the
  other way (0.770 in prompt mode, 11.3 % of its answers not parseable), and
  so does gpt-oss-20b (next point). Native tool calling through Ollama is
  not automatically the more reliable path.
- **gpt-oss-20b fails through the Ollama integration, not on content.** In
  `fc` mode it returned at most one tool call per item (980 of 1,000 items
  one call, 20 none), so it scores 0.000 on `parallel` and
  `parallel_multiple`, while `simple_python` (0.843) and `multiple` (0.855)
  are in the range of the others. In prompt mode Ollama's parser for the
  model's output format turned 974 of 1,000 answers into tool calls,
  although no tools were offered, and left the text empty, which is what
  prompt mode reads: 0.012. Both numbers describe gpt-oss under Ollama
  0.35.0 as an SME would run it, not the model's tool-calling skill alone.
- **Constrained decoding helps every model.** With the schema as Ollama
  `format`, schema validity rises to 0.895 to 0.934 for all eight arms
  (free: 0.733 to 0.905); McNemar p 0.040 for qwen3.8-27b and 0.0001 or
  lower for the others. The weakest free generator gains most: granite4-3b
  from 0.742 to 0.895 on the same 592 schemas. Constrained decoding is not
  slower here: the median time per schema is 0.79 to 0.99 times that of
  free generation. 8 of the 600 schemas (1.3 %) are refused by Ollama's
  schema converter, the same 8 for every arm. Even constrained, 39 to 62 of
  the 592 answers per arm do not validate, 6 to 32 of them per arm because
  they hit the 4,096-token cap (rows and verdicts), so a validator after
  the model stays necessary.
- **Speed and energy.** Median time per tool-calling item ranges from
  223 ms (granite4-3b) and 252 ms (gemma4-26b, a mixture of experts with
  about 4 B active parameters) to 886 ms (qwen3.8-27b). gemma4-26b needs
  70 kJ per 1,000 items, qwen3-14b 288 kJ and mistral-small-3.2 400 kJ (GPU
  only, timed section).

## Caveats

- One run per arm, temperature 0. BFCL is public since 2024 and widely used
  in tool-calling fine-tuning, JSONSchemaBench since 2025; both are likely
  in the training data of recent models.
- qwen3.8-27b: its prompt mode is not measured (741 of 1,000 calls failed
  during an outage of the workstation's Ollama service on 2026-10-11, and
  no retry ran), and its JSON-mode timings are not comparable with the
  other arms, because they ran after the outage with the card at 24.0 GB.
  The `fc` retry returned the same answers as the first job on all 892
  items before the outage. Details: `plan/c4.md`, note of 2026-10-12.
- Peak VRAM is the peak of the whole job over the idle value. Validity says
  only that the JSON matches the schema, not that its content is sensible;
  that is not scored.
- English prompts only; the test is about format discipline.
