# T12: letter-mass probes and template checks

Per model before its full route1 run (`plan/t12.md`): 20 route1 questions (`--per-group 4`), first with prefix `""`, then, if the median letter mass was below 0.5, once with `"Answer: **"`. The template check sends the first probe question once through `/api/chat` (Ollama renders its own template) and once as the runner's raw prompt, each from a cold load, and compares the prompt token counts.

| arm | prefix | questions | median mass | mass < 0.5 | first tokens (up to 4 distinct) | file |
|---|---|---|---|---|---|---|
| qwen3-4b | `""` | 20 | 0.005 | 16 | `"The"` `"A"` | `results/route1/qwen3-4b/2026-10-09-run1-probe.jsonl` |
| qwen3-4b | `"Answer: **"` | 20 | 1 | 0 | `"E"` `"A"` `"G"` `"C"` | `results/route1/qwen3-4b/2026-10-09-run1-probe-answer.jsonl` |
| qwen3-4b | template check | | | | chat 136 tokens, raw 138 (diff 2) | |
| qwen3-14b | `""` | 20 | 1 | 0 | `"A"` `"C"` `"B"` `"G"` | `results/route1/qwen3-14b/2026-10-09-run1-probe.jsonl` |
| qwen3-14b | template check | | | | chat 136 tokens, raw 138 (diff 2) | |
| qwen3-30b-a3b | `""` | 20 | 0.6066 | 9 | `"A"` `"C"` `"The"` `"B"` | `results/route1/qwen3-30b-a3b/2026-10-09-run1-probe.jsonl` |
| qwen3-30b-a3b | template check | | | | chat 136 tokens, raw 138 (diff 2) | |
| qwen3-32b | `""` | 20 | 0.24955 | 12 | `"A"` `"C"` `"The"` `"Answer"` | `results/route1/qwen3-32b/2026-10-09-run1-probe.jsonl` |
| qwen3-32b | `"Answer: **"` | 20 | 1 | 0 | `"A"` `"C"` `"B"` `"G"` | `results/route1/qwen3-32b/2026-10-09-run1-probe-answer.jsonl` |
| qwen3-32b | template check | | | | chat 136 tokens, raw 138 (diff 2) | |
| gemma4-e4b | `""` | 20 | 0.0001 | 20 | `"The"` | `results/route1/gemma4-e4b/2026-10-09-run1-probe.jsonl` |
| gemma4-e4b | `"Answer: **"` | 20 | 0.99975 | 0 | `"A"` `"C"` `"B"` `"G"` | `results/route1/gemma4-e4b/2026-10-09-run1-probe-answer.jsonl` |
| gemma4-e4b | template check | | | | chat 137 tokens, raw 139 (diff 2) | |
| granite4-micro | `""` | 20 | 0.9776 | 0 | `"E"` `"C"` `"G"` `"A"` | `results/route1/granite4-micro/2026-10-09-run1-probe.jsonl` |
| granite4-micro | template check | | | | chat 135 tokens, raw 135 (diff 0) | |
| phi4-14b | `""` | 20 | 0.7460500000000001 | 5 | `"The"` `"C"` `"A"` `"E"` | `results/route1/phi4-14b/2026-10-09-run1-probe.jsonl` |
| phi4-14b | template check | | | | chat 138 tokens, raw 138 (diff 0) | |
| gemma4-12b | `""` | 20 | 0.99905 | 0 | `"A"` `"C"` `"G"` `"E"` | `results/route1/gemma4-12b/2026-10-09-run1-probe.jsonl` |
| gemma4-12b | template check | | | | chat 137 tokens, raw 139 (diff 2) | |
| mistral-small3.2-24b | `""` | 20 | 0.69145 | 8 | `"A"` `"C"` `"The"` `"B"` | `results/route1/mistral-small3.2-24b/2026-10-09-run1-probe.jsonl` |
| mistral-small3.2-24b | template check | | | | chat 128 tokens, raw 128 (diff 0) | |
| qwen3.8-27b | `""` | 20 | 0.9852000000000001 | 0 | `"A"` `"C"` `"B"` `"G"` | `results/route1/qwen3.8-27b/2026-10-09-run1-probe.jsonl` |
| qwen3.8-27b | template check | | | | chat 137 tokens, raw 139 (diff 2) | |
| shisa-de-1 | `""` | 20 | 0.99785 | 0 | `"A"` `"C"` `"B"` `"G"` | `results/route1/shisa-de-1/2026-10-09-run1-probe.jsonl` |
| shisa-de-1 | template check | | | | chat 137 tokens, raw 139 (diff 2) | |

