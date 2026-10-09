# Speed and cost per line, pv1 de, run 1 (2026-10-01)

Computed 2026-10-02 from `results/pv1/<arm>/2026-10-01-run1*.jsonl`, German requirement lines (136), median per line.

- **Jev cost:** median input tokens per line x list price USD 0.042 per million input tokens, output free (docs.typesafe.ai/models, read 2026-10-02). Measured tokens, published price.
- **Local cost:** electricity only, ESTIMATED: GPU busy time per line (sum of the five question latencies, measured) x 450 W (RTX 4090 power limit, an upper bound; power draw was NOT measured) x EUR 0.35 per kWh (assumed). Hardware, cooling and operation are not included. Runs were sequential, one question at a time; batching could lower time and energy per line.

| arm | where | lines | time per line (s) | API cost per 1,000 lines (USD) | electricity per 1,000 lines (EUR, upper bound) |
|---|---|---|---|---|---|
| jev | TypeSafe API, jev-1.13.0 | 136 | 0.25 | 0.074 | n/a |
| winnow-12b | Winnow-12B Q8_0, RTX 4090 | 136 | 2.49 | n/a | 0.109 |
| qwen3.8-27b | qwen3.8:27b, RTX 4090 | 136 | 1.76 | n/a | 0.077 |
| shisa-de-1 | Shisa DE-1 Q4_K_M, RTX 4090 | 136 | 2.35 | n/a | 0.103 |
| qwen3-32b | qwen3:32b, RTX 4090 | 136 | 1.65 | n/a | 0.072 |

Reading: Jev is about seven to ten times faster per line. On cost, Jev at list price (USD 0.07 per 1,000 lines) is in the same range as the electricity alone of a local run at the power limit (EUR 0.07 to 0.11 per 1,000 lines). Hardware, which is not counted here, makes the local run clearly more expensive per line at this volume. A local run pays off where data must not leave the house or the volume carries the hardware; that is not measured here.
