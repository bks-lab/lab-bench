# C9 findings: personal data detection in German text

Design `plan/c9.md` (committed 2026-10-09 before any arm, preparation notes
and a results note of 2026-10-12), runner `bench/c9/run_c9.py`, scorer
`bench/c9/score_c9.py`, shared rules `plan/c0-common.md`. Runs: 2026-10-12 on
the workstation queue (one RTX 4090, Ollama 0.35.0 for the two LLM arms).
Data: the 476 German rows of the Gretel synthetic PII finance test split
(Apache-2.0). Both the texts and the span labels are LLM output. The
documents are not in this repository (push protection flagged invented keys
in them), nor are the result rows; the scores (`results/c9/score.json`,
`results/c9/score.md`), the meta and queue status per arm and the gold audit
(`results/c9/gold-audit.json`) are. Capability row: `results/capabilities.json`.

## Read this first: the gold audit was done by a language model

`plan/c9.md` asks for a human to audit 40 seeded documents. That did not
happen. The maintainer had no time, so the audit was done by a language
model: Claude Haiku 5.5 (`claude-haiku-5-5`) judged every gold span and
listed missed spans, and Claude Opus 5.5 spot-checked every verdict that was
not "ok". No human looked at the 40 documents for this report. Haiku 5.5 is
not one of the seven C9 arms. The verdicts may be wrong in both directions. A human audit of the same 40
documents is still open.

Result of the audit: 149 gold spans, 102 ok, 43 wrong, 4 boundary fixes, 2
missed spans added. The wrong ones are the generator's own mislabels: generic
words such as "Damen", "Herren", "Jetzt", "Pflichtigen", "Ihr Exemplar"
tagged as person, and company addresses. Gold precision is 0.685 (0.711 if
the four boundary fixes count as right), gold recall 0.981. The design says
that below 0.90 the test is "gold noisy": all numbers are descriptive and
there is no ranking claim. The sections below follow that rule.

## Result

Span F1 (exact boundaries, type-agnostic, boundary rule of `bench/c9/c9lib.py`).
Headline: the audited gold of the 40 documents; beside it the original gold of
all 476 documents. Intervals are 95 % bootstrap over documents (1,000
resamples, seed 20261009); the paired difference is to the best arm on the
same documents.

| arm | F1 audited gold, 40 docs | paired diff to best | F1 original gold, 476 docs | recall (orig.) | precision (orig.) |
|---|---|---|---|---|---|
| gemma4-12b | **0.839** [0.706, 0.929] | best | 0.627 [0.593, 0.660] | 0.628 | 0.626 |
| qwen3.8-27b | 0.828 [0.738, 0.902] | 0.011 [-0.115, 0.112] | **0.644** [0.616, 0.672] | 0.641 | 0.647 |
| gliner-pii-domains | 0.739 [0.643, 0.816] | 0.100 [-0.050, 0.239] | 0.564 [0.532, 0.595] | 0.432 | 0.813 |
| gliner2.5-multi | 0.649 [0.517, 0.742] | 0.190 [0.047, 0.345] | 0.515 [0.478, 0.550] | 0.403 | 0.714 |
| gliner-multi-pii | 0.444 [0.282, 0.596] | 0.394 [0.199, 0.589] | 0.427 [0.394, 0.460] | 0.319 | 0.644 |
| presidio | 0.300 [0.213, 0.385] | 0.538 [0.392, 0.667] | 0.271 [0.249, 0.295] | 0.316 | 0.237 |
| regex | 0.150 [0.025, 0.274] | 0.689 [0.528, 0.836] | 0.135 [0.099, 0.177] | 0.075 | 0.728 |

On the original gold the best arm is qwen3.8-27b and gemma4-12b is 0.017
behind ([-0.004, 0.042]); on the audited gold the order of the two is the
other way round and the difference is 0.011 ([-0.115, 0.112]). Neither is
separated. Both golds agree on the groups: the two LLMs first, then the three
GLiNER models (gliner-pii-domains, gliner2.5-multi, gliner-multi-pii in that
order), Presidio and the regex baseline last. Given the noisy gold this is a
description of one run, not a ranking claim. The capability row shows
gemma4-12b by the roadmap rule (not separated from the best, less peak VRAM).

Typed strict F1 (original gold, descriptive) and the machine table are in
`results/c9/score.md`. In short: e-mail is nearly solved by regex and
Presidio (0.98) and by the LLMs (0.92 to 0.94); person and id_number are where
the LLMs lead (person 0.64 against at most 0.56 for the others, id_number
0.59 and 0.70 against at most 0.32); IBAN and account numbers stay at 0.06 to 0.33 for every arm.

## Reading the numbers

- **The audited F1 is higher than the original F1 for every model arm**, as
  expected: the audit removes gold spans that were not personal data, so an
  arm that did not predict them no longer loses recall. Precision of the LLM
  arms stays at 0.74 on the audited gold; they tag more than the audit keeps.
- **Forty documents are a small sample.** The audited intervals are 0.15 to
  0.25 wide and the paired test separates only gliner2.5-multi, gliner-multi-pii,
  Presidio and regex from the best arm; gliner-pii-domains is not separated
  from it there, while on 476 documents it is (0.080 [0.050, 0.114]).
- **Presidio is a recall tool with a precision problem**: recall 0.32 and
  precision 0.24 on the original gold. The regex baseline is precise (0.73)
  and finds 7.5 % of the gold spans; the sanity gate (e-mail recall at least
  0.95) passed with 0.965.
- **Machine.** Median ms per document: regex 0.1, gliner-pii-domains 27.7,
  gliner2.5-multi 21.9, gliner-multi-pii 27.9, Presidio 36.8 (CPU),
  gemma4-12b 517, qwen3.8-27b 1,142. GPU energy per 1,000 documents: GLiNER
  4.5 to 5.0 kJ, gemma4-12b 158 kJ, qwen3.8-27b 397 kJ. Peak VRAM over idle:
  GLiNER 2.0 GB (gliner2.5-multi 7.6 GB), gemma4-12b 9.2 GB, qwen3.8-27b
  18.1 GB.
- **Exposure.** The Gretel set is public since 2024 and may be in the
  training data of the LLM arms and of the GLiNER fine-tunes (plan/c9.md).
- **Not claimed.** No statement about real German documents: the texts are
  synthetic, finance domain, and the gold is model output. No DSGVO
  compliance claim of any kind.
