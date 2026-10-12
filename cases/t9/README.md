# T9 inputs (frozen 2026-10-12)

`requests.jsonl`: 1,000 MASSIVE de requests of route1, each in four
variants (`clean`, `typo`, `translit`, `colloquial`), 4,000 requests with
the questions and options of route1 and two extra fields, `base_line` (the
route1 line) and `variant`. Written by `bench/t9/perturb.py` (rules, seeds,
lists in its docstring); sample lines, hashes and the number of utterances
each variant changed in `manifest.json`; design in `plan/t9.md`.

Source: MASSIVE 1.1 (Amazon, CC BY 4.0), see `cases/route1/README.md` for
attribution and the SLURP notice. The perturbed texts are derivatives under
the same licence.
