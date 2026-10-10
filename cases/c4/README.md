# C4 inputs (frozen 2026-10-10)

Built by `bench/c4/build_c4.py`; hashes in `manifest.json`; design in
`plan/c4.md`, method in `results/c4/method.md`.

- `bfcl.jsonl`: the 1,000 non-live AST items of the Berkeley Function
  Calling Leaderboard (gorilla repository, commit `6ea57973c7a6`, folder
  `berkeley-function-call-leaderboard`), with the exact request of both
  modes (`fc`: user turn and tool list; `prompt`: BFCL's prompt-mode system
  prompt and user turn), built with BFCL's own code.
- `bfcl-answers.jsonl`: the matching `possible_answer` entries.
- `schemas.jsonl`: 600 JSON schemas, 100 from each of six test subsets of
  JSONSchemaBench (`epfl-dlab/JSONSchemaBench`, revision `5bd0f4640bad`).

Licences: BFCL is Apache-2.0 (repository licence of
ShishirPatil/gorilla); JSONSchemaBench is MIT (dataset card). Both are
redistributed here unchanged in content, reformatted as JSON lines.
