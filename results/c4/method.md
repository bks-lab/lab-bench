# C4 method: requests and parsing, fixed before the first arm

Written 2026-10-10, before any C4 arm ran. Design: `plan/c4.md`, inputs
`cases/c4/` (built by `bench/c4/build_c4.py`). Every arm gets the same
requests, with no per-model change and no few-shot example.
`bench/c4/run_c4.py` reads the Part B instruction from this file (the text
between the two `text` fences of the section "Part B instruction"), so the
file and the run cannot drift apart.

## Part A: BFCL non-live, two modes

The 1,000 items of `simple_python` (400), `multiple`, `parallel` and
`parallel_multiple` (200 each), at gorilla commit `6ea57973c7a6`, loaded
with BFCL's own loader (it appends "Note that the provided function is in
Python 3 syntax." to every function description, as on the leaderboard).

- `fc`: the item's user turn as the only message, and the item's functions
  as Ollama's native `tools` field, converted by BFCL's `convert_to_tool` in
  the OpenAI completions style. That style replaces a dot in a function name
  by `_` (`math.factorial` becomes `math_factorial`), as the leaderboard ran
  its "Qwen3-14B (FC)" row; the checker maps the names back
  (`underscore_to_dot`). The answer is the list of tool calls Ollama parsed
  from the model's output. An answer without tool calls is a failure to
  decode, as on the leaderboard.
- `prompt`: BFCL's prompt-mode system prompt (functions as JSON, answer as
  `[func(a=1), ...]`) built by BFCL's `system_prompt_pre_processing_chat_model`,
  then the user turn. The answer text is parsed by BFCL's
  `default_decode_ast_prompting`.

Scoring: BFCL's `ast_checker` through its own `_evaluate_single_ast_entry`
(`bench/c4/check_ast.py`, upstream code imported, nothing reimplemented).

## Part B instruction

```text
Generate one JSON object that is valid against the JSON Schema below. Fill every field you include with a plausible example value. Answer with the JSON object only, without explanation and without Markdown.

JSON Schema:
{schema}
```

`{schema}` is the schema of the row, re-serialised compactly
(`json.dumps(..., separators=(",", ":"))`, same content, no indentation) to
keep long schemas inside the context. One user message, no system message.

- `free`: the instruction only. The answer is the first JSON value in the
  text: the text itself if it parses, else the content of a Markdown code
  fence if there is one, else the span from the first `{` to the last `}`.
- `constrained`: the same message plus Ollama `format` = the row's schema
  as committed (not the compact copy, the same object). A request Ollama
  refuses because of the schema is "not constrainable" and counts towards
  the coverage number, not the validity of `constrained`.

Validity: the parsed value validates against the schema with `jsonschema`
(the validator class of the schema's `$schema` draft, Draft 2020-12 when it
names none), with format checking off (formats are annotations by default
in Draft 2020-12, and JSONSchemaBench does not check them either). A schema
that is itself invalid for its draft is reported and left out of both modes.

## Options, every call

Ollama 0.35.0 `/api/chat`, `temperature` 0, `seed` 1, `num_ctx` 8192,
`num_predict` 1024 (Part A) or 4096 (Part B), `stream` false. Thinking off
(`think: false`) for every model whose Ollama capabilities list thinking;
gpt-oss gets `think: "low"`, its lowest setting. Three warm-up items per
mode, not scored. One call at a time.
