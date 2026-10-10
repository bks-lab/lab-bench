# C6 method: prompt, parsing and baseline, fixed before the first arm

Written 2026-10-10, before any C6 arm ran. Design: `plan/c6.md`, inputs
`cases/c6/` (built by `bench/c6/sample.py`). `bench/c6/run_c6.py` reads the
prompt from this file (the text between the two `text` fences of the
section "Prompt"), so the file and the run cannot drift apart.

## Prompt

```text
Lies den folgenden Text und beantworte die Frage. Antworte mit der kürzesten Textstelle, die die Frage beantwortet, wörtlich und genau so aus dem Text kopiert. Steht die Antwort nicht im Text, antworte mit "keine Antwort". Gib nur ein JSON-Objekt der Form {"answer": "..."} zurück.

Text:
{context}

Frage: {question}
```

One user message, no system message. Ollama `/api/chat`, `format` =
`{"type": "object", "properties": {"answer": {"type": "string"}},
"required": ["answer"]}`, `temperature` 0, `seed` 1, `num_ctx` 8192 (the
longest passage of the test set has 11,647 characters), `num_predict` 256,
thinking off (`think: false`) for every model whose capabilities list it.
Three warm-up questions, not scored; one call at a time. The answer is the
`answer` string of the returned object; an answer that does not parse
counts as the empty string (F1 0).

## Trained reader

`deepset/gelectra-large-germanquad` at `c16378c846eb`, the transformers
question-answering pipeline with `max_answer_len` 30 and the pipeline's
other defaults (`max_seq_len` 384, `doc_stride` 128, best span only, no
"impossible" answer), one question per call on the GPU, all 2,204 test
questions (the sanity gate) and scored on the 1,000 of the sample like the
LLM arms. Transformers 5 has no extractive question-answering pipeline any
more, so this arm runs with transformers 4.57.6 put in front of the venv
(`PYTHONPATH=D:\bench-cap\tf4`, as the C1 arms with remote code).

## Lexical baseline

The passage is split into sentences at line breaks and after `.`, `!` or
`?` followed by white space. Each sentence scores the number of distinct
lower-cased word tokens (`\w+`) it shares with the question, function words
from the fixed list in `bench/c6/sample.py` left out. The first sentence
with the highest score is the answer, returned whole. Result for all 2,204
questions in `results/c6/baseline/result.jsonl`.

## Scoring

SQuAD v1.1 F1 and exact match (lower case, punctuation removed, `a`, `an`,
`the` removed, white space collapsed), maximum over the three answers.
Sensitivity: the same with `der`, `die`, `das`, `ein`, `eine` removed
instead. Span faithfulness: the answer, stripped of surrounding white
space, occurs verbatim in the passage (an empty answer and `keine Antwort`
do not).

## 2026-10-10: reader answer cap

The reader runs with `max_answer_len` 200 instead of 30, after the sanity
gate missed with 30 (cause and check in `plan/c6.md`, notes of
2026-10-10). Prompt, parsing and the LLM settings above are unchanged.
