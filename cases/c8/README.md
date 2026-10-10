# C8 inputs (frozen 2026-10-10)

Written by `bench/gen-c8.py`; design in `plan/c8.md`.

- `chat.jsonl`: 68 `chat-short` and 68 `chat-long` prompts (4 warm-up,
  64 timed): the passage order of each prompt. The text is built at run
  time from the GermanQuAD test passages (deepset, CC BY 4.0) and cut to
  512 or 4,096 tokens of the model at hand.
- `decide.jsonl`: 500 timed and 4 warm-up one-token decision prompts from
  the MASSIVE de requests of `requests/route1/requests.jsonl` (MASSIVE,
  Amazon, CC BY 4.0), rendered for each chat template a C8 arm uses.
