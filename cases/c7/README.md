# C7 inputs (frozen 2026-10-10)

`trials.jsonl`: the model-independent part of the 540 long-context
prompts (names, seven-digit codes, needle depths, passage order), written
by `bench/gen-c7.py`; `manifest.json` holds the prompt, the needle sentence
and the hash. The haystack text itself is built at run time from the
GermanQuAD test passages (deepset, CC BY 4.0) and cut to each model's
token count by `bench/c7/run_c7.py`. Project names are invented.
