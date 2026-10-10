# C6 inputs (frozen 2026-10-10)

`sample.jsonl`: 1,000 questions of the GermanQuAD test split with passage
and the three answers, drawn by `bench/c6/sample.py` (rule and ids in
`manifest.json`, design in `plan/c6.md`).

Source: GermanQuAD by deepset (Möller, Risch, Pietsch 2021),
`deepset/germanquad`, Hugging Face parquet conversion
`refs/convert/parquet` at `a2f3a59f0be8`, licence CC BY 4.0. The passages
are from German Wikipedia. Rows are reproduced unchanged; only the fields
were renamed to plain JSON.
