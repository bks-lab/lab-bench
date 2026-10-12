# C5 inputs (frozen 2026-10-12)

FLORES+ (`openlanguagedata/flores_plus`, devtest, `deu_Latn` and
`eng_Latn`, revision `e707e62e762a`, CC BY-SA 4.0) is gated, and its texts
are not committed: `manifest.json` (written by `bench/c5/freeze.py`) holds
the two file hashes, the 1,012 ids and one SHA-256 per sentence and
language. To reproduce, accept the dataset terms on Hugging Face, download
the two files into `.cache/c5/` and check them with `freeze.py` (it must
reproduce the manifest byte for byte). Design: `plan/c5.md`.
