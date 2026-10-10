# C1 and C2 runners (capability family, plan/c1.md, plan/c2.md)

One arm per call, started by the PC job queue from the templates in
`bench/queue/jobs/c1/` and `bench/queue/jobs/c2/` (file name prefix = run
order, the sanity arm first; every other arm lists it in `after_ok`).

| file | role |
|---|---|
| `bench/c1/run_c1.py` | embedders, BM25 baseline, rerankers (`--base-run`) |
| `bench/c2/run_c2.py` | faster-whisper arms and NeMo arms |
| `bench/c1/capcommon.py` | timed sections, bootstrap, versions (shared) |
| `bench/lib/norm_de.py` | the fixed C2 text normaliser |
| `bench/c1/prefetch.py`, `bench/c2/prefetch.py` | pinned downloads into `HF_HOME`, CTranslate2 conversion |
| `cases/c1/fetch.py`, `cases/c2/fetch.py` | data download and hash check against the committed manifests |

Each run writes `result.jsonl` (one row per query or utterance) and
`meta.json` (versions, arm config, summary with 95 % intervals, timed
sections as local ISO timestamps for the queue's 1 s nvidia-smi CSV, gate
result). Exit 3 = sanity gate failed, 1 = error.

## PC layout (the GPU workstation)

- data: `D:\bench-data\c1\{miracl,germandpr}\*.parquet`,
  `D:\bench-data\c2\test.tsv`, `audio\test\*.wav`, `models\*-ct2`
- models: `HF_HOME=D:\hf`, jobs run with `HF_HUB_OFFLINE=1`
- `D:\bench-cap\.venv`: torch 2.11.0+cu128, transformers 5.19,
  sentence-transformers 6.1, faster-whisper 1.2.1, docling
- `D:\bench-cap\tf4`: transformers 4.57.6 + sentence-transformers 5.1.2,
  put in front via `PYTHONPATH` for `gte-multi` and `nomic-v2-moe`, whose
  remote code does not run on transformers 5
- `D:\bench-cap\.venv-nemo`: NeMo 3.0.0, same torch

pip on that PC: `set PIP_CONFIG_FILE=nul` (lower case; pip compares with
`os.devnull` case-sensitively, so `NUL` still loads the global NVIDIA
index) and `-c D:\bench-cap\constraints-torch.txt` so no package swaps the
CUDA torch for the CPU build from PyPI.
