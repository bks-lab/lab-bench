# C1 findings: German retrieval on one RTX 4090

Design `plan/c1.md` (committed 2026-10-09 before any run), shared rules
`plan/c0-common.md`. Runs: 2026-10-09, PC job queue, one job at a time.
Scores: `results/c1/score.md`, chart data: `results/c1/summary.json`,
capability row: `results/capabilities.json`. Independent recompute of the
headline numbers from the raw rows and the qrels: `bench/check-cap.py`
(16 C1 checks agree). Notes after the run: the dated section of
2026-10-10 at the end of `plan/c1.md`.

## Sanity gate

Both gates passed. `bge-m3` on MIRACL de: nDCG@10 0.5767 against MTEB's
0.5759 (tolerance 0.015), Recall@100 0.963 against 0.9626. `e5-large` on
GermanDPR: 0.8284 against 0.8290. The rewritten MIRACL revision
(`332a9acb49f5`, preparation note in `plan/c1.md`) reproduces the
published number.

## Result

| arm | params | nDCG@10 MIRACL de | 95 % CI | GermanDPR nDCG@10 | query ms | index passages/s | peak VRAM GB |
|---|---|---|---|---|---|---|---|
| qwen3-emb-8b | 7.6 B | **0.613** | 0.576 to 0.651 | 0.841 | 53.7 | 44 | 21.8 |
| nomic-v2-moe | 475 M | 0.582 | 0.543 to 0.620 | 0.816 | 28.2 | 840 | 2.6 |
| bge-m3 | 568 M | 0.577 | 0.540 to 0.616 | 0.826 | 15.1 | 1,189 | 3.0 |
| e5-large | 560 M | 0.564 | 0.527 to 0.603 | 0.828 | 12.5 | 1,194 | 3.0 |
| qwen3-emb-0.6b | 596 M | 0.546 | 0.510 to 0.583 | 0.795 | 39.5 | 335 | 9.3 |
| gte-multi | 305 M | 0.511 | 0.473 to 0.552 | 0.797 | 8.8 | 2,377 | 2.4 |
| e5-large-instruct | 560 M | 0.510 | 0.474 to 0.550 | 0.805 | 13.4 | 1,204 | 3.0 |
| bm25 (baseline, CPU) | none | 0.199 | 0.168 to 0.232 | 0.511 | 0.3 | 18,525 | 0 |

- **Best embedder: Qwen3-Embedding-8B**, nDCG@10 0.613, skill 0.52 over
  BM25. The paired bootstrap separates it from every other arm
  (nomic-v2-moe is 0.031 lower, p 0.016; bge-m3 0.036 lower, p 0.006).
- **The price is time and memory.** Qwen3-Embedding-8B indexes 44 passages
  per second against about 1,200 for bge-m3 and e5-large: the 71,277
  MIRACL passages take 27 minutes instead of one, at 10,100 GPU joules per
  1,000 passages instead of about 360. The torch peak fills 21.8 GB of the
  card at batch 64. A nightly re-index of a few hundred thousand passages
  still fits in a night on one card; a few million do not.
- **Among the small models no order is shown.** nomic-v2-moe, bge-m3 and
  e5-large lie within 0.02 of each other with overlapping intervals; each
  is separated from Qwen3-Embedding-8B (the plan tests every arm against
  the best only). gte-multi is the fastest neural arm (2,377 passages/s,
  8.8 ms per query) at 0.511.
- **Instructions did not help here.** multilingual-e5-large-instruct with
  the web search instruction of its card scores 0.510, below plain
  e5-large (0.564), and Qwen3-Embedding-0.6B (0.546) stays below the 2022
  encoders. GermanDPR shows the same top (0.841) and compresses the rest
  into 0.795 to 0.828; Recall@100 there is 0.99 to 1.0 for every neural
  arm, as the plan expected, and is not interpreted.
- **BM25 is weak on this set by construction.** The hard-negatives version
  of MIRACL keeps per query the passages that retrieval systems ranked
  high, so many negatives share the query's words. 0.199 is a floor for
  this corpus, not a statement about BM25 on company documents.

## Rerankers on top of qwen3-emb-8b (2026-10-10)

Both rerankers rescore the top 100 of `qwen3-emb-8b`, the best embedder,
chosen after the fact as the plan says. Each row is a pipeline: the
embedder indexes and retrieves, the reranker reorders. Recall@100 is the
embedder's (0.980) by construction.

| pipeline | reranker params | nDCG@10 MIRACL de | 95 % CI | GermanDPR nDCG@10 | rerank ms per query | query ms total | reranker job peak VRAM GB |
|---|---|---|---|---|---|---|---|
| qwen3-emb-8b + bge-reranker-v2-m3 | 568 M | **0.640** | 0.605 to 0.677 | 0.879 | 155 | 209 | 2.2 |
| qwen3-emb-8b + Qwen3-Reranker-0.6B | 596 M | 0.634 | 0.600 to 0.667 | 0.858 | 1,022 | 1,078 | 16.2 |
| qwen3-emb-8b alone | | 0.613 | 0.576 to 0.651 | 0.841 | | 54 | |

- **bge-reranker-v2-m3 adds 0.028 nDCG@10** over the embedder alone, and
  the paired bootstrap separates the two (p 0.036). On GermanDPR it adds
  0.038. It costs 155 ms per query for 100 pairs at 512 tokens and 2.2 GB.
- **Qwen3-Reranker-0.6B is not separated from bge-reranker-v2-m3**
  (0.006 lower, p 0.60) but takes 6.6 times as long per query and 16 GB,
  because it scores each pair as a causal language model over the full
  prompt (max length 8,192 tokens, the model card's; `plan/c1.md` fixes
  512 tokens for the embedders only).
- **The first Qwen3-Reranker run did not finish.** Job
  `20261010-071230-c1-+qwen3-rerank` hit its 60 min queue timeout with the
  card full (24.0 GB) and no output. Cause: the forward call computed the
  language-model head over every position of every pair (151k vocabulary)
  and then kept only the last position. `bench/c1/run_c1.py` now asks for
  the last position only (`logits_to_keep=1`); the scores are the same
  (checked on CPU, differences below 2e-10). The rerun
  `20261010-101637-c1-+qwen3-rerank` with that fix, same model, revision,
  prompt, length and base run, finished in 41 GPU minutes; that is the run
  scored here.

## Capability map

Best local result: the pipeline **qwen3-emb-8b + bge-reranker-v2-m3**
(both Apache-2.0, open source), 0.640 [0.605, 0.677] against BM25 0.199,
209 ms per query at batch 1 (54 ms retrieval plus 155 ms reranking),
21.8 GB peak (the embedder's indexing job; the reranker needs 2.2 GB and
runs after it), 10,091 J per 1,000 passages indexed. The paired test does
not separate it from the Qwen3-Reranker pipeline, which has the same peak,
so the row states the tie and keeps bge-reranker-v2-m3. For a stock that
is large or changes daily, bge-m3 or e5-large (3 GB, about 27 times faster
indexing, 0.06 to 0.08 lower) is the practical choice; a reranker on top of
one of them was not measured.

## Limits

- One run per arm; indexing throughput was measured once, not three times
  as `plan/c0-common.md` asks for throughput numbers.
- Passages are cut at 512 tokens; 537 to 1,295 MIRACL passages per arm were
  truncated (counts per arm in `score.md`). The Qwen3 tokeniser cuts more.
- Exposure: bge-m3 and multilingual-e5 list MIRACL train in their
  fine-tuning data; the dev queries are not in train, but the corpus is the
  same German Wikipedia. Qwen3-Embedding's training data is not itemised.
- Rerankers were run on top of `qwen3-emb-8b` only, the best embedder
  chosen after the fact as the plan says.
