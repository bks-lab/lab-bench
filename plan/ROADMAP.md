# Test bench roadmap

Written 2026-10-09, before any of the tests below ran. A test moves from
"planned" to "measured" only through a dated design file in `plan/` that is
committed before its first run (see `plan/einv1.md` for the form).

## Principle: every tool family gets its home ground

A comparison that only asks what one tool is built for flatters that tool.
So every family in the tool map gets at least one test that plays to its
declared strength, named here in advance, next to the tests where it is a
guest. A family that loses on its own home ground is reported as such.

| Family | Declared strength | Home-ground test (status) |
|---|---|---|
| Jev | typed judgments with probabilities over a state | pv1 match (measured), route1 (measured), T1 automation rate (planned), T6 e-invoice yes/no (planned) |
| GLiNER Decide | fast classification into given labels | fast-decisions (measured), T4 trained on own labels (planned), T10 batched throughput (planned) |
| GLiNER extraction and PII | finding spans in text | ex1 slots (measured), T7 personal data (planned) |
| Local LLM | general purpose, runs in house, structured output | ex1 slots (measured), T12 local model series on route1 and ex1 (planned), T8 invoice fields (planned), T11 long documents (planned) |
| Classic trained classifier (new) | cheap and strong when labelled data exists | T3 (planned) |
| Hosted LLM | general purpose, no setup | T5 dropped 2026-10-09, see below |

## Planned tests

| # | Test | Adds | Data | Effort |
|---|---|---|---|---|
| T1 | Automation rate: share of cases a tool can decide alone at 95 % accuracy, and calibration (ECE, Brier) of its probabilities | row | existing route1 and pv1 rows, no new runs | small |
| T2 | Stability: second run with shuffled options, flip rate | row | route1 | small |
| T3 | Classic baseline: multilingual sentence embeddings plus logistic regression trained on the MASSIVE train split | column | MASSIVE train/test | small |
| T4 | GLiNER Decide trained with 10, 50 and all examples per class | variant | MASSIVE train | medium |
| T5 | Hosted LLM on route1 and ex1 (not on pv1, where a Claude model is the judge) | column | route1, ex1 | small |
| T6 | E-invoice yes/no over free text | row | KoSIT, see `plan/einv1.md` | medium |
| T7 | Personal data detection in German text | row | public PII set, licence to check | medium |
| T8 | Invoice fields from rendered KoSIT invoices, XML fields as gold | row | KoSIT (Apache-2.0) | medium |
| T9 | Robustness: typos and colloquial wording in MASSIVE de | row | MASSIVE de, perturbed | small |
| T10 | Throughput with batching | row | route1 | small |
| T11 | Long documents: classify texts beyond the encoder context | row | to choose | medium |
| T12 | Local open model series: eleven open-weight models in four size classes on route1 and ex1, one workstation (RTX 4090, 24 GB), with latency, VRAM and throughput; see `plan/t12.md` | size-class columns | route1, ex1 | large (machine hours) |

Order (as written before the change below): T1, T3 and T5 first (cheap, they close the largest gaps), then T4 and
T2, then T6 to T11.

## Capabilities of a local workstation (no Jev)

Added 2026-10-09 on Michael Boiman's request: tests that do not compare
against Jev, but ask what one in-house PC with open models can do on German
business tasks. The machine is the PC with the RTX 4090 (24 GB) that ran the
local arms. Shared rules for all of them, written before any run:
`plan/c0-common.md` (machine record, 95 % bootstrap intervals with 1,000
seeded resamples, a sanity gate against a published number before other
models are scored, licence class per model, training data exposure per
dataset). One dated design per test:

| # | Capability | Data (licence) | Primary metric | Sanity reference | Design | GPU h |
|---|---|---|---|---|---|---|
| C1 | German retrieval for RAG: embedders, then rerankers | MIRACL de hard negatives (CC BY-SA 4.0), GermanDPR (CC BY 4.0) | nDCG@10 | MTEB: bge-m3 0.5759 on MIRACL de | `plan/c1.md` | 1.5 |
| C2 | German speech to text | FLEURS de_de test (CC BY 4.0) | WER | canary-1b-v2 card: 4.40 | `plan/c2.md` | 2 |
| C3 | Invoice fields from rendered e-invoices (also T8) | KoSIT test suite + KoSIT visualisation (Apache-2.0) | field exact match | internal: every gold value is in the rendered PDF | `plan/c3.md` | 3 |
| C4 | Structured output and tool calling | BFCL v3 non-live (Apache-2.0), JSONSchemaBench (MIT) | AST accuracy, schema validity | BFCL: Qwen3-14B (FC) simple 95.25 % | `plan/c4.md` | 3 |
| C5 | Translation de to en and en to de | FLORES+ devtest (CC BY-SA 4.0, gated) | chrF++ | OPUS-MT leaderboard: opus-mt-de-en chrF 66.70 | `plan/c5.md` | 3 |
| C6 | German reading comprehension | GermanQuAD test (CC BY 4.0) | F1 | GELECTRA-large F1 88.1 | `plan/c6.md` | 2.5 |
| C7 | Long context, 4k to 32k | synthetic needles in GermanQuAD text | accuracy per length | internal: 95 % single needle at 4k | `plan/c7.md` | 4 |
| C8 | Throughput and energy per model size | synthetic prompts, route1 decisions | tokens/s, decisions/h, J per 1k tokens | internal: three repeats within 5 % | `plan/c8.md` | 3 (+1 vLLM) |
| C9 | Personal data detection (DSGVO) (also T7) | Gretel synthetic PII finance, German rows (Apache-2.0) | span F1, recall | internal: regex e-mail recall 0.95 | `plan/c9.md` | 1.5 |

T7 and T8 above are carried out as C9 and C3. Jev has no span or field
primitive, so those two were never Jev comparisons; their rows in the tool
map take the C results.

### Order of value and the run order on the PC

Ranked by value for BKS customers first, effort second:

1. C1 retrieval: every in-house AI project starts with search over own
   documents; small effort.
2. C2 speech to text: meeting minutes are the most asked-for first use, and
   BKS-Lab already transcribes locally; small effort.
3. C3 invoice fields: the paperwork every company has; medium effort
   (renderer setup, CPU only, done beforehand on a Mac).
4. C9 personal data: the DSGVO precondition for most other uses; medium
   effort (gold audit by a person).
5. C4 structured output: what makes a model integrable; small effort.
6. C8 throughput and energy: sizing for every offer; small effort.
7. C5 translation; small effort.
8. C6 reading comprehension (the answer step after C1); small effort.
9. C7 long context; medium effort, longest GPU time.

Run order on the PC after T12 has finished: C1, C2, C3, C9, C4, C8, C5, C6,
C7. All preparation without a GPU (downloads, KoSIT rendering, sampling,
freezing of `cases/`, the C7 and C8 generators, the C9 gold audit) is done
before the first GPU job. Estimated GPU time: 23.5 h for the nine tests,
about 28 h with sanity reruns, plus 1 h if the vLLM arm of C8 runs. That is
three to four unattended nights.

### Licence picture

Every model chosen for C1 to C9 is `open source` (Apache-2.0 or MIT) or
`open source (CC BY)` (the two NVIDIA speech models and opus-mt-en-de).
Gemma 4 is Apache-2.0 (Google's Gemma 4 licence page, read 2026-10-09),
unlike Gemma 3, whose terms restrict use. Dropped for licence reasons:
jina-embeddings-v3 and jina-reranker-v2 (CC BY-NC 4.0), NLLB-200 and
TowerInstruct (CC BY-NC 4.0), XCOMET (CC BY-NC-SA 4.0), Llama 3.2 Vision
(multimodal rights not granted to companies based in the EU),
nvidia/gliner-PII (NVIDIA Open Model Licence, English only),
embeddinggemma-300m (Gemma terms, gated; Qwen3-Embedding-0.6B covers the
size class). The capability map still has a licence class column, so a
restricted model added later is visible as such.

### Capability map for the website

A future page on bks-lab.com shows one row per capability, filled from
`results/capabilities.json`, which the C scorers write. Columns, fixed now:

| column | content |
|---|---|
| capability | C1 to C9 name |
| best local model | highest primary metric; if its interval overlaps the next model's and no paired test separates them, the one with less peak VRAM is shown and the tie is stated |
| licence class | as in `plan/c0-common.md` |
| quality | primary metric with 95 % interval, and the baseline or reference next to it |
| speed | the C file's speed unit (ms per query, real-time factor, s per invoice, tokens/s) |
| peak VRAM | GB, from nvidia-smi |
| energy | GPU joules per unit of work |
| fits on one 24 GB card | yes, or `offloaded` |

A second, smaller table per capability lists the smallest model within the
interval of the best one, because that is often the better recommendation
for a customer's hardware.

### PC job queue (specified, not built)

GPU jobs from several agents must never overlap and should run unattended.
One runner on the PC processes job files in order; agents only enqueue.

- Folder `D:\jmb-queue\` with `incoming\`, `running\`, `done\`, `failed\`,
  `logs\` and a `PAUSE` file switch.
- A job is one JSON file `incoming\<yyyymmdd-hhmmss>-<test>-<arm>.json`:
  `id`, `test` (C1 to C9, T-numbers), `repo_ref` (commit of the bench repository,
  bks-lab/lab-bench since 2026-10-10, to run), `cwd`, `cmd` (argument list, no shell string), `env`,
  `timeout_min`, `expected_gpu_min`, `outputs` (paths the job must create),
  `requested_by`. Agents copy the file in over SSH and do nothing else on
  the PC.
- One runner process, started at logon as a scheduled task, holds a lock
  file (`runner.lock` with PID) so a second runner exits at once. It takes
  the oldest file by name, moves it to `running\`, checks that no other
  compute process is on the GPU (`nvidia-smi --query-compute-apps`), checks
  out `repo_ref` in a clean worktree, starts the 1 s nvidia-smi sampler
  (`c0-common.md`), runs `cmd` with stdout and stderr to
  `logs\<id>.log`, stops the sampler, verifies `outputs` exist, and moves
  the job to `done\` or `failed\` with a `<id>.status.json` (start, end,
  exit code, GPU minutes, peak VRAM, energy, machine record).
- `PAUSE` present: the runner finishes the current job and waits. A job
  over `timeout_min` is killed and marked failed. The runner never pushes
  to git; agents fetch results and logs over SSH and commit them in a PR.
- Ollama-based jobs restart the Ollama service with the job's `env`
  (`OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, KV cache type) and
  restore the default afterwards.
- A status command (`queue status`) prints the running job, the queue with
  expected GPU minutes, and the last ten results.

Building it is a separate task; the C runs wait for it or are started by
hand one at a time until it exists.

## Change 2026-10-09: T5 dropped, focus on local models

On Michael Boiman's direction of 2026-10-09 the bench focuses on local
open-weight models and on what one workstation can do. T5 (hosted LLM) is
dropped and not run; the row stays in the table above for the record. T12
takes its place in the order, directly after T1 and T3.

Home ground of the local family, restated for that focus:

| Local model test | What it plays to | Status |
|---|---|---|
| ex1 slots | structured output (JSON) from free text, no training | measured (Winnow-12B), T12 adds the series |
| route1 | zero-shot classification with given labels, in house | measured (Winnow-12B), T12 adds the series |
| T12 | the whole size range on one 24 GB card: accuracy against latency, VRAM and throughput | planned, `plan/t12.md` |
| T8 (as C3) | invoice fields from rendered documents | planned, `plan/c3.md` |
| T11 | long documents beyond an encoder's context | planned |

A local model that loses on these tests loses on its own ground and is
reported as such.

## Change 2026-10-10: one public repository

TypeSafe permitted publishing Jev's per-question outputs on 2026-10-10
(`docs/permissions/2026-10-10-typesafe-jev-outputs.md`), which removed the
only reason for a private working repository next to a public export. From
that day on every test in this roadmap is designed, run, scored and
published in bks-lab/lab-bench. bks-lab/jev-match-bench is archived as
private provenance history; the export script is gone, and its leak scan
runs in CI on every push and pull request (README, "Repository history and
leak scan"). Job files name lab-bench commits in `repo_ref`; the queue
folder on the PC keeps its name.
