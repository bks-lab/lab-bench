# lab-bench

A repeatable test bench for the CV-to-posting match: does an open model, run
locally or hosted in the EU, answer the match questions as well as Jev?

## Notice: public export, Jev outputs under their own terms

This repository is the public export of BKS-Lab's private bench
(source commit `36d699ed8ff5`). It carries the cases, requests, references,
reports and the result rows of every arm.

- Jev's per-question outputs (the result rows under `results/*/jev/`) are
  published with TypeSafe's permission of 2026-10-10
  ([record](docs/permissions/2026-10-10-typesafe-jev-outputs.md)), for
  reproducing the measurements only. They must not be used to train or
  distil models or to build a product that competes with TypeSafe. Use is
  subject to TypeSafe's
  [Acceptable Use Policy](https://typesafe.ai/legal/acceptable-use-policy)
  and [Master Customer Agreement](https://typesafe.ai/legal/mca), section
  2.3(b). Neither the MIT nor the CC BY 4.0 licence of this repository
  covers them, see [LICENSE-JEV-OUTPUTS](LICENSE-JEV-OUTPUTS).
- Not exported: the export tooling and every file not yet on the export
  path list (each export commit lists them).
- The scripts still run Jev with your own key (`TYPESAFE_API_KEY`, see
  `bench/run-jev.mjs`).

## Why

The match on `mboiman.github.io/<lang>/match/` asks TypeSafe's Jev many small
typed questions per posting line (is it a requirement, is it mandatory, which
area, which CV entry is the evidence, how well is it covered) and computes the
score in code. TypeSafe hosts its services in the USA (TypeSafe privacy
policy, https://typesafe.ai/legal/privacy-policy, linked from
https://docs.typesafe.ai/legal, read 2026-10-09). For a service that handles
other people's CVs we want to know whether a model we can run ourselves is
good enough, and we want to be able to repeat that test whenever a new model
appears.

Write-up of the results: [Jev against local models: decisions on our own
hardware](https://bks-lab.com/en/blog/jev-gegen-lokale-modelle/) on the
BKS-Lab blog, and the lab page [Decision models](https://bks-lab.com/en/labor/entscheidungsmodelle/)
with the questions and numbers.

## Three families

The bench now has three families of test cases. Each has its own cases,
requests, reference and method file.

| family | task | data | reference | method |
|---|---|---|---|---|
| `pv1` (match) | CV against a posting line: requirement, must, area, evidence, level | invented postings and CVs, de and en | blind model judges, adjudicated | [results/pv1/method.md](results/pv1/method.md) |
| `route1` (routing) | put a short text into one of a fixed set of classes | MASSIVE 1.1 test (de, en, 2,974 each) and fastino/fast-decisions dev (en, single-label tasks) | the datasets' gold labels | [results/route1/method.md](results/route1/method.md) |
| `ex1` (extraction) | find the slots of a spoken command | 1,000 MASSIVE 1.1 test utterances per language | MASSIVE slot annotation | [results/ex1/method.md](results/ex1/method.md) |

Results, one line each (run 1 of each arm):

- **pv1:** Jev leads on axis and level, Qwen3.8-27B on evidence (87.9 % de
  against Jev 81.0 %), GLiNER2.5 Decide stays at the most-frequent-answer
  line (best mean of five questions 53.6 % de, baseline 54.3 %, Jev 86.9 %).
  [gliner.md](results/pv1/gliner.md), [crosscheck.md](results/pv1/crosscheck.md)
- **route1:** on MASSIVE Jev is first (scenario 72.8 % de, 74.1 % en),
  Winnow-12B within 0.3 to 3.5 points, GLiNER 13.2 to 31.8 points behind; on
  fast-decisions (26 labels or fewer) the English Decide models, Jev and
  Winnow-12B are level at 63.0 to 64.4 %.
  [findings.md](results/route1/findings.md)
- **ex1:** slot F1 Winnow-12B 45.1 % de and 48.9 % en, best GLiNER 22.5 % de
  and 38.1 % en; Jev not applicable.
  [findings.md](results/ex1/findings.md)
- Machine-readable summary of every measured cell for the tool map:
  [results/toolmap.json](results/toolmap.json).

Jev takes part in pv1 and route1. It has no extraction primitive, so ex1
compares a local LLM and GLiNER only. Sources, hashes and licences of the
route1 and ex1 data: [cases/route1/README.md](cases/route1/README.md).
The full route1 and ex1 run is one command per machine:
`bench/run-route-ex.sh` (Jev and Ollama) and `bench/run-route-ex.bat`
(GLiNER on the GPU machine).

A second family is planned and preregistered, but not yet run:
**capabilities of a local workstation**, C1 to C9. These tests do not
compare against Jev. They ask what one in-house PC (RTX 4090, 24 GB) with
open models does on German business tasks: retrieval for RAG, speech to
text, invoice fields from rendered e-invoices, structured output and tool
calling, translation, reading comprehension, long context, throughput and
energy, and personal data detection. Each reports quality with a 95 %
bootstrap interval, speed, peak VRAM, GPU energy and the licence class of
the model, and each must first reproduce a published reference number
before other models are scored. Designs: [plan/c0-common.md](plan/c0-common.md)
and `plan/c1.md` to `plan/c9.md`; data, licences, run order and GPU hours:
[plan/ROADMAP.md](plan/ROADMAP.md).

## Tool map

`results/toolmap.json` is the machine-readable summary: one row per task,
one cell per tool family (`jev`, `gliner`, `local-llm`). `node
bench/toolmap.mjs` rebuilds it from the score reports, the pv1 references,
the result rows and the register `plan/toolmap-plan.yaml`. It recounts the
route1 and pv1 accuracies from the rows and stops if they differ from the
reports.

route1 and pv1 cells carry `top_family` (the family with the highest value in
that row) and `mcnemar_vs_top_p`: an exact paired McNemar between the cell's
best arm and the top family's best arm on the same items. Cells other than
Jev also carry `mcnemar_vs_jev_p`. Both compared arms are chosen after the
fact as the best of their family and no p value is corrected for that, so
p >= 0.05 says the difference is not established, not that the two are level.
ex1 cells have no test, and their informative skill is `exact_frame_skill`
(whole frames against extracting nothing), since F1 against an empty
extraction is 0 by construction.

Every cell has a `state`:

| state | meaning |
|---|---|
| `measured` | a value from a run, with `best_arm`, baseline, skill, version and `source_report` |
| `planned` | a dated test design is committed under `plan/` (`plan` field), nothing is measured yet |
| `open` | a sensible test nobody has designed yet |
| `not_applicable` | the tool cannot do the task, `reason` says why (Jev on extraction) |

The register also carries `tools`: maker, how each tool runs, licence and
sources. A row can exist before any measurement. Planned tests are written
down in `plan/` and committed before they run, with data, questions, arms,
reference, metric, baseline and thresholds fixed in advance, so the design
cannot be fitted to the results. The first one is
[plan/einv1.md](plan/einv1.md): ten yes/no questions over the free text of
34 KoSIT e-invoices.

## Principles

1. **Frozen versions.** A bench version (`pv1`) never changes once the
   reference judgment starts. New cases make a new version.
2. **Invented CVs only.** Every arm may see every case, including APIs in the
   USA, and the cases can live in a repository.
3. **The service's own questions.** `bench/match.ts` is a vendored copy of the
   service's `src/lib/match.ts` (source commit in its header). The bench
   measures what the service does, including its faults.
4. **Every question on its own.** The total score is derived and can hide
   errors that cancel out.
5. **The reference is decided blind**, before anyone looks at model answers.
   For pv1 that is three blind Opus judges, and for German a cross-check by
   blind Sonnet and Fable judges with a researched adjudication of every
   disagreement (`adjudicated-a`, see Reference judgment pv1). A human
   sample is still pending.

## Layout

| Path | What |
|---|---|
| `cases/pv1/` | postings (en, de), 8 invented CVs (en, de), 10 pairs. See its README |
| `bench/match.ts` | vendored questions and scoring of the service |
| `bench/build-requests.mjs` | cases to `requests/<pv>/requests.jsonl`, the exact requests the service would send |
| `bench/run-jev.mjs` | Jev arm, TypeSafe API |
| `bench/run-local.mjs` | local arm through Ollama: options as letters, probabilities from the logprobs of the answer letter |
| `bench/run-gliner.py` | GLiNER2.5 Decide arm (Python, `pip install "gliner2[local]"`): one classification task per question, modes `prompt`, `intext`, `--focus`. Result: [`results/pv1/gliner.md`](results/pv1/gliner.md) |
| `bench/normalize.mjs` | one output row per question, the same for every arm |
| `bench/analyze.mjs` | report: run-to-run flips, option-order effect, agreement with Jev, total score per pair |
| `bench/significance.mjs` | paired exact McNemar per question, each arm against Jev on run 1, scored against a reference: `node bench/significance.mjs --ref claude-a --lang de --md results/pv1/significance-de.md` |
| `results/<pv>/<arm>/` | one JSONL file per run |
| `bench/freeze-route-ex.mjs`, `bench/build-route-ex.mjs` | route1 and ex1: freeze the source data into `cases/`, then build requests and gold |
| `bench/run-local-extract.mjs`, `bench/run-gliner-extract.py` | ex1 arms: local LLM with JSON output, GLiNER2 entity extraction |
| `bench/score-route.mjs`, `bench/score-extract.mjs` | route1 and ex1 scorers, reports in `results/route1/` and `results/ex1/` |
| `bench/toolmap.mjs`, `plan/toolmap-plan.yaml` | tool map generator and its register, see Tool map |
| `plan/` | dated test designs, committed before the run |
| `bench/judge-server.mjs`, `bench/judge.html` | blind reference judgment: `node bench/judge-server.mjs`, then http://localhost:4350/ (English cases and labels) or http://localhost:4350/?lang=de (German). Listens on 127.0.0.1 only. Never shows a model answer, saves every item at once, resumes where you stopped |
| `reference/<pv>/` | blind reference judgment, `<judge>.<lang>.jsonl`, judges and models in `judges.json`. pv1: claude-a and claude-b (de), claude-c (en), all `claude-opus-5-5`, 2026-10-01; sonnet-a, fable-a and adjudicated-a (de), 2026-10-02 |

## Running

Needs Node.js 22.18 or later: the `.mjs` scripts import `bench/match.ts`
directly, which relies on Node's built-in type stripping. The GLiNER arm
needs Python 3 with `pip install "gliner2[local]"`.

```bash
npm install
node bench/build-requests.mjs                     # cases -> requests
node bench/run-jev.mjs --runs 5                   # needs TYPESAFE_API_KEY, see the script header
node bench/run-local.mjs --model qwen3:32b --template chatml --runs 1   # Ollama at OLLAMA_HOST or localhost:11434
python bench/run-gliner.py --model fastino/GLiNER2.5-Decide-1B --arm gliner2.5-decide-1b-focus --focus
node bench/analyze.mjs --md results/pv1/report.md
```

Run 5 of 5 is the shuffled run: the options of every choice question go out in
a seeded random order. `--unit line|project|profile`, `--only p01,p02` and
`--run-ids 1,5` narrow a run.

## How a local model answers a typed question

Jev returns a probability distribution per question. A raw model does not, so
`run-local.mjs` asks one question per call: the state as named sections, Jev's
instruction word for word, the options as letters A, B, C, and the assistant
turn pre-filled up to `Answer: **`. The probability of an option is the
probability of its letter as the next token, renormalised over the options.
`mass` in each row says how much probability the letters had before that; a
low mass means the model wanted to answer something else. Winnow-12B and
Shisa DE-1 ran with `--template gemma4 --prefix ''`. Shisa DE-1 is trained to
put the letter first. Winnow-12B's own server puts `Answer:\n` before the
letter (winnow-inference, native/protocol.h), which this template leaves out. The prefix is not neutral: on 15
questions Shisa changed 6 answers between `''` and `Answer: **` (2026-10-01).
Locally all 348 answers of a second qwen3:32b run on p01 and p04 repeated
exactly (2026-10-01), the probabilities did not (up to 0.19 apart), so
calibration needs repeated runs here
too. Full method, the recomputed numbers and every difference between the
Jev arm and the local arms: [results/pv1/method.md](results/pv1/method.md).

## Known service faults the bench keeps

- The service's line splitter drops one-word lines. In German,
  "ISTQB-Zertifizierung" disappears, in English "ISTQB certification" stays,
  so `sap-testmanagement` has 13 English and 12 German lines (2026-10-01).

## Reference judgment pv1

Three blind Opus judges (model `claude-opus-5-5`, 2026-10-01, recorded in
`reference/pv1/judges.json`), none saw a model answer: `claude-a` and
`claude-b` judged the German lines independently, `claude-c` the English ones.
Scoring uses `claude-a` (de) and `claude-c` (en). `claude-b` is a comparison
row only. The English lines have a single judge and no such cross-check.
A and B agree on 100 % of is_req and must, 97 % of axis, 95 % of evidence and
level, so the gaps of the models below are real, not judge noise. Caveat: two
judges from the same model family can share a blind spot; a human sample on a
few pairs is still worth doing before a public claim. `bench/score.mjs` scores
every arm against a reference (`results/pv1/score-de.md`, `score-en.md`).

Cross-check (2026-10-02, German only): two more blind judges from other
families, `sonnet-a` (`claude-sonnet-5-5`) and `fable-a` (`claude-fable-5-1`),
judged the same 636 pairs. The three agree on 90.4 % of them, on all is_req
and must answers. Each of the 61 disagreements went to a fresh researched
adjudicator (Sonnet or Fable); 26 verdicts changed the claude-a choice (13 of
them raise a level by one, 5 replace an entry with none). The result is
`reference/pv1/adjudicated-a.de.jsonl` (changed rows marked
`"adjudicated": true`), scored in `results/pv1/score-de-adjudicated.md` and
`significance-de-adjudicated.md`. Method, verdict counts and which findings
hold: `results/pv1/crosscheck.md`. In short: Qwen3.8-27B still invents
evidence least (now p = 0.016 against Jev) and is now first on evidence; Jev
stays first on level and axis, but Winnow-12B is no longer significantly
behind Jev on axis.

The `note` field in the reference files is the judges' raw output, kept
verbatim. Some judges wrote their notes in German, and some notes quote
third-party web pages with their URLs (see Licence).

Found by the judges: the service cuts CV entries at 600 characters
(`plain()` in match.ts), three entries end mid-sentence: s-claims-agents in
cv03 (de and en), s-s4-retail-testmgr in cv02 (de), s-trade-lakehouse in cv07
(de), found by scanning requests/pv1/requests.jsonl. The judges flagged only
p04-de-l09. Most is_req
disagreements are the title and frame lines (title, duration, location), which
the judges call no and every model calls yes; that is a definition question.

## Coverage

Which arm ran on which family, run 1 unless noted. "n/a" means the tool has
no way to answer that kind of question, "not run" means it could and was not
measured yet.

| tool family | arm | pv1 (match) | route1 (routing) | ex1 (extraction) |
|---|---|---|---|---|
| Jev | jev | runs 1 to 5 | run 1 | n/a (choice, score and noul only) |
| local LLM | winnow-12b | yes | yes | yes |
| local LLM | qwen3.8-27b | yes | not run | not run |
| local LLM | qwen3-32b | yes | not run | not run |
| local LLM | shisa-de-1 | yes | not run | not run |
| GLiNER Decide | gliner2.5-multi-decide (prompt, intext, focus) | yes | prompt, intext | n/a (classifier) |
| GLiNER Decide | gliner2.5-decide (prompt, intext, focus) | yes | prompt, intext | n/a (classifier) |
| GLiNER Decide | gliner2.5-decide-1b (prompt, intext, focus) | yes | prompt, intext | n/a (classifier) |
| GLiNER extraction | gliner2.5-multi | n/a | n/a | de, en |
| GLiNER extraction | gliner2-large | n/a | n/a | en only |

`focus` is a pv1 adaptation (only the state sections a question names) and
does not apply to route1, whose state is one short text. Planned next: the
e-invoice yes/no test in [plan/einv1.md](plan/einv1.md), written down before
it runs. The local LLMs other than Winnow-12B were left out of route1 and ex1
on purpose: Winnow-12B was the best local arm in pv1, and every extra arm
makes "best arm of the family" more flattering (see the note in
`results/toolmap.json`).

References: pv1 de against `adjudicated-a` (a model judgment cross-checked by
two more model judges) and the first judge `claude-a`, pv1 en against a
single model judge `claude-c`. route1 and ex1 against the gold labels of the
public datasets. No human reference sample yet.

## Public export

The bench has two homes: this private working repository and the public
repository [bks-lab/lab-bench](https://github.com/bks-lab/lab-bench). Since
2026-10-10 the public one also carries Jev's per-question outputs (the
result rows under `results/*/jev/`): TypeSafe permitted publishing them,
see [docs/permissions/2026-10-10-typesafe-jev-outputs.md](docs/permissions/2026-10-10-typesafe-jev-outputs.md).
The public copy is made by `bench/export-public.sh`, which:

- copies only the paths listed in `bench/export-allow.json`, so a new kind
  of file stays private until someone allows it,
- reduces every Jev row to the fields listed under `jevRows` in
  `bench/export-allow.json`, so a request id, header or account field is
  dropped instead of published,
- runs `bench/leak-scan.mjs`, which parses every structured file and does
  not reuse the selection rules,
- makes one commit with the GitHub noreply address.

Refresh after a merge to main (run in the private repository):

```bash
gh repo clone bks-lab/lab-bench /tmp/lab-bench
bench/export-public.sh --ref main --into /tmp/lab-bench
git -C /tmp/lab-bench push origin main
```

## Local runs

The local arms ran on a single workstation with an RTX 4090. Result rows
record it as `local-rtx4090` in the `host` field.

## Licence

- Code (everything under `bench/` except `bench/match.ts`): MIT, see
  [LICENSE](LICENSE).
- Data (cases, requests, reference judgments and results under `cases/`,
  `requests/`, `reference/` and `results/`): CC BY 4.0, see
  [LICENSE-DATA](LICENSE-DATA), except the Jev outputs below.
- `bench/match.ts` is a vendored copy from
  [mboiman/mboiman.github.io](https://github.com/mboiman/mboiman.github.io)
  and stays under CC BY 4.0 with the source credit in its header.
- Jev outputs (the result rows under `results/*/jev/`) are not covered by
  the MIT or the CC BY 4.0 licence. They are outputs produced by TypeSafe
  Jev, published with TypeSafe's permission of 2026-10-10
  ([record](docs/permissions/2026-10-10-typesafe-jev-outputs.md)), for
  reproducing the measurements only. They must not be used to train or
  distil models or to build a product that competes with TypeSafe. Use is
  subject to TypeSafe's
  [Acceptable Use Policy](https://typesafe.ai/legal/acceptable-use-policy)
  and [Master Customer Agreement](https://typesafe.ai/legal/mca), section
  2.3(b). Full notice: [LICENSE-JEV-OUTPUTS](LICENSE-JEV-OUTPUTS), and a
  `NOTICE.md` in every `results/*/jev/` directory.
- Quotes from third-party sources inside the judges' notes (the `Sources:`
  parts with URLs) belong to their authors and are not covered by the
  CC BY 4.0 grant.
- The route1 and ex1 data are not BKS-Lab's own. The MASSIVE rows
  (`cases/route1/massive-*.jsonl`, `cases/ex1/`, the requests, gold rows and
  result rows built from them) are CC BY 4.0 by Amazon and include text
  from SLURP, also CC BY 4.0. `cases/route1/fastdec-en.jsonl` and what is
  built from it are Apache-2.0, published by Fastino, see
  [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Credits and details:
  exceptions 4 and 5 of [LICENSE-DATA](LICENSE-DATA) and
  [cases/route1/README.md](cases/route1/README.md).
