---
license: cc-by-4.0
language:
  - de
  - en
pretty_name: BKS-Lab lab-bench results
tags:
  - benchmark
  - evaluation
  - local-models
  - german
  - sovereign-ai
size_categories:
  - n<1K
configs:
  - config_name: toolmap
    data_files: data/toolmap.jsonl
    default: true
  - config_name: capabilities
    data_files: data/capabilities.jsonl
---

# lab-bench results

Aggregated results of [lab-bench](https://github.com/bks-lab/lab-bench), the
test bench BKS-Lab uses to decide which model can run a task on hardware we
control. Every number here is measured on one RTX 4090 with 24 GB, or comes
from a hosted service where the table says so.

This dataset holds the summary tables only. Cases, requests, references,
per-question result rows and the code that produced them live in the
repository, at the commit named below. Every row links to its report there.

- Built from commit `{{COMMIT}}` of bks-lab/lab-bench, tool map generated {{GENERATED}}.
- Write-ups: [Test bench](https://bks-lab.com/en/labor/test-bench/) and
  [Decision models](https://bks-lab.com/en/labor/entscheidungsmodelle/) on bks-lab.com.

## Configs

**toolmap.** One row per task and tool family. Families are a hosted
decision API (TypeSafe Jev), GLiNER encoders, open LLMs through Ollama and a
classic trained classifier. Tasks are routing (MASSIVE 1.1, fast-decisions),
slot extraction (MASSIVE 1.1) and a CV to job posting match. `state` says
whether a cell is measured, planned, open or not applicable. `skill` is
`(value - baseline) / (100 - baseline)`. `best_arm` is chosen after the fact
on the same data, which flatters families with many arms. The p values are
exact McNemar tests against the top family and against Jev, uncorrected.
Fields that are not columns are kept in `extra_json`.

**capabilities.** One row per capability test (C1 German retrieval for RAG,
C2 German speech to text, C3 invoice fields from page images, more to come).
Each row names the best local model, its licence class, the quality value
with a 95 % bootstrap interval, the baseline, a sanity check against a
published reference value, speed, peak VRAM and GPU energy.

## What is not in here, and why

- **No per-question outputs of Jev.** They are published in the repository
  with TypeSafe's permission for reproducing the measurements only and must
  not be used to train or distil a model or to build a competing product,
  see [LICENSE-JEV-OUTPUTS](https://github.com/bks-lab/lab-bench/blob/{{COMMIT}}/LICENSE-JEV-OUTPUTS).
  A dataset on the Hub is a training source, so they stay out. The
  aggregated scores of Jev are benchmark results and are included.
- **No cases.** The routing and extraction cases come from MASSIVE 1.1
  (Amazon, CC BY 4.0) and fastino/fast-decisions (Apache-2.0). Use them from
  their publishers.

## How to cite the numbers

Run 1 of each arm, one run per arm unless a report says otherwise. A cell
compares tool families on one task and says nothing about another task. If a
number matters to you, read its report first: the method, the limits and the
open questions are written down there.

## Licence

The tables are CC BY 4.0, BKS-Lab contributors. Rows derived from MASSIVE
1.1 also credit Amazon (CC BY 4.0), rows derived from fast-decisions credit
Fastino (Apache-2.0).
