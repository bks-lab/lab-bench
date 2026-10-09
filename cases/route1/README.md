# Test cases route1 (routing and classification)

Frozen inputs of the route1 family: short texts that a router has to put into
one of a fixed set of classes. Unlike pv1 nothing here is invented. All rows
come from two public datasets and keep their gold labels, so the reference is
the dataset's own annotation (`reference/route1/`), not a judge.

Frozen on 2026-10-09 by `bench/freeze-route-ex.mjs`. Any change to these
files makes a new version (`route2`).

## Files

| file | rows | fields |
|---|---|---|
| `massive-de.jsonl` | 2,974 | `id`, `utt`, `annot_utt`, `scenario`, `intent` |
| `massive-en.jsonl` | 2,974 | same, same ids |
| `fastdec-en.jsonl` | 1,700 | `id` (`<domain>-<row>`), `domain`, `row`, `input`, `tasks` (`task`, `labels`, `gold`) |

## MASSIVE 1.1

- Source: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz,
  downloaded 2026-10-09, SHA-256 of the tarball
  `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
- Taken: every row of partition `test` in `1.1/data/de-DE.jsonl` and
  `1.1/data/en-US.jsonl`, only the fields `id`, `utt`, `annot_utt`,
  `scenario`, `intent`. 2,974 rows per language, 18 scenarios, 60 intents.
- de-DE is a localisation of en-US. The freeze script checks that both files
  have the same ids and that every id carries the same scenario and intent in
  both languages. Both checks passed.
- Licence: CC BY 4.0, Copyright Amazon.com Inc. or its affiliates.
- SLURP: MASSIVE includes and translates text from SLURP (Bastianelli et al.,
  "SLURP: A Spoken Language Understanding Resource Package", EMNLP 2020,
  https://github.com/pswietojanski/slurp), licensed under CC BY 4.0. The
  MASSIVE tarball says so in `1.1/NOTICE.md` and asks for a SLURP citation in
  `1.1/CITATION.md`.
- Attribution: Jack FitzGerald et al., "MASSIVE: A 1M-Example Multilingual
  Natural Language Understanding Dataset with 51 Typologically-Diverse
  Languages", 2022, arXiv:2204.08582. https://github.com/alexa/massive
- Changes: rows filtered to the test partition, fields reduced, sorted by id.
  No text was changed.

## fast-decisions (dev split)

- Source: https://huggingface.co/datasets/fastino/fast-decisions at revision
  `35afb79a0f611a1db44380e855ac2a821b0a4da1` (last modified 2026-10-08), the 17
  files `<domain>.jsonl`, 100 rows each, English only.
- Taken: every row, but only the single-label tasks (`multi_label: false`).
  2,600 single-label tasks in 1,700 rows. The 300 multi-label tasks
  (`product_area`, `aspects`, `genres`) are dropped.
- Licence: Apache License 2.0 (dataset card). Published by Fastino; the card
  carries no copyright line.
- Citation requested by the card: Urchade Zaratiana, Gil Pasternak, Oliver
  Boyd, George Hurn-Maloney and Ash Lewis, "GLiNER2: An Efficient Multi-Task
  Information Extraction System with Schema-Driven Interface", 2025,
  arXiv:2507.18546.
- Changes: multi-label tasks removed, fields renamed (`true_label[0]` is
  `gold`). No text was changed.
- **Home ground on purpose.** Fastino built this dataset. The dataset card
  calls it "the classification suite behind GLiNER2.5-Decide" and does not
  say whether the dev rows were used in training. The dev rows may be close to
  the Decide training data, so a high Decide score here is expected and is
  not a neutral result. The held-out test split (300 rows per domain) is not
  published, so Fastino's own numbers cannot be reproduced; the card itself
  says not to report a score on these files as the benchmark.
- One task, `support_intent:intent`, has 28 labels. The local arm can offer at
  most 26 letters, see `results/route1/method.md`.

## Public test data

MASSIVE test has been public since 2022, and fast-decisions dev is public
on Hugging Face. No arm (Jev, Gemma 4 and the Winnow LoRA, the GLiNER models) was checked for
exposure to these rows.

## Reproducing the freeze

From an empty working folder outside the repository (`<repo>` is the
repository root):

```bash
curl -L -o massive.tgz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
shasum -a 256 massive.tgz   # 4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577
tar -xzf massive.tgz 1.1/data/de-DE.jsonl 1.1/data/en-US.jsonl
mkdir fastdec
for d in agent_handoff banking_intent benefits_request clinic_request document_type \
         email_triage news_topic paper_field product_feedback restaurant_review \
         review_sentiment screen_tags sports_recap support_intent support_topic \
         ticket_route travel_request; do
  curl -L -o "fastdec/$d.jsonl" "https://huggingface.co/datasets/fastino/fast-decisions/resolve/35afb79a0f611a1db44380e855ac2a821b0a4da1/$d.jsonl"
done
node <repo>/bench/freeze-route-ex.mjs --massive 1.1/data --fastdec fastdec
node <repo>/bench/build-route-ex.mjs
```

`--massive` points to the extracted `1.1/data` folder, `--fastdec` to the
folder with the 17 domain files. Afterwards `git status` in the repository
is clean: every file under `cases/`, `requests/` and `reference/` comes out
byte-identical.
