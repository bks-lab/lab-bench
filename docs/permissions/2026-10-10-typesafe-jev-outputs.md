# Permission: publishing Jev per-question outputs (TypeSafe, 2026-10-10)

## Record

- From: Avi Vemuri, TypeSafe AI (sent through TypeSafe's support system)
- To: Michael Boiman, BKS-Lab
- Date: 2026-10-10, 02:54 (+02:00)
- Subject: "Re: Publishing Jev benchmark outputs: question about MCA section 2.3(b)"
- Answer to: BKS-Lab's question whether the per-question outputs of Jev
  that this bench records (the rows under `results/*/jev/`) may be
  published, given section 2.3(b) of TypeSafe's Master Customer Agreement.
  BKS-Lab offered options; TypeSafe chose option 1, publishing the
  per-question outputs together with the benchmark results.

E-mail addresses of both sides are left out of this record on purpose.

## Operative text (quoted verbatim)

> Thanks for checking first. Publishing per-question outputs and benchmark
> results is fine (option 1), as long as the use follows our Acceptable Use
> Policy and the MCA. That means no distillation, no training a model on the
> outputs, and no building a competing product (MCA 2.3(b)).
>
> AUP: https://typesafe.ai/legal/acceptable-use-policy
> MCA: https://typesafe.ai/legal/mca

## Conditions that travel with the published outputs

1. No distillation: the outputs must not be used to distil a model.
2. No training: the outputs must not be used to train or fine-tune a model.
3. No competing product: the outputs must not be used to build a product
   that competes with TypeSafe (MCA section 2.3(b)).
4. Use follows TypeSafe's Acceptable Use Policy
   (https://typesafe.ai/legal/acceptable-use-policy) and Master Customer
   Agreement (https://typesafe.ai/legal/mca).

## What this changes in the bench

- `bench/export-allow.json` no longer excludes `results/*/jev/`. The public
  export (bks-lab/lab-bench) carries the Jev result rows of pv1 and route1.
- The export keeps only the known row fields of Jev rows and drops any
  other field, and `bench/leak-scan.mjs` fails on a Jev row with a field
  outside that list, so a future request id, header or account field cannot
  slip out.
- The published Jev rows carry a usage notice:
  [LICENSE-JEV-OUTPUTS](../../LICENSE-JEV-OUTPUTS) at the repository root and
  `NOTICE.md` in every `results/*/jev/` directory. The CC BY 4.0 grant of
  [LICENSE-DATA](../../LICENSE-DATA) does not cover these rows.
- This record itself is public, without e-mail addresses.
