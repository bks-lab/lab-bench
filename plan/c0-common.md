# C0: shared rules for the capability family C1 to C9

Written 2026-10-09, before any C run. Every C design in `plan/` points
here; where a C file says something different, the C file holds for that
test. A change after the commit goes into a new dated section at the end,
never into the text above it.

## What the family is for

The tests T1 to T11 compare tools against Jev. The capability family asks a
different question: what can a single in-house workstation do on its own,
with open models, on German business tasks? The machine is the workstation
that ran the local arms (RTX 4090 24 GB, Windows, Ollama 0.35.0, a Python
venv with torch CUDA; result rows record it as `local-rtx4090`). Jev takes
no part. A German SME
that wants AI in house needs three answers per task: how good, how fast,
and whether it fits on one card. Every C test reports all three.

## Machine record, per run

Every result file starts with a `machine` object, written by the runner,
never by hand:

- GPU name, driver version, CUDA version (`nvidia-smi
  --query-gpu=name,driver_version --format=csv` and the CUDA version line of
  `nvidia-smi`), power limit (`power.limit`), GPU and memory clocks at idle
- OS build, CPU, RAM
- runtime versions: Ollama, Python, torch, transformers,
  sentence-transformers, faster-whisper, NeMo, or whatever the arm uses
- per arm: model id, revision or Ollama digest, quantisation, context
  length (`num_ctx`), batch size, concurrency (`OLLAMA_NUM_PARALLEL`)

GPU sampling runs for the whole arm: `nvidia-smi
--query-gpu=timestamp,memory.used,power.draw,utilization.gpu,temperature.gpu
--format=csv -l 1`, written next to the result file. From it the scorer
takes peak VRAM (max `memory.used` minus the idle value before the model
loads, both reported), mean power during the timed section, and energy as
the sum of the 1 s power samples in that section (joules). Energy is GPU
only, never the whole system, and is labelled so. A sample interval of 1 s
undercounts bursts shorter than a second; the per-item energy is therefore
reported only as total energy divided by items, never per single item.

Latency is wall time per item at batch size 1 after three warm-up items,
unless a C file names a batched mode as well. Model load time is reported
separately and never mixed into per-item latency.

## Uncertainty

- Every headline metric gets a 95 % bootstrap confidence interval:
  percentile method, 1,000 resamples, numpy `default_rng(20261009)`. The
  unit that is resampled is named in each C file (query, utterance,
  sentence, invoice, document, prompt). Where items share a source (for
  example several FLEURS speakers reading the same sentence), the C file
  names the cluster and the bootstrap resamples clusters.
- A difference between two models is called a difference only if a paired
  test on the same items says so at p < 0.05 (paired bootstrap of the
  difference with the same generator, or an exact McNemar test for
  right/wrong items), or if the two intervals do not overlap. Otherwise the
  result is "no difference shown".
- Speed numbers get the same bootstrap over items for medians; throughput
  numbers are repeated three times and reported as mean and range.

## Pipeline sanity gate

Each C file names one published reference number for one candidate on the
same data, or says why none exists and names an internal check instead.
That candidate runs first. If our number is outside the stated tolerance,
the test stops, the cause is found and written into a dated section of the
C file, and only then do the other arms run. A gate that passes is
reported with the number.

## Licence class

Every candidate carries one of these classes, read on the model page on
2026-10-09 (licence field of the Hugging Face card and the linked licence
text, or the Ollama library page):

- `open source`: an OSI licence on the weights (Apache-2.0, MIT)
- `open source (CC BY)`: CC BY 4.0 on the weights, permissive with
  attribution, not an OSI software licence
- `open weights, restricted`: a licence with use restrictions or an
  acceptable use policy (Llama, Gemma 3 terms, NVIDIA Open Model Licence)

Non-commercial licences (CC BY-NC, CC BY-NC-SA) are not candidates at all:
an SME could not use them in production, so a result would advise nothing.
The capability map shows the class next to every model.

## Training data exposure

Public test sets may be in a model's training data. Each C file states,
per dataset, what is known about exposure. No arm is checked for
contamination beyond that statement, and a high score on an exposed set is
reported with that caveat.

## Files

- inputs: `cases/<cN>/` (only where the licence allows publishing the text,
  otherwise ids and hashes only, as each C file says)
- results: `results/<cN>/<arm>/<date>-run1.jsonl`, `results/<cN>/score.md`,
  `results/<cN>/findings.md`, `results/<cN>/gpu/<arm>.csv`
- capability map row: `results/capabilities.json` (see `plan/ROADMAP.md`)

## Where it runs

All GPU runs go through the PC job queue described in `plan/ROADMAP.md`,
one job at a time, after the T12 series has finished. Preparation that
needs no GPU (downloads, rendering, sampling, freezing inputs) runs on a
Mac or on the PC CPU beforehand.
