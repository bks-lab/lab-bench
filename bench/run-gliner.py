#!/usr/bin/env python3
"""
GLiNER arm: asks Fastino's GLiNER2.5 Decide classifiers the same questions Jev
gets, one question per call, and stores the label probabilities.

    python bench/run-gliner.py --model fastino/GLiNER2.5-multi-Decide [--arm gliner2.5-multi-decide]
        [--mode prompt|intext] [--focus] [--pv pv1] [--run 1] [--lang de] [--unit line] [--limit N] [--device cuda]
        [--per-group N] [--tag smoke]
    -> results/<pv>/<arm>/<date>-run<k>[-<lang>][-<unit>][-<tag>].jsonl

Needs `pip install "gliner2[local]"` (2.0.0 was used) and, for the GPU,
torch with CUDA. Ran on a local workstation (RTX 4090) on 2026-10-08.

How a typed question becomes a GLiNER call:
  - the passage is the state as named sections, built exactly like
    run-local.mjs (`requirement`, and `cv` with one line per entry as
    cv.<id>), so a local LLM and GLiNER read the same text
  - one single-label task per call, the task instruction is Jev's wording,
    the labels are the option keys of normalize.mjs (noul: yes/no, score:
    "0".."3", choice: the criteria keys) with Jev's option text as the label
    description
  - two adaptations, both forced by the model and recorded per row in `adapt`:
      * GLiNER rejects "(" and ")" in instructions, labels and descriptions
        (they are prompt markers), so they become "[" and "]"
      * the evidence options are references like `cv.s-x`. An encoder cannot
        look a reference up, so each evidence label gets the text of its CV
        entry as description instead of the bare reference
  - family route1 (any --pv not starting with "pv"): the label is the
    option's text (the readable label name, "alarm: set") and the
    description is the same text, so GLiNER sees what the other arms see;
    the answer is mapped back to the option key. No evidence references
    exist there. --per-group N keeps the first N requests of every (source,
    language) group, for smoke runs
  - the probabilities are the classifier's own softmax over the labels of the
    task (gliner2 ClassificationScores.probability, activation auto =
    softmax for an exclusive task). They are a distribution over the options,
    not a calibrated probability, the same caveat as the letter logprobs
  - `arm_revision` is the commit sha of the cached Hugging Face snapshot the
    model was loaded from (null when it cannot be read)
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--arm')
ap.add_argument('--pv', default='pv1')
ap.add_argument('--run', type=int, default=1)
ap.add_argument('--lang', default='')
ap.add_argument('--unit', default='')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--device', default='cuda')
# prompt: Jev's instruction goes into the task's prompt field (the model card's
#   "question over a passage" form). intext: the instruction is appended to the
#   passage as "Question: ..." the way run-local.mjs puts it in the user turn,
#   and the task gets no prompt
ap.add_argument('--mode', default='prompt', choices=['prompt', 'intext'])
# focus: the passage holds only the state sections the question names in
#   backticks (`requirement`, `cv`, `project`), so a requirement question does
#   not carry the whole CV. An encoder adaptation, the other arms always got
#   the full state
ap.add_argument('--focus', action='store_true')
ap.add_argument('--host', default='local', help='label stored in every row, not a network address')
ap.add_argument('--per-group', type=int, default=0)
ap.add_argument('--tag', default='')
args = ap.parse_args()
ARM = args.arm or args.model.split('/')[-1].lower()
FAMILY_PV = args.pv.startswith('pv')

RESERVED = ('[P]', '[L]', '[C]', '[E]', '[R]', '[DESCRIPTION]', '[EXAMPLE]', '[OUTPUT]')


def clean(s):
    """GLiNER's prompt markers may not appear in prompt strings."""
    s = s.replace('(', '[').replace(')', ']')
    for tok in RESERVED:
        s = s.replace(tok, tok[:-1] + ' ]')
    return s


def state_text(state):
    parts = []
    for k, v in state.items():
        if isinstance(v, dict):
            parts.append(f'`{k}`:\n' + '\n'.join(f'{k}.{i}: {t}' for i, t in v.items()))
        else:
            parts.append(f'`{k}`:\n{v}')
    return '\n\n'.join(parts)


def options_of(q):
    """Same order as normalize.mjs optionsOf."""
    if q['type'] == 'noul':
        return ['no', 'yes']
    if q['type'] == 'score':
        return [str(i) for i in range(len(q['criteria']))]
    return list(q['criteria'].keys())


REF = re.compile(r'^`cv\.([^`]+)`$')


def labels_of(q, state):
    """{label: description} in option order, plus the adaptations made."""
    adapt = set()
    if not FAMILY_PV:
        # route1: label = description = the option text, keyed by the text
        out = {v: v for v in q['criteria'].values()}
        cleaned = {clean(k): clean(v) for k, v in out.items()}
        if list(cleaned) != list(out):
            adapt.add('parens')
        return cleaned, adapt
    if q['type'] == 'noul':
        return {'no': 'No', 'yes': 'Yes'}, adapt
    if q['type'] == 'score':
        texts = q['criteria']
        out = {str(i): t for i, t in enumerate(texts)}
    else:
        out = {}
        for k, v in q['criteria'].items():
            m = REF.match(v)
            if m and isinstance(state.get('cv'), dict) and m.group(1) in state['cv']:
                v = state['cv'][m.group(1)]
                adapt.add('evidence-text')
            out[k] = v
    cleaned = {k: clean(v) for k, v in out.items()}
    if any(cleaned[k] != out[k] for k in out):
        adapt.add('parens')
    return cleaned, adapt


def model_revision(repo):
    """Commit sha of the cached Hugging Face snapshot the model was loaded from, or None."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:
        return None


def main():
    import torch
    from gliner2.classification import Classifier, ClassificationSchema, ClassificationConfig

    reqs = [json.loads(l) for l in (ROOT / 'requests' / args.pv / 'requests.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.lang:
        reqs = [r for r in reqs if r['lang'] == args.lang]
    if args.unit:
        reqs = [r for r in reqs if r['unit'] == args.unit]
    if args.per_group:
        seen = {}
        keep = []
        for r in reqs:
            k = (r['line'].split('-')[0], r['lang'])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.per_group:
                keep.append(r)
        reqs = keep
    if args.limit:
        reqs = reqs[:args.limit]

    t_load = time.perf_counter()
    clf = Classifier.from_pretrained(args.model)
    clf.to(device=args.device)
    clf.eval()
    load_s = time.perf_counter() - t_load
    import gliner2
    version = getattr(gliner2, '__version__', None)
    try:
        from importlib.metadata import version as pkg_version
        version = version or pkg_version('gliner2')
    except Exception:
        pass
    runtime = f'gliner2 {version}, torch {torch.__version__}, {args.device}'
    revision = model_revision(args.model)
    cfg = ClassificationConfig()

    date = dt.date.today().isoformat()
    suffix = (f'-{args.lang}' if args.lang else '') + (f'-{args.unit}' if args.unit else '') + (f'-{args.tag}' if args.tag else '')
    out_dir = ROOT / 'results' / args.pv / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{date}-run{args.run}{suffix}.jsonl'

    # one warm-up call, so the first measured latency is not the CUDA start
    warm = ClassificationSchema().single('warmup', {'yes': 'Yes', 'no': 'No'}, instruction='Is this a test?')
    clf.score('This is a test.', warm, config=cfg)

    n = errors = 0
    t0 = time.perf_counter()
    with open(out_file, 'w', encoding='utf-8') as out:
        for r in reqs:
            state = r['request']['state']
            text = state_text(state)
            for qid, q in r['request']['questions'].items():
                if args.focus:
                    named = set(re.findall(r'`([a-z_]+)`', q['instructions']))
                    if q['type'] == 'choice' and any(REF.match(v) for v in q['criteria'].values()):
                        named.add('cv')
                    text = state_text({k: v for k, v in state.items() if k in named} or state)
                options = options_of(q)
                labels, adapt = labels_of(q, state)
                instr = clean(q['instructions'])
                if instr != q['instructions']:
                    adapt.add('parens')
                task = re.sub(r'[^A-Za-z0-9_]', '_', qid)
                probs = choice = err = None
                ms = None
                try:
                    if args.mode == 'prompt':
                        schema = ClassificationSchema().single(task, labels, instruction=instr)
                        passage = text
                    else:
                        schema = ClassificationSchema().single(task, labels)
                        passage = f'{text}\n\nQuestion: {q["instructions"]}'
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    scores = clf.score(passage, schema, config=cfg)
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    ms = round((time.perf_counter() - t) * 1000)
                    p = {k: scores.probability(task, k) for k in labels}
                    if not FAMILY_PV:  # labels are option texts: map back to the option keys
                        p = {key: p[clean(text)] for key, text in q['criteria'].items()}
                    probs = [round(p[o], 4) for o in options]
                    choice = options[max(range(len(options)), key=lambda i: p[options[i]])]
                except Exception as e:  # keep going, record the error per row
                    err = f'{type(e).__name__}: {e}'[:300]
                    errors += 1
                row = {
                    'pv': r['pv'], 'unit': r['unit'], 'pair': r['pair'], 'posting': r['posting'], 'cv': r['cv'],
                    'line': r['line'], 'question': qid, 'qtype': q['type'], 'lang': r['lang'],
                    'arm': ARM, 'arm_version': args.model, 'arm_revision': revision, 'mode': args.mode + ('+focus' if args.focus else ''), 'runtime': runtime, 'host': args.host,
                    'run': args.run, 'shuffled': False, 'latency_ms': ms, 'tokens_in': None, 'cost_usd': 0,
                    'ts': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'options': options, 'probs': probs, 'choice': choice,
                }
                if adapt:
                    row['adapt'] = sorted(adapt)
                if err:
                    row['error'] = err
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                n += 1
                if n % 200 == 0:
                    print(f'{n} questions, {round(time.perf_counter() - t0)} s', file=sys.stderr, flush=True)
    print(f'{ARM} run {args.run}: {n} questions, {errors} errors, load {load_s:.1f} s, '
          f'{round(time.perf_counter() - t0)} s -> {out_file.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
