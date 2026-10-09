#!/usr/bin/env python3
"""
GLiNER extraction arm (family ex1): Fastino's GLiNER2 entity extractors find
the MASSIVE slots of an utterance.

    python bench/run-gliner-extract.py --model fastino/gliner2.5-multi-v1 [--arm gliner2.5-multi]
        [--pv ex1] [--lang de] [--limit N] [--threshold 0.5] [--call-threshold 0.1] [--run 1] [--tag smoke]
        [--device cuda] [--check-spans] [--check-threshold]
    -> results/<pv>/<arm>/<date>-run<k>[-<lang>][-<tag>].jsonl, one row per utterance

Needs `pip install "gliner2[local]"` (2.0.0) and, for the GPU, torch with CUDA.

How the call is made:
  - gliner2.AutoExtractor.from_pretrained(model) (it loads span and boundary
    checkpoints alike), then extract_entities(utterance, entity_types,
    threshold=call_threshold, include_confidence=True, include_spans=True)
  - entity_types are the label texts of cases/<pv>/slot-labels.json (MASSIVE
    name with underscores replaced by spaces, "place_name" -> "place name",
    "timeofday" stays), the same texts the local LLM gets in its prompt. No
    descriptions. Returned labels are mapped back to the MASSIVE type before
    they are stored
  - overlap_policy is left at the library default
  - one call per utterance at the low call threshold (0.1); every returned
    entity with its confidence goes to `candidates`. `slots`, the primary
    prediction, are the candidates with confidence >= --threshold (0.5, the
    library default, fixed before any run). Both thresholds are stored per
    row. score-extract.mjs reads `candidates` for a sensitivity table at
    other thresholds from the same run
  - --check-threshold makes a second call at --threshold and counts the
    utterances where its result differs from the filtered candidates, to
    show that filtering a low-threshold call equals calling at 0.5
  - the scored value is the returned text. Spans (start, end) are stored for
    inspection only; --check-spans tests whether text == utterance[start:end]
  - `arm_revision` is the commit sha of the cached Hugging Face snapshot the
    model was loaded from (null when it cannot be read)
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--arm')
ap.add_argument('--pv', default='ex1')
ap.add_argument('--run', type=int, default=1)
ap.add_argument('--lang', default='')
ap.add_argument('--limit', type=int, default=0, help='first N utterances per language')
ap.add_argument('--threshold', type=float, default=0.5, help='primary threshold: slots = candidates at or above it')
ap.add_argument('--call-threshold', type=float, default=0.1, help='threshold of the actual call; candidates down to it are stored')
ap.add_argument('--device', default='cuda')
ap.add_argument('--tag', default='')
ap.add_argument('--host', default='local', help='label stored in every row, not a network address')
ap.add_argument('--check-spans', action='store_true')
ap.add_argument('--check-threshold', action='store_true', help='second call at --threshold, compared with the filtered candidates')
args = ap.parse_args()
ARM = args.arm or args.model.split('/')[-1].lower()


def model_revision(repo):
    """Commit sha of the cached Hugging Face snapshot the model was loaded from, or None."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:
        return None


def entities(res, type_of):
    """Flat list of returned entities, label mapped back to the MASSIVE type."""
    out = []
    for lab, items in (res.get('entities') or {}).items():
        for it in items:
            if isinstance(it, str):
                it = {'text': it}
            s = {'type': type_of.get(lab, lab), 'label': lab, 'text': it.get('text', '')}
            if 'confidence' in it:
                s['confidence'] = round(float(it['confidence']), 4)
            if 'start' in it and 'end' in it:
                s['start'], s['end'] = it['start'], it['end']
            out.append(s)
    return out


def primary(cands):
    """Candidates at or above the primary threshold (all of them when no confidence came back)."""
    return [s for s in cands if s.get('confidence', 1.0) >= args.threshold]


def ident(s):
    return (s['type'], s['text'], s.get('start'), s.get('end'))


def main():
    import torch
    from gliner2 import AutoExtractor
    try:
        from importlib.metadata import version as pkg_version
        version = pkg_version('gliner2')
    except Exception:
        version = None

    labels = json.loads((ROOT / 'cases' / args.pv / 'slot-labels.json').read_text(encoding='utf-8'))['labels']
    type_of = {lab: typ for typ, lab in labels.items()}
    label_list = list(labels.values())
    cases = []
    for lang in ([args.lang] if args.lang else ['de', 'en']):
        rows = [json.loads(l) for l in (ROOT / 'cases' / args.pv / f'sample-{lang}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        if args.limit:
            rows = rows[:args.limit]
        cases += [dict(r, lang=lang) for r in rows]

    t_load = time.perf_counter()
    m = AutoExtractor.from_pretrained(args.model)
    m.to(args.device)
    m.eval()
    load_s = time.perf_counter() - t_load
    runtime = f'gliner2 {version}, torch {torch.__version__}, {args.device}'
    revision = model_revision(args.model)
    m.extract_entities('wake me up at five am', label_list, threshold=args.call_threshold)  # warm-up

    date = dt.date.today().isoformat()
    suffix = (f'-{args.lang}' if args.lang else '') + (f'-{args.tag}' if args.tag else '')
    out_dir = ROOT / 'results' / args.pv / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{date}-run{args.run}{suffix}.jsonl'

    n = errors = span_ok = span_bad = thr_same = thr_diff = 0
    t0 = time.perf_counter()
    with open(out_file, 'w', encoding='utf-8') as out:
        for c in cases:
            cands, err, ms = [], None, None
            try:
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                t = time.perf_counter()
                res = m.extract_entities(c['utt'], label_list, threshold=args.call_threshold, include_confidence=True, include_spans=True)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                ms = round((time.perf_counter() - t) * 1000)
                cands = entities(res, type_of)
                for s in cands:
                    if args.check_spans and 'start' in s:
                        if c['utt'][s['start']:s['end']] == s['text']:
                            span_ok += 1
                        else:
                            span_bad += 1
                if args.check_threshold:
                    direct = entities(m.extract_entities(c['utt'], label_list, threshold=args.threshold, include_confidence=True, include_spans=True), type_of)
                    if sorted(map(ident, direct)) == sorted(map(ident, primary(cands))):
                        thr_same += 1
                    else:
                        thr_diff += 1
            except Exception as e:  # keep going, record the error per row
                err = f'{type(e).__name__}: {e}'[:300]
                errors += 1
            row = {
                'pv': args.pv, 'unit': 'utt', 'pair': None, 'posting': None, 'cv': None,
                'line': f"massive-{c['id']}-{c['lang']}", 'question': 'slots', 'lang': c['lang'],
                'arm': ARM, 'arm_version': args.model, 'arm_revision': revision, 'runtime': runtime, 'host': args.host, 'run': args.run,
                'threshold': args.threshold, 'call_threshold': args.call_threshold, 'latency_ms': ms, 'tokens_in': None, 'cost_usd': 0,
                'ts': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
                'slots': primary(cands),
                'candidates': cands,
            }
            if err:
                row['error'] = err
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
            if n % 200 == 0:
                print(f'{n} utterances, {round(time.perf_counter() - t0)} s', file=sys.stderr, flush=True)
    spans = f', spans text==utt[start:end] {span_ok} of {span_ok + span_bad}' if args.check_spans else ''
    if args.check_threshold:
        spans += f', direct call at {args.threshold} equals filtered candidates in {thr_same} of {thr_same + thr_diff} utterances'
    print(f'{ARM} run {args.run}: {n} utterances, {errors} errors, load {load_s:.1f} s, '
          f'{round(time.perf_counter() - t0)} s{spans} -> {out_file.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
