#!/usr/bin/env python3
"""
Classic baseline for route1 MASSIVE: sentence embeddings plus logistic
regression, trained on the MASSIVE 1.1 train split of the same locale.
Design: plan/t3.md.

  python bench/run-classic.py --massive-dir <dir with de-DE.jsonl, en-US.jsonl> \
      [--tarball-sha256 <hex>] [--shots 0|10] [--date YYYY-MM-DD] [--limit N]

--shots 0 trains on the full train split (arm classic-e5-lr), --shots 10 on
10 rows per class (arm classic-e5-lr-10shot). Writes
results/route1/<arm>/<date>-run1.jsonl (with --limit: -smoke suffix).
"""
import argparse, datetime, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import sklearn, sentence_transformers, torch
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

MODEL = 'intfloat/multilingual-e5-base'
REVISION = 'd128750597153bb5987e10b1c3493a34e5a4502a'
PREFIX = 'query: '
SEED = 20261009
LOCALES = {'de': 'de-DE.jsonl', 'en': 'en-US.jsonl'}
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--massive-dir', required=True)
ap.add_argument('--tarball-sha256', default=None)
ap.add_argument('--shots', type=int, default=0)
ap.add_argument('--date', default=datetime.date.today().isoformat())
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

ARM = 'classic-e5-lr' if a.shots == 0 else f'classic-e5-lr-{a.shots}shot'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
runtime = (f'python {platform.python_version()}, torch {torch.__version__}, sentence-transformers '
           f'{sentence_transformers.__version__}, scikit-learn {sklearn.__version__}, {device}')
model = SentenceTransformer(MODEL, revision=REVISION, device=device)


def embed(texts, bs=64):
    return model.encode([PREFIX + t for t in texts], batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def sample(rows, key, k):
    """k rows per class of `key`, seeded, without replacement; all rows of a smaller class."""
    rng = np.random.default_rng(SEED)
    out = []
    for c in sorted({r[key] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[key] == c]
        pick = idx if len(idx) <= k else sorted(rng.choice(idx, size=k, replace=False).tolist())
        out += [rows[i] for i in pick]
    return out


def fit(rows, key, emb_of):
    X = np.stack([emb_of[r['id']] for r in rows]); y = [r[key] for r in rows]
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)
    if clf.n_iter_.max() >= 2000: print(f'warning: {key} classifier hit max_iter', file=sys.stderr)
    return clf


reqs = [json.loads(l) for l in (ROOT / 'requests/route1/requests.jsonl').read_text().splitlines() if l.strip()]
reqs = [r for r in reqs if r['line'].startswith('massive-')]
out_dir = ROOT / 'results/route1' / ARM
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f'{a.date}-run1{"-smoke" if a.limit else ""}.jsonl'
fh = out_file.open('w')
for lang, fname in LOCALES.items():
    raw = (Path(a.massive_dir) / fname).read_text().splitlines()
    train = [json.loads(l) for l in raw if l.strip()]
    train = [{'id': r['id'], 'utt': r['utt'], 'scenario': r['scenario'], 'intent': r['intent']}
             for r in train if r['partition'] == 'train']
    train_sha = hashlib.sha256('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in train).encode()).hexdigest()
    sets = {'scenario': train, 'intent': train} if a.shots == 0 else \
           {'scenario': sample(train, 'scenario', a.shots), 'intent': sample(train, 'intent', a.shots)}
    need = {r['id']: r['utt'] for s in sets.values() for r in s}
    ids = sorted(need)
    t0 = time.time(); E = embed([need[i] for i in ids]); emb_of = dict(zip(ids, E))
    clfs = {q: fit(rows, q, emb_of) for q, rows in sets.items()}
    print(f'{lang}: {len(train)} train rows, used {[len(s) for s in sets.values()]}, fit {time.time() - t0:.0f} s', file=sys.stderr)
    train_info = {'source': 'MASSIVE 1.1 partition train', 'file': f'1.1/data/{fname}', 'rows_in_split': len(train),
                  'rows_used': {q: len(s) for q, s in sets.items()}, 'rows_sha256': train_sha,
                  'tarball_sha256': a.tarball_sha256, 'shots_per_class': a.shots or None,
                  'seed': SEED if a.shots else None}
    lreqs = [r for r in reqs if r['lang'] == lang]
    if a.limit: lreqs = lreqs[:a.limit]
    # warm-up, not timed
    for _ in range(3): clfs['scenario'].predict_proba(embed([lreqs[0]['text']], bs=1))
    for r in lreqs:
        utt = r['request']['state']['utterance']
        for q, spec in r['request']['questions'].items():
            opts = list(spec['criteria'].keys())
            clf = clfs[q]; classes = list(clf.classes_)
            t = time.perf_counter()
            p = clf.predict_proba(embed([utt], bs=1))[0]
            ms = round((time.perf_counter() - t) * 1000)
            full = dict(zip(classes, p))
            unknown = [o for o in opts if o not in full]
            if unknown and a.shots == 0: sys.exit(f'{r["line"]} {q}: options unknown to the classifier: {unknown}')
            sub = np.array([full.get(o, 0.0) for o in opts])
            sub = sub / sub.sum() if sub.sum() > 0 else np.full(len(opts), 1 / len(opts))
            row = {'pv': 'route1', 'unit': 'utt', 'pair': None, 'posting': None, 'cv': None, 'line': r['line'],
                   'question': q, 'qtype': 'choice', 'lang': lang, 'arm': ARM,
                   'arm_version': f'{MODEL} + sklearn LogisticRegression(C=1.0, multinomial)',
                   'arm_revision': REVISION, 'runtime': runtime, 'host': 'local', 'run': 1, 'shuffled': False,
                   'latency_ms': ms, 'tokens_in': None, 'cost_usd': 0,
                   'ts': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
                   'train': train_info, 'options': opts, 'probs': [round(float(x), 4) for x in sub],
                   'choice': opts[int(np.argmax(sub))]}
            if q == 'intent': row['choice_flat'] = classes[int(np.argmax(p))]
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{lang}: wrote {len(lreqs)} requests', file=sys.stderr)
fh.close()
print(f'wrote {out_file}', file=sys.stderr)
