/**
 * One row per answered question, the same shape for every arm, so any arm can
 * be laid next to Jev (README, output format). Options are always listed in
 * the order the question defined them, never in the order a shuffled run sent
 * them, so rows of different runs compare position by position.
 */
export function optionsOf(q) {
  if (q.type === 'noul') return ['no', 'yes'];
  if (q.type === 'score') return (q.criteria || []).map((_, i) => String(i));
  return Object.keys(q.criteria || {});
}

export function rowsFor(req, answer, meta) {
  const out = [];
  for (const [qid, q] of Object.entries(req.request.questions)) {
    const a = answer?.answers?.[qid];
    const options = optionsOf(q);
    let probs = null; let choice = null;
    if (a?.type === 'noul' || (q.type === 'noul' && typeof a?.noul === 'number')) {
      probs = [1 - a.noul, a.noul].map(x => Math.round(x * 1e4) / 1e4);
      choice = a.noul >= 0.5 ? 'yes' : 'no';
    } else if (a) {
      probs = a.probabilities ? options.map(o => a.probabilities[o] ?? 0) : null;
      choice = q.type === 'score' ? String(Math.round(a.score)) : a.choice;
      if (q.type === 'score' && probs) choice = options[probs.indexOf(Math.max(...probs))];
    }
    out.push({
      pv: req.pv, unit: req.unit, pair: req.pair, posting: req.posting, cv: req.cv, line: req.line,
      question: qid, qtype: q.type, lang: req.lang, ...meta,
      options, probs, choice, expected: q.type === 'score' && a?.score != null ? a.score : undefined,
      error: answer?.error,
    });
  }
  return out;
}

/** Deterministic shuffle (mulberry32), so a shuffled run can be repeated. */
export function shuffled(arr, seed) {
  let t = seed >>> 0;
  const rnd = () => { t += 0x6d2b79f5; let x = t; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i -= 1) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/** The request with the options of every choice question in shuffled order. */
export function withShuffledOptions(request, seed) {
  const questions = {};
  for (const [qid, q] of Object.entries(request.questions)) {
    if (q.type === 'choice') {
      const keys = shuffled(Object.keys(q.criteria), seed + qid.length);
      questions[qid] = { ...q, criteria: Object.fromEntries(keys.map(k => [k, q.criteria[k]])) };
    } else questions[qid] = q;
  }
  return { ...request, questions };
}
