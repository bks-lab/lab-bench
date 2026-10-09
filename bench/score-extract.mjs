#!/usr/bin/env node
/**
 * Scores every arm of the ex1 family against the MASSIVE gold slots.
 *
 *   node bench/score-extract.mjs [--pv ex1] [--run 1] [--tag smoke] [--md results/ex1/score.md]
 *
 * A slot is the pair (slot type, normalised value text); normalised means
 * lowercase, trimmed, spaces collapsed, trailing punctuation removed. Per
 * utterance gold and prediction are multisets of such pairs; a true positive
 * is a pair in both (counted up to the smaller multiplicity).
 *   P, R, F1     micro over all utterances of an arm and language
 *   exact frame  share of utterances whose predicted multiset equals gold
 *                exactly (an utterance without gold slots counts when the
 *                arm predicts nothing)
 *   errors       rows with a call or parse error (prediction = empty)
 *   p50 ms       median latency per utterance
 * Sensitivity (GLiNER arms only): rows that carry `candidates` with a
 * confidence (run-gliner-extract.py calls at threshold 0.1) are rescored at
 * 0.3, 0.4, 0.5 and 0.6 from the same run. 0.5 is the primary result (the
 * library default, fixed before any run; it must reproduce the main table).
 * The other thresholds are read on the test data itself: a sensitivity
 * check, not tuning, and no threshold may be picked from that table.
 * Baseline "no extraction": predicts nothing, so R = F1 = 0 and P is
 * undefined; its exact-frame rate is the share of utterances without slots.
 * Jev is not an arm here: its API offers choice, score and noul, no
 * extraction (docs.typesafe.ai/introduction).
 * Exits with 1 when no arm has rows, and warns about expected arms without
 * rows.
 */
import fs from 'node:fs';
import path from 'node:path';
import { arms as listArms, runRows, readJsonl } from './lib/results.mjs';
import { normValue, median } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'ex1');
const RUN = Number(arg('run', '1'));
const TAG = arg('tag', '');
const MD = arg('md', '');
const ROOT = path.resolve(import.meta.dirname, '..');

const gold = new Map();
for (const lang of ['de', 'en']) for (const g of readJsonl(path.join(ROOT, 'reference', PV, `gold.${lang}.jsonl`))) gold.set(g.line, g);
const key = s => `${s.type}\u0000${normValue(s.text)}`;
const bag = slots => { const m = new Map(); for (const s of slots) { const k = key(s); m.set(k, (m.get(k) || 0) + 1); } return m; };

const SENS = [0.3, 0.4, 0.5, 0.6];
const PRIMARY = 0.5;
const atThreshold = th => r => (r.candidates || []).filter(s => s.confidence >= th);
const hasCandidates = rows => rows.length && rows.every(r => Array.isArray(r.candidates) && r.call_threshold != null && r.candidates.every(s => typeof s.confidence === 'number'));

function score(rows, lang, pick = r => r.slots || []) {
  const t = { tp: 0, fp: 0, fn: 0, n: 0, exact: 0, errors: 0, lat: [] };
  const per = {};
  const typ = k => k.split('\u0000')[0];
  const add = (ty, f, v) => { (per[ty] ||= { tp: 0, fp: 0, fn: 0 })[f] += v; };
  for (const r of rows) {
    const g = gold.get(r.line);
    if (!g || g.lang !== lang) continue;
    t.n += 1; if (r.error) t.errors += 1; t.lat.push(r.latency_ms);
    const G = bag(g.slots); const P = bag(pick(r));
    let same = G.size === P.size;
    for (const [k, c] of G) { const p = P.get(k) || 0; const hit = Math.min(c, p); t.tp += hit; t.fn += c - hit; add(typ(k), 'tp', hit); add(typ(k), 'fn', c - hit); if (p !== c) same = false; }
    for (const [k, c] of P) { const hit = Math.min(c, G.get(k) || 0); t.fp += c - hit; add(typ(k), 'fp', c - hit); }
    if (same) t.exact += 1;
  }
  return { ...t, per };
}
const prf = ({ tp, fp, fn }) => { const p = tp + fp ? tp / (tp + fp) : null; const r = tp + fn ? tp / (tp + fn) : null; const f = p != null && r != null && p + r ? (2 * p * r) / (p + r) : 0; return { p, r, f }; };
const pct = x => (x == null ? 'n/a' : `${(100 * x).toFixed(1)} %`);

const data = Object.fromEntries(listArms(ROOT, PV).map(a => [a, runRows(ROOT, PV, a, { run: RUN, tag: TAG })]).filter(([, r]) => r.length));
if (!Object.keys(data).length) { console.error(`no rows for ${PV} run ${RUN}${TAG ? ` tag ${TAG}` : ''}: nothing to score`); process.exit(1); }
const missing = PV === 'ex1' ? ['winnow-12b', 'gliner2.5-multi', 'gliner2-large'].filter(a => !data[a]) : [];
if (missing.length) console.error(`warning: no rows for ${missing.join(', ')}`);
const out = [];
const say = x => { out.push(x); console.log(x); };
say(`# ${PV}: slot extraction against MASSIVE gold${TAG ? ` (${TAG} run, not a result)` : ''}\n`);
say(`Run ${RUN} of each arm${TAG ? `, files tagged \`${TAG}\`` : ''}. Jev: not applicable (no extraction primitive).\n`);
const perLang = {};
for (const lang of ['de', 'en']) {
  const res = Object.entries(data).map(([a, rows]) => [a, score(rows, lang)]).filter(([, s]) => s.n);
  if (!res.length) continue;
  perLang[lang] = res;
  say(`## ${lang}\n`);
  say('| arm | utterances | gold slots | predicted | P | R | F1 | exact frame | errors | p50 ms |');
  say('|---|---|---|---|---|---|---|---|---|---|');
  // baseline on the utterances of the first arm (all arms run the same sample)
  const lines = new Set(data[res[0][0]].filter(r => r.lang === lang).map(r => r.line));
  const gl = [...gold.values()].filter(g => g.lang === lang && lines.has(g.line));
  say(`| no extraction (baseline) | ${gl.length} | ${gl.reduce((s, g) => s + g.slots.length, 0)} | 0 | n/a | 0.0 % | 0.0 % | ${pct(gl.filter(g => !g.slots.length).length / gl.length)} | | |`);
  for (const [a, s] of res) {
    const m = prf(s);
    say(`| ${a} | ${s.n} | ${s.tp + s.fn} | ${s.tp + s.fp} | ${pct(m.p)} | ${pct(m.r)} | ${pct(m.f)} | ${pct(s.exact / s.n)} | ${s.errors} | ${median(s.lat) ?? 'n/a'} |`);
  }
  say('');
}
for (const [lang, res] of Object.entries(perLang)) {
  const types = {};
  for (const [, s] of res) for (const [ty, c] of Object.entries(s.per)) types[ty] = Math.max(types[ty] || 0, c.tp + c.fn);
  const list = Object.entries(types).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  say(`## F1 per slot type, ${lang}\n`);
  say(`| slot type | gold | ${res.map(([a]) => a).join(' | ')} |`);
  say(`|---|---|${res.map(() => '---').join('|')}|`);
  for (const [ty, n] of list) say(`| ${ty} | ${n} | ${res.map(([, s]) => (s.per[ty] ? pct(prf(s.per[ty]).f) : '')).join(' | ')} |`);
  say('');
}
{
  const sens = Object.entries(data).filter(([, rows]) => hasCandidates(rows));
  for (const [a, rows] of sens) {
    const bad = rows.filter(r => r.threshold !== PRIMARY || r.call_threshold > Math.min(...SENS) || JSON.stringify(atThreshold(r.threshold)(r).map(key).sort()) !== JSON.stringify((r.slots || []).map(key).sort()));
    if (bad.length) console.error(`warning: ${a}: ${bad.length} rows where slots are not the candidates at the primary threshold ${PRIMARY}, or the call threshold is above ${Math.min(...SENS)}`);
  }
  if (sens.length) {
    say('## GLiNER threshold sensitivity\n');
    say(`Same run, rescored from the stored candidates (call threshold ${[...new Set(sens.flatMap(([, rows]) => rows.map(r => r.call_threshold)))].join(', ')}). **${PRIMARY} is the primary result**: the library default, fixed before any run, and identical to the tables above. ${SENS.filter(t => t !== PRIMARY).join(', ')} are a sensitivity check read on the test data itself, not tuning; no threshold may be chosen from this table.\n`);
    say('| lang | arm | threshold | role | predicted | P | R | F1 | exact frame |');
    say('|---|---|---|---|---|---|---|---|---|');
    for (const lang of ['de', 'en']) for (const [a, rows] of sens) for (const th of SENS) {
      const s = score(rows, lang, atThreshold(th));
      if (!s.n) continue;
      const m = prf(s);
      const role = th === PRIMARY ? '**primary**' : 'sensitivity (test data)';
      say(`| ${lang} | ${a} | ${th.toFixed(1)} | ${role} | ${s.tp + s.fp} | ${pct(m.p)} | ${pct(m.r)} | ${pct(m.f)} | ${pct(s.exact / s.n)} |`);
    }
    say('');
  }
}
if (MD) { fs.mkdirSync(path.dirname(path.resolve(MD)), { recursive: true }); fs.writeFileSync(path.resolve(MD), out.join('\n') + '\n'); }
