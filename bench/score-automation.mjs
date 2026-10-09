#!/usr/bin/env node
/**
 * T1 (plan/t1.md): automation rate and calibration of every arm, from the
 * existing run 1 rows of route1 and pv1. No new runs.
 *
 *   node bench/score-automation.mjs [--pv route1|pv1|all]
 *
 * Writes results/<pv>/automation.md and results/<pv>/automation.json (read
 * by bench/toolmap.mjs). Per arm, language and item set:
 *   acc        share right; an unanswered row counts as wrong
 *   auto95/90  in-sample automation rate: the largest share of items with
 *              confidence >= t (over every distinct confidence t, ties never
 *              split) whose accuracy is >= 95 % / 90 %. Optimistic: the
 *              threshold is chosen on the rows it is scored on
 *   cf95/90    2-fold cross-fitted: threshold chosen on one half (split by
 *              line, mulberry32 seed 20261009), applied to the other; mean
 *              coverage of the two halves, accuracy of all passed items
 *   ECE        10 equal-width confidence bins, answered rows only
 *   Brier      multi-class, sum over offered options, answered rows only
 * Confidence = the probability the arm gives its own answer. Unanswered rows
 * never pass a threshold and stay in the denominator.
 * Baseline: most frequent answer per option set; its confidence is that
 * label's in-sample share in the option set.
 */
import fs from 'node:fs';
import path from 'node:path';
import { arms as listArms, runRows, readJsonl } from './lib/results.mjs';
import { sourceOf, rng, median } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const WHICH = arg('pv', 'all');
const ROOT = path.resolve(import.meta.dirname, '..');
const SEED = 20261009;
const TARGETS = [0.95, 0.9];
const LANGS = ['de', 'en'];
export const ENGLISH_ONLY = a => /^gliner2\.5-decide/.test(a);
export const famOf = a => (a === 'jev' ? 'jev' : a.startsWith('gliner') ? 'gliner' : a.startsWith('classic') ? 'classic' : 'local-llm');

// ------------------------------------------------------------ metrics

/** In-sample threshold: {t, coverage, acc} of the largest passing set, t null if none. */
export function bestThreshold(items, target) {
  const s = items.filter(x => x.conf != null).sort((a, b) => b.conf - a.conf);
  let n = 0, right = 0, best = { t: null, passed: 0, right: 0 };
  for (let i = 0; i < s.length; i++) {
    n += 1; if (s[i].right) right += 1;
    if (i + 1 < s.length && s[i + 1].conf === s[i].conf) continue; // never split ties
    if (right / n >= target && n > best.passed) best = { t: s[i].conf, passed: n, right };
  }
  return { ...best, coverage: items.length ? best.passed / items.length : 0, acc: best.passed ? best.right / best.passed : null };
}

/** Split lines into two halves by a seeded shuffle of the sorted ids. */
export function halves(items, seed = SEED) {
  const lines = [...new Set(items.map(x => x.line))].sort();
  const r = rng(seed);
  for (let i = lines.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [lines[i], lines[j]] = [lines[j], lines[i]]; }
  const a = new Set(lines.slice(0, Math.floor(lines.length / 2)));
  return [items.filter(x => a.has(x.line)), items.filter(x => !a.has(x.line))];
}

export function crossFit(items, target) {
  const [A, B] = halves(items);
  let covSum = 0, passed = 0, right = 0;
  for (const [fit, apply] of [[A, B], [B, A]]) {
    const { t } = bestThreshold(fit, target);
    const p = t == null ? [] : apply.filter(x => x.conf != null && x.conf >= t);
    covSum += apply.length ? p.length / apply.length : 0;
    passed += p.length; right += p.filter(x => x.right).length;
  }
  return { coverage: covSum / 2, acc: passed ? right / passed : null, passed };
}

export function calibration(items) {
  const ans = items.filter(x => x.conf != null);
  const bins = Array.from({ length: 10 }, (_, i) => ({ lo: i / 10, hi: (i + 1) / 10, n: 0, conf: 0, right: 0 }));
  let brier = 0, brierN = 0;
  for (const x of ans) {
    const b = bins[Math.min(9, Math.floor(x.conf * 10))];
    b.n += 1; b.conf += x.conf; if (x.right) b.right += 1;
    if (x.probs) { brier += x.probs.reduce((s, p, i) => s + (p - (x.options[i] === x.gold ? 1 : 0)) ** 2, 0); brierN += 1; }
  }
  let ece = 0;
  for (const b of bins) if (b.n) ece += (b.n / ans.length) * Math.abs(b.right / b.n - b.conf / b.n);
  return {
    answered: ans.length, ece: ans.length ? ece : null, brier: brierN ? brier / brierN : null,
    bins: bins.map(b => ({ bin: `${b.lo.toFixed(1)}-${b.hi.toFixed(1)}`, n: b.n, mean_conf: b.n ? +(b.conf / b.n).toFixed(4) : null, acc: b.n ? +(b.right / b.n).toFixed(4) : null })),
  };
}

const r4 = x => (x == null ? null : +x.toFixed(4));
export function metrics(items) {
  const out = { n: items.length, acc: r4(items.filter(x => x.right).length / items.length) };
  for (const t of TARGETS) {
    const k = Math.round(t * 100);
    const b = bestThreshold(items, t);
    const c = crossFit(items, t);
    Object.assign(out, {
      [`auto${k}`]: r4(b.coverage), [`auto${k}_threshold`]: b.t, [`auto${k}_acc`]: r4(b.acc),
      [`cf${k}`]: r4(c.coverage), [`cf${k}_acc`]: r4(c.acc),
    });
  }
  const cal = calibration(items);
  Object.assign(out, { answered: cal.answered, ece: r4(cal.ece), brier: r4(cal.brier), reliability: cal.bins });
  const mass = items.map(x => x.mass).filter(x => x != null).sort((a, b) => a - b);
  if (mass.length) Object.assign(out, { mass_median: r4(median(mass)), mass_p10: r4(mass[Math.floor(mass.length * 0.1)]) });
  return out;
}

/** Most frequent answer per option set, confidence = its in-sample share. */
function baselineItems(golds) {
  const sets = {};
  for (const g of golds) { const s = (sets[g.optionSet] ||= {}); s[g.gold] = (s[g.gold] || 0) + 1; }
  const top = {};
  for (const [k, s] of Object.entries(sets)) {
    const n = Object.values(s).reduce((a, b) => a + b, 0);
    const [label, c] = Object.entries(s).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0];
    top[k] = { label, share: c / n };
  }
  return golds.map(g => ({ line: g.line, conf: top[g.optionSet].share, right: top[g.optionSet].label === g.gold }));
}

// ------------------------------------------------------------ item sets

function route1Setup() {
  const gold = new Map();
  for (const lang of LANGS) for (const g of readJsonl(path.join(ROOT, 'reference', 'route1', `gold.${lang}.jsonl`))) gold.set(`${g.line}|${g.question}`, g);
  const opts = new Map();
  for (const l of fs.readFileSync(path.join(ROOT, 'requests', 'route1', 'requests.jsonl'), 'utf8').trim().split('\n')) {
    const r = JSON.parse(l);
    for (const [qid, q] of Object.entries(r.request.questions)) opts.set(`${r.line}|${qid}`, Object.keys(q.criteria).length);
  }
  const part = g => {
    if (sourceOf(g.line) === 'fastdec') return 'fastdec, all single-label tasks';
    if (g.question === 'scenario') return 'massive scenario';
    return opts.get(`${g.line}|intent`) > 1 ? 'massive intent, 2+ options' : null; // 1-option intent left out
  };
  const optionSet = g => (sourceOf(g.line) === 'massive' && g.question === 'intent' ? `intent|${g.choice.split('_')[0]}` : g.question);
  const golds = [...gold.values()].filter(part).map(g => ({ key: `${g.line}|${g.question}`, line: g.line, lang: g.lang, part: part(g), optionSet: optionSet(g), gold: g.choice }));
  return { parts: ['massive scenario', 'massive intent, 2+ options', 'fastdec, all single-label tasks'], golds };
}

function pv1Setup() {
  const golds = [];
  for (const [lang, ref] of [['de', 'adjudicated-a'], ['en', 'claude-c']]) {
    const rows = readJsonl(path.join(ROOT, 'reference', 'pv1', `${ref}.${lang}.jsonl`));
    const isReq = new Set(rows.filter(r => r.question === 'is_req' && r.choice === 'yes').map(r => r.line));
    for (const r of rows) {
      if (!['is_req', 'must', 'axis', 'evidence', 'level'].includes(r.question)) continue;
      if (r.question !== 'is_req' && !isReq.has(r.line)) continue;
      // evidence: options differ per line, "none" is the one shared answer, so the option set is the question
      golds.push({ key: `${r.line}|${r.question}`, line: r.line, lang, part: r.question, optionSet: r.question, gold: r.choice });
    }
  }
  return { parts: ['is_req', 'must', 'axis', 'evidence', 'level'], golds, refs: { de: 'adjudicated-a', en: 'claude-c' } };
}

function armItems(pv, arm, golds) {
  const rows = new Map(runRows(ROOT, pv, arm, { run: 1 }).map(r => [`${r.line}|${r.question}`, r]));
  const out = [];
  for (const g of golds) {
    const r = rows.get(g.key);
    if (!r) continue;
    const answered = r.choice != null && Array.isArray(r.probs);
    const conf = answered ? r.probs[r.options.indexOf(r.choice)] : null;
    out.push({ ...g, conf, right: answered && r.choice === g.gold, probs: answered ? r.probs : null, options: r.options, mass: r.mass ?? null, notArgmax: answered && conf < Math.max(...r.probs) });
  }
  return out;
}

// ------------------------------------------------------------ report

const pct = x => (x == null ? 'n/a' : `${(100 * x).toFixed(1)} %`);
const f3 = x => (x == null ? 'n/a' : x.toFixed(3));

function score(pv) {
  const { parts, golds, refs } = pv === 'route1' ? route1Setup() : pv1Setup();
  const armNames = listArms(ROOT, pv).filter(a => runRows(ROOT, pv, a, { run: 1 }).length);
  const res = { pv, design: 'plan/t1.md', seed: SEED, targets: TARGETS, references: refs || { de: 'gold', en: 'gold' }, langs: {} };
  const md = [];
  const say = x => md.push(x);
  say(`# ${pv}: automation rate and calibration (T1)\n`);
  say(`Design: \`plan/t1.md\`. Run 1 of every arm. ${pv === 'pv1' ? 'Line questions only; German rows against `adjudicated-a`, English rows against `claude-c`; `must`, `axis`, `evidence`, `level` only on lines the reference calls a requirement.' : 'Gold labels; MASSIVE intent items with a single option left out.'} Generated by \`node bench/score-automation.mjs --pv ${pv}\`.\n`);
  say('Columns: `auto95` / `auto90` = in-sample automation rate (largest share of items, ranked by the arm\'s probability for its own answer, whose accuracy stays at or above 95 % / 90 %; the threshold is chosen on the same rows, so this is optimistic). `cf95` / `cf90` = 2-fold cross-fitted (threshold chosen on one half of the lines, seed 20261009, applied to the other; mean coverage, and in brackets the accuracy of the passed items). `ECE` over 10 equal-width bins and multi-class `Brier`, both on answered rows. `mass` = median letter mass before renormalising (local LLM only). Unanswered rows count as wrong and are never automated. `baseline` = most frequent answer per option set, confidence = its in-sample share.\n');
  say('The probabilities are distributions over the offered options (GLiNER: a softmax over labels; local LLM: next-token letter probabilities renormalised over the letters), not calibrated claims that the answer is right.\n');
  const notArgmax = {};
  for (const lang of LANGS) {
    const L = (res.langs[lang] = { sets: {} });
    const langGolds = golds.filter(g => g.lang === lang);
    const sets = [['pooled', langGolds], ...parts.map(p => [p, langGolds.filter(g => g.part === p)])].filter(([, gs]) => gs.length);
    for (const [name, gs] of sets) {
      const S = (L.sets[name] = { n: gs.length, baseline: metrics(baselineItems(gs)), arms: {} });
      delete S.baseline.reliability; delete S.baseline.ece; delete S.baseline.brier; delete S.baseline.answered;
      for (const a of armNames) {
        const items = armItems(pv, a, gs);
        if (!items.length) continue;
        if (name === 'pooled') notArgmax[`${a} ${lang}`] = items.filter(x => x.notArgmax).length;
        S.arms[a] = { family: famOf(a), english_only_extra: lang === 'de' && ENGLISH_ONLY(a), ...metrics(items) };
      }
      say(`## ${name === 'pooled' ? `pooled (${pv === 'route1' ? (lang === 'en' ? 'scenario, intent 2+ options, fastdec' : 'scenario, intent 2+ options') : 'all five line questions'})` : name}, ${lang}\n`);
      say('| arm | family | n | acc | auto95 | auto90 | cf95 (acc) | cf90 (acc) | ECE | Brier | mass |');
      say('|---|---|---|---|---|---|---|---|---|---|---|');
      const order = Object.entries(S.arms).sort((x, y) => y[1].auto95 - x[1].auto95 || y[1].auto90 - x[1].auto90 || (x[0] < y[0] ? -1 : 1));
      for (const [a, m] of order) say(`| ${a}${m.english_only_extra ? ' (English model, extra)' : ''} | ${m.family} | ${m.n} | ${pct(m.acc)} | ${pct(m.auto95)} | ${pct(m.auto90)} | ${pct(m.cf95)} (${pct(m.cf95_acc)}) | ${pct(m.cf90)} (${pct(m.cf90_acc)}) | ${f3(m.ece)} | ${f3(m.brier)} | ${m.mass_median ?? ''} |`);
      const b = S.baseline;
      say(`| baseline: most frequent answer | | ${b.n} | ${pct(b.acc)} | ${pct(b.auto95)} | ${pct(b.auto90)} | ${pct(b.cf95)} (${pct(b.cf95_acc)}) | ${pct(b.cf90)} (${pct(b.cf90_acc)}) | | | |`);
      say('');
    }
    // reliability data: best arm per family on the pooled set (English-only models not eligible on de)
    const P = L.sets.pooled;
    say(`## reliability, pooled, ${lang}: best arm per family by auto95\n`);
    for (const fam of ['jev', 'gliner', 'local-llm', 'classic']) {
      const cand = Object.entries(P.arms).filter(([, m]) => m.family === fam && !m.english_only_extra).sort((x, y) => y[1].auto95 - x[1].auto95 || y[1].auto90 - x[1].auto90 || (x[0] < y[0] ? -1 : 1));
      if (!cand.length) continue;
      const [a, m] = cand[0];
      say(`${fam}: \`${a}\`\n`);
      say('| confidence bin | n | mean confidence | accuracy |');
      say('|---|---|---|---|');
      for (const b of m.reliability) if (b.n) say(`| ${b.bin} | ${b.n} | ${b.mean_conf.toFixed(3)} | ${pct(b.acc)} |`);
      say('');
    }
  }
  const na = Object.entries(notArgmax).filter(([, v]) => v);
  say(`Rows where the arm's answer is not its most probable option (confidence = probability of the answer): ${na.length ? na.map(([k, v]) => `${k}: ${v}`).join(', ') : 'none'}.\n`);
  fs.writeFileSync(path.join(ROOT, 'results', pv, 'automation.md'), md.join('\n'));
  fs.writeFileSync(path.join(ROOT, 'results', pv, 'automation.json'), JSON.stringify(res, null, 1) + '\n');
  console.log(md.join('\n'));
}

if (import.meta.url === `file://${process.argv[1]}`) for (const pv of WHICH === 'all' ? ['route1', 'pv1'] : [WHICH]) score(pv);
