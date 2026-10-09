#!/usr/bin/env node
/**
 * What a bench run says before any human reference exists.
 *
 *   node bench/analyze.mjs [--pv pv1] [--base jev] [--md report.md]
 *
 * Per arm:
 *   flip      share of answers that differ from run 1 in the other unshuffled
 *             runs (Jev varies even on identical input)
 *   shuffle   share that differ from run 1 when only the option order changed
 *   agree     share of run-1 answers equal to the base arm's run 1, per question
 *   score     the service's total per pair (bench/match.ts scoreMatch), mean
 *             and standard deviation over the runs, next to the design intent
 *   latency   p50 and p95 per request
 *
 * Accuracy against the truth needs the human reference (reference/<pv>/),
 * which does not exist yet; this report says nothing about who is right.
 */
import fs from 'node:fs';
import path from 'node:path';
import YAML from 'yaml';
import { AXES, scoreMatch, DEFAULT_WEIGHTS } from './match.ts';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'pv1');
const BASE = arg('base', 'jev');
const MD = arg('md', '');
const ROOT = path.resolve(import.meta.dirname, '..');
const RES = path.join(ROOT, 'results', PV);
const pairs = YAML.parse(fs.readFileSync(path.join(ROOT, 'cases', PV, 'pairs.yaml'), 'utf8'));
const pairList = Array.isArray(pairs) ? pairs : pairs.pairs;

/** arm -> run -> rows; the newest date per arm wins. */
function load(arm) {
  const files = fs.readdirSync(path.join(RES, arm)).filter(f => f.endsWith('.jsonl')).sort();
  const latest = files.at(-1).slice(0, 10);
  const runs = {};
  for (const f of files.filter(x => x.startsWith(latest))) {
    const run = Number(f.match(/run(\d+)/)[1]);
    runs[run] = (runs[run] || []).concat(fs.readFileSync(path.join(RES, arm, f), 'utf8').trim().split('\n').map(l => JSON.parse(l)));
  }
  return { date: latest, runs };
}

const key = r => `${r.lang}|${r.line}|${r.question}`;
const pct = x => `${(100 * x).toFixed(1)} %`;
const quant = (xs, q) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };

function scores(rows) {
  const by = new Map(rows.map(r => [key(r), r]));
  const get = (lang, line, q) => by.get(`${lang}|${line}|${q}`);
  const exp = r => r?.probs ? r.probs.reduce((s, p, i) => s + p * i, 0) : 0;
  const out = {};
  for (const p of pairList) {
    for (const lang of ['en', 'de']) {
      const demand = Object.fromEntries(AXES.map(a => [a.key, exp(get(lang, `${p.posting}-project`, `demand:${a.key}`)) / 3]));
      const profile = Object.fromEntries(AXES.map(a => [a.key, exp(get(lang, `${p.cv}-profile`, `depth:${a.key}`)) / 3]));
      const lines = [...new Set(rows.filter(r => r.pair === p.id && r.lang === lang).map(r => r.line))];
      const reqs = lines.map(l => {
        const isReq = get(lang, l, 'is_req')?.probs?.[1] ?? 0;
        const level = exp(get(lang, l, 'level')); // the service uses Jev's score, the expectation, not the argmax
        return { isReq, must: get(lang, l, 'must')?.probs?.[1] ?? 0, axis: get(lang, l, 'axis')?.choice ?? 'other', level, cover: level / 3, counts: isReq >= 0.5 };
      });
      out[`${p.id}|${lang}`] = scoreMatch(reqs, demand, profile, DEFAULT_WEIGHTS).total;
    }
  }
  return out;
}

const arms = fs.readdirSync(RES).filter(a => fs.statSync(path.join(RES, a)).isDirectory());
const data = Object.fromEntries(arms.map(a => [a, load(a)]));
const lines = [];
const say = s => { lines.push(s); console.log(s); };

say(`# ${PV}: run report without reference (${new Date().toISOString().slice(0, 10)})\n`);
say('| arm | date | runs | answers/run | flip vs run 1 | shuffle vs run 1 | agree with ' + BASE + ' | p50 ms | p95 ms |');
say('|---|---|---|---|---|---|---|---|---|');
const questionsSeen = new Set();
for (const [arm, { date, runs }] of Object.entries(data)) {
  const r1 = new Map(runs[1].map(r => [key(r), r]));
  const nums = Object.keys(runs).map(Number);
  const shuffledRun = nums.find(n => runs[n][0]?.shuffled);
  let flips = 0; let flipN = 0; let sh = 0; let shN = 0;
  for (const n of nums.filter(n => n !== 1)) {
    for (const r of runs[n]) {
      const a = r1.get(key(r)); if (!a) continue;
      if (n === shuffledRun) { shN += 1; if (a.choice !== r.choice) sh += 1; } else { flipN += 1; if (a.choice !== r.choice) flips += 1; }
    }
  }
  const base = data[BASE]?.runs[1] ? new Map(data[BASE].runs[1].map(r => [key(r), r])) : null;
  let ag = 0; let agN = 0;
  if (base && arm !== BASE) for (const r of runs[1]) { const b = base.get(key(r)); if (b) { agN += 1; if (b.choice === r.choice) ag += 1; } questionsSeen.add(r.question.split(':')[0]); }
  const lat = runs[1].map(r => r.latency_ms).filter(x => x != null);
  say(`| ${arm} | ${date} | ${nums.length} | ${runs[1].length} | ${flipN ? pct(flips / flipN) : 'n/a'} | ${shN ? pct(sh / shN) : 'n/a'} | ${arm === BASE ? 'base' : agN ? pct(ag / agN) : 'n/a'} | ${lat.length ? quant(lat, 0.5) : ''} | ${lat.length ? quant(lat, 0.95) : ''} |`);
}

say(`\n## Agreement with ${BASE} per question (run 1)\n`);
const qs = ['is_req', 'must', 'axis', 'evidence', 'level', 'demand', 'role', 'depth'];
say(`| arm | ${qs.join(' | ')} |`);
say(`|---|${qs.map(() => '---').join('|')}|`);
for (const [arm, { runs }] of Object.entries(data)) {
  if (arm === BASE || !data[BASE]) continue;
  const base = new Map(data[BASE].runs[1].map(r => [key(r), r]));
  const cells = qs.map(q => {
    const rs = runs[1].filter(r => r.question.split(':')[0] === q && base.has(key(r)));
    return rs.length ? `${pct(rs.filter(r => base.get(key(r)).choice === r.choice).length / rs.length)} (${rs.length})` : 'n/a';
  });
  say(`| ${arm} | ${cells.join(' | ')} |`);
}

say('\n## Total score per pair (mean ± sd over unshuffled runs)\n');
say(`| pair | intent | ${Object.keys(data).flatMap(a => [`${a} en`, `${a} de`]).join(' | ')} |`);
say(`|---|---|${Object.keys(data).flatMap(() => ['---', '---']).join('|')}|`);
const perArm = Object.fromEntries(Object.entries(data).map(([arm, { runs }]) => [arm, Object.entries(runs).filter(([, rs]) => !rs[0]?.shuffled).map(([, rs]) => scores(rs))]));
for (const p of pairList) {
  const cells = Object.keys(data).flatMap(arm => ['en', 'de'].map(lang => {
    const xs = perArm[arm].map(s => s[`${p.id}|${lang}`]).filter(x => x != null);
    return xs.length ? `${mean(xs).toFixed(0)} ± ${sd(xs).toFixed(1)}` : 'n/a';
  }));
  say(`| ${p.id} | ${p.design_intent} | ${cells.join(' | ')} |`);
}
say('\nNo reference judgment yet: agreement with Jev is not accuracy, and a stable answer can be stably wrong.');
if (MD) fs.writeFileSync(path.resolve(MD), lines.join('\n') + '\n');
