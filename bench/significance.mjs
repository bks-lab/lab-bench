#!/usr/bin/env node
/**
 * Paired significance of every arm against Jev, per question, on run 1.
 *
 *   node bench/significance.mjs --ref claude-a [--lang de] [--pv pv1] [--base jev] [--md out.md]
 *
 * Same lines and the same definitions as bench/score.mjs: a line counts for an
 * arm when its answer matches the reference (for invented and missed: when the
 * error occurs). For every arm and question the lines where only the base is
 * right (b) and only the arm is right (c) feed an exact two-sided binomial
 * McNemar test, p = min(1, 2 * P(X <= min(b, c))) with X ~ Bin(b + c, 0.5).
 * For invented and missed, "right" means the error does NOT occur.
 */
import fs from 'node:fs';
import path from 'node:path';
import { mcnemar } from './lib/stats.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'pv1');
const LANG = arg('lang', 'de');
const REF = arg('ref');
const BASE = arg('base', 'jev');
const MD = arg('md', '');
const ROOT = path.resolve(import.meta.dirname, '..');
const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

const ref = new Map(readJsonl(path.join(ROOT, 'reference', PV, `${REF}.${LANG}.jsonl`)).map(r => [`${r.line}|${r.question}`, r.choice]));
const isReq = line => ref.get(`${line}|is_req`) === 'yes';

function run1(arm) {
  const dir = path.join(ROOT, 'results', PV, arm);
  const files = fs.readdirSync(dir).filter(f => /run1(-\w+)?\.jsonl$/.test(f)).sort();
  const latest = files.at(-1).slice(0, 10);
  const rows = files.filter(f => f.startsWith(latest)).flatMap(f => readJsonl(path.join(dir, f))).filter(r => r.lang === LANG);
  return new Map(rows.map(r => [`${r.line}|${r.question}`, r.choice]));
}

// Per metric: which reference keys count, and whether an answer is "right".
const METRICS = {
  is_req: { keys: (l, q) => q === 'is_req', ok: (g, w) => g === w },
  must: { keys: (l, q) => q === 'must' && isReq(l), ok: (g, w) => g === w },
  axis: { keys: (l, q) => q === 'axis' && isReq(l), ok: (g, w) => g === w },
  evidence: { keys: (l, q) => q === 'evidence' && isReq(l), ok: (g, w) => g === w },
  level: { keys: (l, q) => q === 'level' && isReq(l), ok: (g, w) => g === w },
  'level ±1': { keys: (l, q) => q === 'level' && isReq(l), ok: (g, w) => Math.abs(Number(g) - Number(w)) <= 1 },
  'invented evidence': { keys: (l, q, w) => q === 'evidence' && isReq(l) && w === 'none', ok: g => g === 'none' },
  'missed evidence': { keys: (l, q, w) => q === 'evidence' && isReq(l) && w !== 'none', ok: g => g !== 'none' },
};

if (!fs.existsSync(path.join(ROOT, 'results', PV, BASE))) {
  // e.g. the public export, which ships no Jev rows: nothing to pair against
  console.warn(`warning: no rows for the base arm ${BASE} in results/${PV}/, skipped. Choose another base with --base <arm>.`);
  process.exit(0);
}
const base = run1(BASE);
const arms = fs.readdirSync(path.join(ROOT, 'results', PV))
  .filter(a => a !== BASE && fs.statSync(path.join(ROOT, 'results', PV, a)).isDirectory());

const out = [];
const say = x => { out.push(x); console.log(x); };
say(`# ${PV} ${LANG}: paired McNemar, each arm against \`${BASE}\`, reference \`${REF}\`\n`);
say(`| question | n | arm | ${BASE} right | arm right | only ${BASE} right | only arm right | p (exact, two-sided) |`);
say('|---|---|---|---|---|---|---|---|');
for (const [name, m] of Object.entries(METRICS)) {
  const keys = [...ref].filter(([k, w]) => { const [l, q] = k.split('|'); return m.keys(l, q, w); });
  for (const arm of arms) {
    const a = run1(arm);
    let x = 0, y = 0, b = 0, c = 0, n = 0;
    for (const [k, w] of keys) {
      const gb = base.get(k), ga = a.get(k);
      if (gb == null || ga == null) continue;
      n += 1;
      const okB = m.ok(gb, w), okA = m.ok(ga, w);
      if (okB) x += 1;
      if (okA) y += 1;
      if (okB && !okA) b += 1;
      if (!okB && okA) c += 1;
    }
    say(`| ${name} | ${n} | ${arm} | ${x} | ${y} | ${b} | ${c} | ${mcnemar(b, c).toFixed(3)} |`);
  }
}
say(`\nRun 1 of each arm. Only lines both arms answered. For invented and missed evidence, "right" means the error does not occur, so "${BASE} right" counts the lines where ${BASE} does not invent (or does not miss).`);
if (MD) fs.writeFileSync(path.resolve(MD), out.join('\n') + '\n');
