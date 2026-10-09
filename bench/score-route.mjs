#!/usr/bin/env node
/**
 * Scores every arm of the route1 family against the gold labels.
 *
 *   node bench/score-route.mjs [--pv route1] [--run 1] [--base jev] [--tag smoke] [--md results/route1/score.md]
 *
 * Per arm, source (massive, fastdec), language and question:
 *   n          gold items the arm was asked (rows present in its run file)
 *   answered   rows with an answer (local arm: none for questions over 26 options)
 *   acc        share equal to gold; an unanswered row counts as wrong
 *   macro-F1   mean F1 over the labels seen in gold or prediction
 *   base       "most frequent answer": per question (per option set), the
 *              most frequent gold label among the same rows. For the
 *              MASSIVE intent question the option set is the gold scenario,
 *              for fast-decisions the task
 *   skill      (acc - base) / (1 - base)
 *   vs base    exact two-sided McNemar against the --base arm on the items
 *              both arms have a row for, an unanswered row counting as
 *              wrong (as in acc): b = only base right, c = only arm right
 *   p50 ms     median latency per call. Every arm makes one call per
 *              question; only Jev rows written by a joint call (no
 *              `call_scope: "question"`, run files before 2026-10-09 fixes)
 *              are counted once per request
 * Groups: `fastdec, tasks with 26 options or fewer` leaves out
 * support_intent:intent (28 options), which the local arm cannot be asked;
 * it is the like-for-like fast-decisions comparison.
 * Jev cost: median input tokens per request (summed over its calls) x USD
 * 0.042 per million input tokens (docs.typesafe.ai/models, read 2026-10-02),
 * output free.
 * Exits with 1 when no arm has rows, and warns about expected arms without
 * rows, so a forgotten copy from the GPU machine does not pass as a report.
 */
import fs from 'node:fs';
import path from 'node:path';
import { mcnemar } from './lib/stats.mjs';
import { arms as listArms, runRows, readJsonl } from './lib/results.mjs';
import { sourceOf, median } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'route1');
const RUN = Number(arg('run', '1'));
const BASE = arg('base', 'jev');
const TAG = arg('tag', '');
const MD = arg('md', '');
const ROOT = path.resolve(import.meta.dirname, '..');
const USD_PER_M_IN = 0.042;
const EN_ONLY = new Set(['fastino/GLiNER2.5-Decide', 'fastino/GLiNER2.5-Decide-1B']);

const gold = new Map();
for (const lang of ['de', 'en']) for (const g of readJsonl(path.join(ROOT, 'reference', PV, `gold.${lang}.jsonl`))) gold.set(`${g.line}|${g.question}`, g);
// option count per intent question, to report the 2+ option subset
const reqOpts = new Map();
for (const l of fs.readFileSync(path.join(ROOT, 'requests', PV, 'requests.jsonl'), 'utf8').trim().split('\n')) {
  const r = JSON.parse(l);
  for (const [qid, q] of Object.entries(r.request.questions)) reqOpts.set(`${r.line}|${qid}`, Object.keys(q.criteria).length);
}

// group of a gold item: the rows of the report; optionSet: the unit of the baseline
function groups(g) {
  const src = sourceOf(g.line);
  if (src === 'massive') {
    const out = [`massive ${g.question}`];
    if (g.question === 'intent' && reqOpts.get(`${g.line}|intent`) > 1) out.push('massive intent, 2+ options');
    return out;
  }
  const out = ['fastdec, all single-label tasks', `fastdec ${g.question}`];
  if (reqOpts.get(`${g.line}|${g.question}`) <= 26) out.push('fastdec, tasks with 26 options or fewer');
  return out;
}
const optionSet = g => (sourceOf(g.line) === 'massive' && g.question === 'intent' ? `intent|${g.choice.split('_')[0]}` : g.question);

function loadArm(arm) {
  const rows = runRows(ROOT, PV, arm, { run: RUN, tag: TAG });
  return new Map(rows.map(r => [`${r.line}|${r.question}`, r]));
}

const armNames = listArms(ROOT, PV);
const data = Object.fromEntries(armNames.map(a => [a, loadArm(a)]).filter(([, m]) => m.size));
const order = [BASE, ...Object.keys(data).filter(a => a !== BASE)].filter(a => data[a]);
if (!order.length) { console.error(`no rows for ${PV} run ${RUN}${TAG ? ` tag ${TAG}` : ''}: nothing to score`); process.exit(1); }
const EXPECTED = ['jev', 'winnow-12b', 'gliner2.5-multi-decide', 'gliner2.5-multi-decide-intext', 'gliner2.5-decide', 'gliner2.5-decide-intext', 'gliner2.5-decide-1b', 'gliner2.5-decide-1b-intext'];
const missing = PV === 'route1' ? EXPECTED.filter(a => !data[a]) : [];
if (missing.length) console.error(`warning: no rows for ${missing.join(', ')}`);

function stats(arm, lang, group) {
  const rows = data[arm];
  const items = [];
  for (const [k, r] of rows) {
    const g = gold.get(k);
    if (!g || g.lang !== lang || !groups(g).includes(group)) continue;
    items.push({ k, g, r });
  }
  if (!items.length) return null;
  const n = items.length;
  const answered = items.filter(x => x.r.choice != null).length;
  const right = items.filter(x => x.r.choice === x.g.choice).length;
  // baseline: most frequent gold label per option set, on these rows
  const sets = {};
  for (const { g } of items) { const s = (sets[optionSet(g)] ||= {}); s[g.choice] = (s[g.choice] || 0) + 1; }
  const baseRight = Object.values(sets).reduce((a, s) => a + Math.max(...Object.values(s)), 0);
  // macro-F1 over labels in gold or prediction (labels prefixed by question)
  const tp = {}; const fp = {}; const fn = {};
  for (const { g, r } of items) {
    const gl = `${g.question}|${g.choice}`; const pl = r.choice != null ? `${g.question}|${r.choice}` : null;
    if (pl === gl) tp[gl] = (tp[gl] || 0) + 1;
    else { fn[gl] = (fn[gl] || 0) + 1; if (pl) fp[pl] = (fp[pl] || 0) + 1; }
  }
  const labels = new Set([...Object.keys(tp), ...Object.keys(fp), ...Object.keys(fn)]);
  let f1sum = 0;
  for (const l of labels) { const t = tp[l] || 0; const d = 2 * t + (fp[l] || 0) + (fn[l] || 0); f1sum += d ? (2 * t) / d : 0; }
  // McNemar against the base arm
  let b = 0; let c = 0; let paired = 0;
  if (arm !== BASE && data[BASE]) {
    for (const { k, g, r } of items) {
      const rb = data[BASE].get(k);
      if (!rb) continue;
      paired += 1;
      const okB = rb.choice != null && rb.choice === g.choice; const okA = r.choice != null && r.choice === g.choice;
      if (okB && !okA) b += 1; if (!okB && okA) c += 1;
    }
  }
  // latency: once per call; only joint Jev calls cover several rows
  const joint = x => x.r.arm === 'jev' && x.r.call_scope !== 'question';
  const lat = [
    ...new Map(items.filter(joint).map(x => [x.r.line, x.r.latency_ms])).values(),
    ...items.filter(x => !joint(x)).map(x => x.r.latency_ms),
  ];
  const acc = right / n; const base = baseRight / n;
  return { n, answered, acc, f1: f1sum / labels.size, base, skill: base < 1 ? (acc - base) / (1 - base) : null, b, c, paired, p50: median(lat), version: items[0].r.arm_version };
}

const pct = x => (x == null ? 'n/a' : `${(100 * x).toFixed(1)} %`);
const out = [];
const say = x => { out.push(x); console.log(x); };
say(`# ${PV}: arms against the gold labels${TAG ? ` (${TAG} run, not a result)` : ''}\n`);
say(`Run ${RUN} of each arm${TAG ? `, files tagged \`${TAG}\`` : ''}. Base for McNemar: \`${BASE}\`${data[BASE] ? '' : ' (no rows, McNemar columns left empty)'}. Unanswered rows count as wrong.\n`);
const allGroups = ['massive scenario', 'massive intent', 'massive intent, 2+ options', 'fastdec, all single-label tasks', 'fastdec, tasks with 26 options or fewer'];
for (const lang of ['de', 'en']) {
  for (const group of allGroups) {
    const rowsOut = order.map(a => [a, stats(a, lang, group)]).filter(([, s]) => s);
    if (!rowsOut.length) continue;
    say(`## ${group}, ${lang}\n`);
    say(`| arm | n | answered | acc | macro-F1 | base | skill | only ${BASE} right | only arm right | p (McNemar) | p50 ms |`);
    say('|---|---|---|---|---|---|---|---|---|---|---|');
    for (const [a, s] of rowsOut) {
      const extra = lang === 'de' && EN_ONLY.has(s.version) ? ' (English model, extra)' : '';
      const mc = a === BASE || !data[BASE] ? '| | | ' : `| ${s.b} | ${s.c} | ${mcnemar(s.b, s.c).toFixed(3)} `;
      say(`| ${a}${extra} | ${s.n} | ${s.answered} | ${pct(s.acc)} | ${pct(s.f1)} | ${pct(s.base)} | ${s.skill == null ? 'n/a' : s.skill.toFixed(3)} ${mc}| ${s.p50 ?? 'n/a'} |`);
    }
    say('');
  }
}

// per fast-decisions task: accuracy only
const tasks = [...new Set([...gold.values()].filter(g => sourceOf(g.line) === 'fastdec').map(g => g.question))].sort();
const taskRows = tasks.map(t => [t, order.map(a => stats(a, 'en', `fastdec ${t}`))]).filter(([, ss]) => ss.some(Boolean));
if (taskRows.length) {
  say('## fastdec per task, en (accuracy; base in the last column)\n');
  say(`| task | ${order.join(' | ')} | base |`);
  say(`|---|${order.map(() => '---').join('|')}|---|`);
  for (const [t, ss] of taskRows) say(`| ${t} | ${ss.map(s => (s ? pct(s.acc) : '')).join(' | ')} | ${pct(ss.find(Boolean).base)} |`);
  say('');
}

// Jev cost
if (data.jev) {
  say('## Jev cost\n');
  say('| source | lang | requests | median input tokens | USD per 1,000 requests |');
  say('|---|---|---|---|---|');
  // input tokens per request: summed over its calls (a joint call counts once)
  const byReq = new Map();
  for (const r of data.jev.values()) {
    const e = byReq.get(r.line) || { line: r.line, lang: r.lang, tokens: 0, seen: false };
    if (r.call_scope === 'question') e.tokens += r.tokens_in ?? 0;
    else if (!e.seen) e.tokens += r.tokens_in ?? 0;
    e.seen = true;
    byReq.set(r.line, e);
  }
  for (const src of ['massive', 'fastdec']) for (const lang of ['de', 'en']) {
    const rs = [...byReq.values()].filter(r => sourceOf(r.line) === src && r.lang === lang);
    if (!rs.length) continue;
    const m = median(rs.map(r => r.tokens));
    say(`| ${src} | ${lang} | ${rs.length} | ${m} | ${m == null ? 'n/a' : (m * USD_PER_M_IN / 1e6 * 1000).toFixed(4)} |`);
  }
  say(`\nUSD ${USD_PER_M_IN} per million input tokens, output free (docs.typesafe.ai/models, read 2026-10-02).`);
}
if (MD) { fs.mkdirSync(path.dirname(path.resolve(MD)), { recursive: true }); fs.writeFileSync(path.resolve(MD), out.join('\n') + '\n'); }
