#!/usr/bin/env node
/**
 * Builds results/toolmap.json: one row per task, one cell per tool family
 * (jev, gliner, local-llm).
 *
 *   node bench/toolmap.mjs [--out results/toolmap.json] [--plan plan/toolmap-plan.yaml]
 *
 * Measured cells come from the score reports (results/route1/score.md,
 * results/ex1/score.md, results/pv1/score-*.md), the pv1 references and the
 * result rows (versions, dates, Jev tokens). pv1 accuracies are recounted
 * from the rows and must equal the report, otherwise the script exits 1.
 *
 * The register plan/toolmap-plan.yaml adds what no measurement can say: the
 * tools (maker, how they run, licence, sources), the state of cells that are
 * not measured (not_applicable with a reason, open), and rows that are
 * planned before any measurement, each pointing to its dated test design
 * under plan/. Every cell ends up with a `state`:
 *   measured        a value from a run, see source_report
 *   planned         a written, committed test design exists (`plan`), no run yet
 *   open            a sensible test nobody has designed yet
 *   not_applicable  the tool cannot do the task (`reason`)
 *
 * Needs the result rows of every arm, Jev included. Without them it stops
 * with a message instead of writing a partial map.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parse as parseYaml } from 'yaml';
import { readJsonl, arms, runRows } from './lib/results.mjs';
import { mcnemar } from './lib/stats.mjs';
import { sourceOf } from './lib/route-ex.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const OUT = arg('out', 'results/toolmap.json');
const PLAN = arg('plan', 'plan/toolmap-plan.yaml');

const FAMILIES = ['jev', 'gliner', 'local-llm'];
const STATES = ['measured', 'planned', 'open', 'not_applicable'];

// relative paths are repo-relative, absolute ones stay as given
const abs = f => path.resolve(ROOT, f);
const rd = f => fs.readFileSync(abs(f), 'utf8');
const rdj = f => readJsonl(abs(f));
/** p as the reports print it: three decimals, "<0.001" when that would read 0.000. */
const fmtP = p => { const s = p.toFixed(3); return s === '0.000' ? '<0.001' : Number(s); };
/**
 * Exact paired McNemar of two arms on the items both have, each given as
 * Map(key -> right?). b = only the first arm right, c = only the second.
 */
function paired(first, second) {
  let b = 0, c = 0, n = 0;
  for (const [k, okA] of first) {
    if (!second.has(k)) continue;
    n += 1; const okB = second.get(k);
    if (okA && !okB) b += 1; if (!okA && okB) c += 1;
  }
  return { n, b, c, p: mcnemar(b, c) };
}
/**
 * Top family of a row and each cell's test against it. Top = highest value
 * (ties: more items right, then FAMILIES order). Every cell gets top_family;
 * the others also mcnemar_vs_top_p (exact, two-sided, between this family's
 * best arm and the top family's best arm on the items both answered) and the
 * discordant counts. `items` maps family -> Map(key -> right?).
 */
function testAgainstTop(cells, items) {
  const fams = FAMILIES.filter(f => cells[f] && items[f]);
  if (!fams.length) return;
  const right = f => [...items[f].values()].filter(Boolean).length;
  const top = [...fams].sort((x, y) => (cells[y].value - cells[x].value) || (right(y) - right(x)) || (FAMILIES.indexOf(x) - FAMILIES.indexOf(y)))[0];
  for (const f of fams) {
    cells[f].top_family = top;
    if (f === top) { cells[f].mcnemar_vs_top_p = null; continue; }
    const t = paired(items[top], items[f]);
    Object.assign(cells[f], { mcnemar_vs_top_p: fmtP(t.p), vs_top_paired: t.n, vs_top_only_top_right: t.b, vs_top_only_arm_right: t.c });
  }
}
const pnum = s => { const m = String(s).match(/-?[\d.]+/); return m ? Number(m[0]) : null; };
const med = a => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const fail = msg => { console.error(`toolmap: ${msg}`); process.exit(1); };

// ---------------------------------------------------------------- inputs

for (const pv of ['route1', 'pv1']) {
  if (!runRows(ROOT, pv, 'jev').length) fail(`no Jev result rows in results/${pv}/jev/. The map needs every arm's rows (versions, dates, tokens); a public export without Jev rows ships the committed results/toolmap.json instead.`);
}

/** Markdown report -> { section title: [ {column: cell} ] } */
function sections(md) {
  const out = {}; let cur = null;
  for (const line of md.split('\n')) {
    if (line.startsWith('## ')) { cur = line.slice(3).trim(); out[cur] = []; continue; }
    if (cur && line.startsWith('|') && !line.startsWith('|---')) out[cur].push(line.split('|').slice(1, -1).map(c => c.trim()));
  }
  for (const k of Object.keys(out)) { const [h, ...rows] = out[k]; out[k] = rows.map(r => Object.fromEntries(h.map((c, i) => [c, r[i]]))); }
  return out;
}

/** Versions, dates, Jev tokens per arm for route1 and ex1; pv1 baselines, versions and dates. */
function computeStats() {
  const out = {};
  for (const pv of ['route1', 'ex1']) {
    out[pv] = {};
    for (const arm of arms(ROOT, pv)) {
      const rows = runRows(ROOT, pv, arm, { run: 1 });
      const groups = {};
      for (const r of rows) { const g = `${r.line.split('-')[0]}-${r.lang}`; (groups[g] ||= []).push(r); }
      const o = {
        versions: [...new Set(rows.map(r => r.arm_version))],
        digests: [...new Set(rows.map(r => r.arm_digest).filter(Boolean))],
        revisions: [...new Set(rows.map(r => r.arm_revision).filter(Boolean))],
        runtime: [...new Set(rows.map(r => r.runtime).filter(Boolean))],
        ts: [rows.map(r => r.ts).sort()[0]],
        groups: {},
      };
      if (arm === 'jev') {
        for (const [g, rs] of Object.entries(groups)) {
          const t = med(rs.map(r => r.tokens_in).filter(x => x != null));
          o.groups[g] = { med_tokens_in: t, usd_per_1000_decisions: +(t * 0.042 / 1e6 * 1000).toFixed(4) };
        }
      }
      out[pv][arm] = o;
    }
  }
  // pv1 baseline: most frequent reference answer per question, scored like bench/score.mjs
  out.pv1_base = {};
  for (const [lang, ref] of [['de', 'adjudicated-a'], ['en', 'claude-c']]) {
    const rows = rdj(`reference/pv1/${ref}.${lang}.jsonl`);
    const req = new Set(rows.filter(r => r.question === 'is_req' && r.choice === 'yes').map(r => r.line));
    const res = {};
    for (const q of ['is_req', 'must', 'axis', 'evidence', 'level']) {
      const rs = rows.filter(r => r.question === q && (q === 'is_req' || req.has(r.line)));
      const cnt = {}; for (const r of rs) cnt[r.choice] = (cnt[r.choice] || 0) + 1;
      const [top, k] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0];
      // evidence: per line the options differ; "none" is the one shared answer
      const kk = q === 'evidence' ? (cnt.none || 0) : k;
      res[q] = { n: rs.length, answer: q === 'evidence' ? 'none' : top, correct: kk, pct: +(100 * kk / rs.length).toFixed(1) };
    }
    out.pv1_base[lang] = res;
  }
  // pv1 versions and dates: the plain run 1 file of each arm
  out.pv1 = {};
  for (const arm of arms(ROOT, 'pv1')) {
    const dir = path.join(ROOT, 'results', 'pv1', arm);
    const rows = fs.readdirSync(dir).filter(f => /run1\.jsonl$/.test(f)).flatMap(f => readJsonl(path.join(dir, f)));
    out.pv1[arm] = { versions: [...new Set(rows.map(r => r.arm_version))], date: [...new Set(rows.map(r => (r.ts || '').slice(0, 10)))] };
  }
  return out;
}

const STATS = computeStats();
const route = sections(rd('results/route1/score.md'));
const ex = sections(rd('results/ex1/score.md'));
const R1 = 'results/route1/score.md', E1 = 'results/ex1/score.md';
const ver = (pv, arm) => {
  const s = STATS[pv][arm];
  const base = s.versions.join(', ');
  if (s.digests.length) return `${base} (Ollama digest ${s.digests[0].slice(0, 12)}, ${s.runtime.join(', ')})`;
  if (s.revisions.length) return `${base} (revision ${s.revisions[0].slice(0, 12)})`;
  return base;
};
const date = (pv, arm) => STATS[pv][arm].ts[0].slice(0, 10);
const famOf = a => a === 'jev' ? 'jev' : a.startsWith('gliner') ? 'gliner' : 'local-llm';
const EXTRA = ' (English model, extra)';
const cleanArm = a => a.replace(EXTRA, '');

// ---------------------------------------------------------------- route1 items

// Same item definition as bench/score-route.mjs: gold items of the group the
// arm has a row for, an unanswered row counting as wrong. The accuracy
// recounted from these items must equal the report, otherwise the script stops.
const routeGold = new Map();
for (const lang of ['de', 'en']) for (const g of rdj(`reference/route1/gold.${lang}.jsonl`)) routeGold.set(`${g.line}|${g.question}`, g);
const routeOpts = new Map();
for (const r of rdj('requests/route1/requests.jsonl')) for (const [qid, q] of Object.entries(r.request.questions)) routeOpts.set(`${r.line}|${qid}`, Object.keys(q.criteria).length);
function routeGroups(g) {
  if (sourceOf(g.line) === 'massive') {
    const out = [`massive ${g.question}`];
    if (g.question === 'intent' && routeOpts.get(`${g.line}|intent`) > 1) out.push('massive intent, 2+ options');
    return out;
  }
  const out = ['fastdec, all single-label tasks', `fastdec ${g.question}`];
  if (routeOpts.get(`${g.line}|${g.question}`) <= 26) out.push('fastdec, tasks with 26 options or fewer');
  return out;
}
const routeRunCache = {};
/** Map(key -> right?) of one route1 arm on one report section ("massive scenario, de"). */
function routeItems(arm, section) {
  const i = section.lastIndexOf(', ');
  const group = section.slice(0, i), lang = section.slice(i + 2);
  routeRunCache[arm] ||= runRows(ROOT, 'route1', arm, { run: 1 });
  const out = new Map();
  for (const r of routeRunCache[arm]) {
    const k = `${r.line}|${r.question}`; const g = routeGold.get(k);
    if (!g || g.lang !== lang || !routeGroups(g).includes(group)) continue;
    out.set(k, r.choice != null && r.choice === g.choice);
  }
  return out;
}

// ---------------------------------------------------------------- measured rows

function routeRow(id, task, lang, section, extraKeys = {}) {
  const rows = route[section];
  const row = { id, kind: 'route', task, lang, metric: 'accuracy', source_report: R1, section, cells: {} };
  const items = {};
  for (const fam of FAMILIES) {
    let cand = rows.filter(r => famOf(cleanArm(r.arm)) === fam);
    const extra = cand.filter(r => r.arm.includes(EXTRA));
    cand = cand.filter(r => !r.arm.includes(EXTRA));
    if (!cand.length) continue; // no eligible arm: the register has to say why
    const best = Math.max(...cand.map(r => pnum(r.acc)));
    const top = cand.filter(r => pnum(r.acc) === best);
    const r = top[0];
    const arm = cleanArm(r.arm);
    const cell = {
      best_arm: arm, tied_with: top.slice(1).map(x => cleanArm(x.arm)), arms_compared: cand.length,
      value: pnum(r.acc), macro_f1: pnum(r['macro-F1']), baseline: pnum(r.base), baseline_name: 'most frequent answer',
      skill: pnum(r.skill), n: pnum(r.n), answered: pnum(r.answered),
      mcnemar_vs_jev_p: fam === 'jev' ? null : (r['p (McNemar)'] === '0.000' ? '<0.001' : pnum(r['p (McNemar)'])),
      p50_ms_per_decision: pnum(r['p50 ms']),
      measured: date('route1', arm), model_version: ver('route1', arm), source_report: R1,
    };
    if (fam === 'jev') {
      const g = STATS.route1.jev.groups[`${section.startsWith('fastdec') ? 'fastdec' : 'massive'}-${lang}`];
      cell.median_input_tokens_per_decision = g.med_tokens_in;
      cell.usd_per_1000_decisions = g.usd_per_1000_decisions;
      cell.cost_note = 'median input tokens per call x USD 0.042 per million input tokens, output free (docs.typesafe.ai/models, read 2026-10-02); per source and language, not per question group';
    }
    if (extra.length) {
      const eb = Math.max(...extra.map(x => pnum(x.acc)));
      const et = extra.filter(x => pnum(x.acc) === eb);
      const e = et[0];
      cell.english_model_extra = { arm: cleanArm(e.arm), tied_with: et.slice(1).map(x => cleanArm(x.arm)), value: pnum(e.acc), skill: pnum(e.skill), note: 'English-only model on German rows, labelled extra in the report, not eligible as best arm' };
    }
    items[fam] = routeItems(arm, section);
    const recount = +(100 * [...items[fam].values()].filter(Boolean).length / items[fam].size).toFixed(1);
    if (recount !== cell.value || items[fam].size !== cell.n) fail(`${id} ${fam}: recount ${recount} % of ${items[fam].size} items differs from the report (${cell.value} % of ${cell.n})`);
    row.cells[fam] = cell;
  }
  testAgainstTop(row.cells, items);
  return Object.assign(row, extraKeys);
}

/** ex1: GLiNER and local LLM only; the Jev cell comes from the register (not_applicable). */
function exRow(id, lang) {
  const rows = ex[lang];
  const base = rows.find(r => r.arm.startsWith('no extraction'));
  const baseFrame = pnum(base['exact frame']);
  const row = { id, kind: 'extract', task: 'slot extraction (MASSIVE slots)', lang, metric: 'micro F1, threshold 0.5 for GLiNER', source_report: E1, section: lang, cells: {} };
  for (const fam of ['gliner', 'local-llm']) {
    const cand = rows.filter(r => !r.arm.startsWith('no extraction') && famOf(r.arm) === fam);
    if (!cand.length) continue; // no arm of this family: the register has to say why
    const best = Math.max(...cand.map(r => pnum(r.F1)));
    const top = cand.filter(r => pnum(r.F1) === best);
    const r = top[0];
    const frame = pnum(r['exact frame']);
    row.cells[fam] = {
      best_arm: r.arm, tied_with: top.slice(1).map(x => x.arm), arms_compared: cand.length,
      value: pnum(r.F1), precision: pnum(r.P), recall: pnum(r.R), baseline: 0, baseline_name: 'no extraction',
      skill: +(pnum(r.F1) / 100).toFixed(3), skill_note: 'baseline F1 is 0, so skill equals F1',
      exact_frame: frame, exact_frame_baseline: baseFrame,
      exact_frame_skill: +(((frame - baseFrame) / (100 - baseFrame))).toFixed(3),
      n: pnum(r.utterances), gold_slots: pnum(r['gold slots']), p50_ms_per_utterance: pnum(r['p50 ms']),
      measured: date('ex1', r.arm), model_version: ver('ex1', r.arm), source_report: E1,
    };
  }
  return row;
}

// pv1: recount from the rows (same rule as bench/score.mjs), check against the report
const PV1 = {
  de: {
    ref: 'adjudicated-a', report: 'results/pv1/score-de-adjudicated.md',
    note: 'adjudicated-a: blind Opus judge claude-a, cross-checked by blind Sonnet and Fable judges, every disagreement adjudicated (results/pv1/crosscheck.md). A model judgment; no human sample yet. The English rows use claude-c, a single Opus judge that was not cross-checked.',
  },
  en: {
    ref: 'claude-c', report: 'results/pv1/score-en.md',
    note: 'claude-c: one blind Opus judge (claude-opus-5-5), not cross-checked. The German reference adjudicated-a was cross-checked by Sonnet and Fable judges with an adjudication of every disagreement (results/pv1/crosscheck.md). Both references are model judgments; no human sample yet.',
  },
};
const PV1_VERSION_NOTE = 'pv1 result rows record the model name only, no revision or digest (route1 and ex1 rows do).';
const PV1_LOCAL = ['qwen3-32b', 'qwen3.8-27b', 'shisa-de-1', 'winnow-12b'];
const PV1_QUESTIONS = ['is_req', 'must', 'axis', 'evidence', 'level'];

/**
 * Per arm and question: [right, n] and the items as Map(key -> right?). Items
 * are the reference keys of that question (requirement lines only, except
 * is_req). A missing answer counts as wrong, as in route1. bench/score.mjs
 * leaves missing answers out of n instead, so an arm with missing answers
 * fails the comparison with the report below, with the reason named.
 */
function pv1Counts(lang) {
  const ref = new Map(rdj(`reference/pv1/${PV1[lang].ref}.${lang}.jsonl`).map(r => [`${r.line}|${r.question}`, r.choice]));
  const isReq = l => ref.get(`${l}|is_req`) === 'yes';
  const out = {};
  for (const arm of arms(ROOT, 'pv1')) {
    const rows = runRows(ROOT, 'pv1', arm, { run: 1 }).filter(r => r.lang === lang);
    const by = new Map(rows.map(r => [`${r.line}|${r.question}`, r.choice]));
    const s = Object.fromEntries(PV1_QUESTIONS.map(q => [q, [0, 0]]));
    const items = Object.fromEntries(PV1_QUESTIONS.map(q => [q, new Map()]));
    let missing = 0;
    for (const [k, want] of ref) {
      const [line, q] = k.split('|');
      if (!s[q] || (q !== 'is_req' && !isReq(line))) continue;
      const got = by.get(k);
      if (got == null) missing += 1;
      const ok = got != null && got === want;
      s[q][1]++; if (ok) s[q][0]++;
      items[q].set(k, ok);
    }
    out[arm] = { s, items, missing };
  }
  return out;
}

const pv1Rows = [];
const mism = [];
for (const lang of ['de', 'en']) {
  const all = pv1Counts(lang);
  const counts = Object.fromEntries(Object.entries(all).map(([a, x]) => [a, x.s]));
  const rep = sections('## t\n' + rd(PV1[lang].report).split('\n').filter(l => l.startsWith('|')).join('\n')).t;
  for (const [arm, s] of Object.entries(counts)) for (const q of Object.keys(s)) {
    const r = rep.find(x => x.arm === arm);
    if (!r || +(100 * s[q][0] / s[q][1]).toFixed(1) !== pnum(r[q])) mism.push(`${lang} ${arm} ${q}${all[arm].missing ? ` (${all[arm].missing} missing answers count as wrong here, bench/score.mjs leaves them out)` : ''}`);
  }
  const base = STATS.pv1_base[lang];
  for (const q of PV1_QUESTIONS) {
    const row = { id: `pv1-${q}-${lang}`, kind: 'match', task: `pv1 ${q}`, lang, metric: 'accuracy against the blind reference', reference: PV1[lang].ref, source_report: PV1[lang].report, cells: {} };
    const items = {};
    for (const fam of FAMILIES) {
      const english = a => a.startsWith('gliner2.5-decide');
      const famArms = Object.keys(counts).filter(a => fam === 'jev' ? a === 'jev' : fam === 'gliner' ? a.startsWith('gliner') : PV1_LOCAL.includes(a));
      const cand = lang === 'de' && fam === 'gliner' ? famArms.filter(a => !english(a)) : famArms;
      const extra = lang === 'de' && fam === 'gliner' ? famArms.filter(english) : [];
      if (!cand.length) continue; // no eligible arm: the register has to say why
      const val = a => counts[a][q][0] / counts[a][q][1];
      const best = Math.max(...cand.map(val));
      const top = cand.filter(a => val(a) === best);
      const a = top[0]; const [k, n] = counts[a][q];
      const b = base[q].correct / base[q].n;
      const st = STATS.pv1[a];
      row.cells[fam] = {
        best_arm: a, tied_with: top.slice(1), arms_compared: cand.length,
        value: +(100 * k / n).toFixed(1), baseline: base[q].pct, baseline_name: `most frequent answer (${base[q].answer})`,
        skill: +(((k / n) - b) / (1 - b)).toFixed(3), n,
        measured: st.date[0], model_version: st.versions.join(', '), source_report: PV1[lang].report,
        baseline_source: 'computed from reference/pv1 (same rule as results/pv1/gliner.md)',
      };
      if (extra.length) {
        const eb = Math.max(...extra.map(val)); const et = extra.filter(x => val(x) === eb);
        row.cells[fam].english_model_extra = { arm: et[0], tied_with: et.slice(1), value: +(100 * eb).toFixed(1), skill: +((eb - b) / (1 - b)).toFixed(3), note: 'English-only model on German rows, not eligible as best arm' };
      }
      items[fam] = all[a].items[q];
      // same test as results/pv1/significance-<lang>.md, kept here so every row carries it
      if (fam !== 'jev' && all.jev) {
        const t = paired(all.jev.items[q], items[fam]);
        Object.assign(row.cells[fam], { mcnemar_vs_jev_p: fmtP(t.p), vs_jev_only_jev_right: t.b, vs_jev_only_arm_right: t.c });
      } else if (fam !== 'jev') row.cells[fam].mcnemar_vs_jev_p = null;
    }
    testAgainstTop(row.cells, items);
    row.reference_note = PV1[lang].note;
    row.version_note = PV1_VERSION_NOTE;
    pv1Rows.push(row);
  }
}
if (mism.length) fail(`pv1 recount differs from the report: ${mism.join(', ')}`);

const measuredRows = [
  routeRow('route1-scenario-de', 'route: MASSIVE scenario (18 classes)', 'de', 'massive scenario, de'),
  routeRow('route1-scenario-en', 'route: MASSIVE scenario (18 classes)', 'en', 'massive scenario, en'),
  routeRow('route1-intent2-de', 'route: MASSIVE intent given the gold scenario, 2+ options', 'de', 'massive intent, 2+ options, de'),
  routeRow('route1-intent2-en', 'route: MASSIVE intent given the gold scenario, 2+ options', 'en', 'massive intent, 2+ options, en'),
  routeRow('route1-fastdec26-en', 'route: fast-decisions dev, single-label tasks with 26 labels or fewer', 'en', 'fastdec, tasks with 26 options or fewer, en', {
    note: 'Home ground for GLiNER: Fastino built fast-decisions, and its dataset card calls it the classification suite behind GLiNER2.5-Decide. Whether these dev rows were used in training is not stated, so a Decide lead or tie here is not a neutral result.',
  }),
  exRow('ex1-slots-de', 'de'),
  exRow('ex1-slots-en', 'en'),
  ...pv1Rows,
];

// ---------------------------------------------------------------- register

const plan = parseYaml(rd(PLAN));
for (const f of FAMILIES) if (!plan.tools?.[f]) fail(`${PLAN}: tools.${f} missing`);

/** A cell from the register: state first, then its fields, checked. */
function registerCell(where, c) {
  if (!STATES.includes(c?.state)) fail(`${where}: state must be one of ${STATES.join(', ')}`);
  if (c.state === 'measured') fail(`${where}: a measured cell comes from a run, not from the register`);
  if (c.state === 'not_applicable' && !c.reason) fail(`${where}: not_applicable needs a reason`);
  if (c.state === 'planned') {
    if (!c.plan) fail(`${where}: planned needs plan`);
    if (!fs.existsSync(path.join(ROOT, c.plan))) fail(`${where}: plan file ${c.plan} does not exist`);
  }
  return { ...c };
}

const rows = measuredRows.map(r => {
  const over = plan.cells?.[r.id] || {};
  const cells = {};
  for (const fam of FAMILIES) {
    if (r.cells[fam] && over[fam]) fail(`${PLAN}: cells.${r.id}.${fam} overrides a measured cell`);
    if (r.cells[fam]) cells[fam] = { state: 'measured', ...r.cells[fam] };
    else if (over[fam]) cells[fam] = registerCell(`${PLAN} cells.${r.id}.${fam}`, over[fam]);
    else fail(`${r.id}: no measurement and no register entry for ${fam}`);
  }
  return { ...r, cells };
});
for (const id of Object.keys(plan.cells || {})) if (!rows.some(r => r.id === id)) fail(`${PLAN}: cells.${id} matches no measured row`);

for (const p of plan.rows || []) {
  if (rows.some(r => r.id === p.id)) fail(`${PLAN}: row ${p.id} already exists as a measured row`);
  const cells = {};
  for (const fam of FAMILIES) cells[fam] = registerCell(`${PLAN} rows.${p.id}.${fam}`, p.cells?.[fam]);
  rows.push({ ...p, cells });
}

// ---------------------------------------------------------------- write

const out = {
  schema: 'jev-match-bench toolmap v2',
  generated: new Date().toISOString().slice(0, 10),
  generated_from: ['results/route1/score.md', 'results/ex1/score.md', 'results/pv1/score-de-adjudicated.md', 'results/pv1/score-en.md', 'reference/pv1/adjudicated-a.de.jsonl', 'reference/pv1/claude-c.en.jsonl', 'result rows for versions, dates and Jev tokens', PLAN],
  findings: { route1: 'results/route1/findings.md', ex1: 'results/ex1/findings.md', pv1: 'results/pv1/gliner.md' },
  note: 'Each cell is one task and one tool family. "best_arm" is the arm of that family with the highest value on the same test data the value is measured on, so it is chosen after the fact and flatters the families with many arms (GLiNER has up to six arms in route1 and nine in pv1, the local family has one arm in route1 and ex1 and four in pv1). Cells are per task: a family that is best on one task says nothing about another. All numbers are run 1, one run per arm. For German rows, English-only GLiNER models (GLiNER2.5-Decide, GLiNER2.5-Decide-1B) are not eligible as best arm, in route1 and pv1 alike; their best value is listed as english_model_extra. pv1 rows: baseline computed from the reference with the rule of results/pv1/gliner.md, run dates 2026-10-01 (Jev, local) and 2026-10-08 (GLiNER). skill = (value - baseline) / (100 - baseline). p50 values in route1 rows are per question group (the p50 ms column of results/route1/score.md); the Speed table of results/route1/findings.md gives medians per source and language, so the two can differ by a few ms. Latencies are not one clock: Jev is an HTTP round trip to api.typesafe.ai, the local LLM an HTTP call over the LAN, GLiNER GPU time in-process. Local cost (hardware, power) was not estimated. Tests: route1 and pv1 cells carry top_family (the family with the highest value in the row) and mcnemar_vs_top_p, an exact two-sided McNemar between this family\'s best arm and the top family\'s best arm on the same items (unanswered counts as wrong); route1 and pv1 cells other than Jev also carry mcnemar_vs_jev_p. Both compared arms are picked after the fact as the best of their family, and no p value is corrected for that or for the number of tests, so p >= 0.05 means the difference is not established, not that the two are level. ex1 cells have no test; their skill is F1 against a baseline F1 of 0, exact_frame_skill compares whole frames against extracting nothing.',
  states: plan.states,
  families: {
    jev: 'TypeSafe Jev, hosted API (api.typesafe.ai)',
    gliner: 'Fastino GLiNER2 / GLiNER2.5 encoders, local GPU (RTX 4090)',
    'local-llm': 'open LLM through Ollama, local GPU (RTX 4090)',
  },
  tools: plan.tools,
  rows,
};
fs.writeFileSync(abs(OUT), JSON.stringify(out, null, 1) + '\n');
for (const r of out.rows) console.log(r.id.padEnd(22), Object.entries(r.cells).map(([f, c]) => c.state === 'measured' ? `${f}:${c.best_arm} ${c.value} (b ${c.baseline}, s ${c.skill})${c.tied_with.length ? ' tie ' + c.tied_with : ''}` : `${f}:${c.state}`).join(' | '));
console.log(`wrote ${OUT}`);
