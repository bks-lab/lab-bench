#!/usr/bin/env node
/**
 * Scores every arm against a reference judgment.
 *
 *   node bench/score.mjs --ref claude-a [--ref2 claude-b] [--lang de] [--pv pv1] [--md out.md]
 *
 * Only lines the reference calls a requirement are scored on must, axis,
 * evidence and level, the way the service only counts lines with is_req.
 *
 *   correct      share of answers equal to the reference (run 1)
 *   level        exact, and within one step (counts half in research 8)
 *   invented     the arm names a CV entry, the reference says none
 *   missed       the arm says none, the reference names an entry
 *   ref2         the second reference against the first: how far two careful
 *                judges agree, the ceiling for any model
 */
import fs from 'node:fs';
import path from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'pv1');
const LANG = arg('lang', 'de');
const REF = arg('ref');
const REF2 = arg('ref2', '');
const MD = arg('md', '');
const ROOT = path.resolve(import.meta.dirname, '..');
const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
const refFile = j => path.join(ROOT, 'reference', PV, `${j}.${LANG}.jsonl`);

const ref = new Map(readJsonl(refFile(REF)).map(r => [`${r.line}|${r.question}`, r.choice]));
const isReq = line => ref.get(`${line}|is_req`) === 'yes';

function run1(arm) {
  const dir = path.join(ROOT, 'results', PV, arm);
  const files = fs.readdirSync(dir).filter(f => /run1(-\w+)?\.jsonl$/.test(f)).sort();
  const latest = files.at(-1).slice(0, 10);
  return files.filter(f => f.startsWith(latest)).flatMap(f => readJsonl(path.join(dir, f))).filter(r => r.lang === LANG);
}

function score(rows) {
  const by = new Map(rows.map(r => [`${r.line}|${r.question}`, r.choice]));
  const s = { is_req: [0, 0], must: [0, 0], axis: [0, 0], evidence: [0, 0], level: [0, 0], level1: [0, 0], invented: [0, 0], missed: [0, 0], role: [0, 0], demand: [0, 0] };
  const add = (k, ok) => { s[k][1] += 1; if (ok) s[k][0] += 1; };
  for (const [k, want] of ref) {
    const [line, q] = k.split('|');
    const got = by.get(k);
    if (got == null) continue;
    if (q === 'is_req') add('is_req', got === want);
    else if (q === 'role') add('role', got === want);
    else if (q.startsWith('demand:')) add('demand', got === want);
    else if (isReq(line)) {
      if (q === 'level') { add('level', got === want); add('level1', Math.abs(Number(got) - Number(want)) <= 1); }
      else add(q, got === want);
      if (q === 'evidence') {
        if (want === 'none') add('invented', got !== 'none');
        else add('missed', got === 'none');
      }
    }
  }
  return s;
}

const pct = ([a, n]) => (n ? `${(100 * a / n).toFixed(1)} %` : 'n/a');
const out = [];
const say = x => { out.push(x); console.log(x); };
const cols = ['is_req', 'must', 'axis', 'evidence', 'level', 'level1', 'invented', 'missed', 'role', 'demand'];
say(`# ${PV} ${LANG}: arms against reference \`${REF}\`\n`);
say(`| arm | ${cols.map(c => ({ level1: 'level ±1', invented: 'invented evidence ↓', missed: 'missed evidence ↓' }[c] || c)).join(' | ')} |`);
say(`|---|${cols.map(() => '---').join('|')}|`);
const rows = [];
if (REF2) rows.push([`${REF2} (second judge)`, score(readJsonl(refFile(REF2)))]);
for (const arm of fs.readdirSync(path.join(ROOT, 'results', PV)).filter(a => fs.statSync(path.join(ROOT, 'results', PV, a)).isDirectory())) rows.push([arm, score(run1(arm))]);
for (const [name, s] of rows) say(`| ${name} | ${cols.map(c => pct(s[c])).join(' | ')} |`);
const n = [...ref.keys()].filter(k => k.endsWith('|is_req'));
say(`\n${n.length} lines, ${n.filter(k => ref.get(k) === 'yes').length} requirements by the reference, ${[...ref].filter(([k, v]) => k.endsWith('|evidence') && v === 'none').length} of them without evidence. ↓ = lower is better. Run 1 of each arm.`);
if (MD) fs.writeFileSync(path.resolve(MD), out.join('\n') + '\n');
