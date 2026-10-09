#!/usr/bin/env node
/**
 * Freezes the source data of the route1 and ex1 families into cases/. Run
 * once; the frozen files are what every later step reads.
 *
 *   node bench/freeze-route-ex.mjs --massive <dir with de-DE.jsonl and en-US.jsonl> --fastdec <dir with the 17 <domain>.jsonl>
 *   -> cases/route1/massive-de.jsonl, massive-en.jsonl   all MASSIVE 1.1 test rows (id, utt, annot_utt, scenario, intent)
 *      cases/route1/fastdec-en.jsonl                     fast-decisions dev rows, single-label tasks only
 *      cases/ex1/sample-de.jsonl, sample-en.jsonl        1,000 test utterances, same ids in both languages, with gold slots
 *      cases/ex1/slot-types.json                         slot types found in the MASSIVE 1.1 test split (de and en)
 *
 * Inputs: the MASSIVE 1.1 tarball (1.1/data/<locale>.jsonl) and the
 * fast-decisions dataset files at a pinned revision. Hashes, URLs and the
 * revision are recorded in cases/route1/README.md by hand.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseSlots, rng } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const MASSIVE = arg('massive');
const FASTDEC = arg('fastdec');
if (!MASSIVE || !FASTDEC) { console.error('--massive and --fastdec are required'); process.exit(1); }
const ROOT = path.resolve(import.meta.dirname, '..');
const SEED = 20261009;
const SAMPLE = 1000;
const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
const writeJsonl = (f, rows) => { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, rows.map(r => JSON.stringify(r)).join('\n') + '\n'); };
const byId = (a, b) => Number(a.id) - Number(b.id);

// --- MASSIVE: all test rows of de-DE and en-US
const massive = {};
for (const [lang, locale] of [['de', 'de-DE'], ['en', 'en-US']]) {
  massive[lang] = readJsonl(path.join(MASSIVE, `${locale}.jsonl`))
    .filter(r => r.partition === 'test')
    .map(r => ({ id: r.id, utt: r.utt, annot_utt: r.annot_utt, scenario: r.scenario, intent: r.intent }))
    .sort(byId);
}
// de and en must be the same utterances: same ids, same scenario and intent
const en = new Map(massive.en.map(r => [r.id, r]));
if (massive.de.length !== massive.en.length) throw new Error(`row counts differ: de ${massive.de.length}, en ${massive.en.length}`);
let labelMismatch = 0;
for (const r of massive.de) {
  const e = en.get(r.id);
  if (!e) throw new Error(`id ${r.id} missing in en-US`);
  if (e.scenario !== r.scenario || e.intent !== r.intent) labelMismatch += 1;
}
if (labelMismatch) throw new Error(`${labelMismatch} ids with different labels in de and en`);
for (const lang of ['de', 'en']) writeJsonl(path.join(ROOT, 'cases', 'route1', `massive-${lang}.jsonl`), massive[lang]);
console.log(`massive: ${massive.de.length} test rows per language, ids and labels aligned`);

// --- fast-decisions: dev rows, single-label tasks only
const fast = [];
let dropped = 0;
for (const f of fs.readdirSync(FASTDEC).filter(f => f.endsWith('.jsonl')).sort()) {
  const domain = f.replace(/\.jsonl$/, '');
  readJsonl(path.join(FASTDEC, f)).forEach((r, i) => {
    const tasks = [];
    for (const c of r.output.classifications) {
      if (c.multi_label) { dropped += 1; continue; }
      if (c.true_label.length !== 1) throw new Error(`${domain} row ${i} task ${c.task}: single-label task with ${c.true_label.length} gold labels`);
      tasks.push({ task: c.task, labels: c.labels, gold: c.true_label[0] });
    }
    if (tasks.length) fast.push({ id: `${domain}-${i}`, domain, row: i, input: r.input, tasks });
  });
}
writeJsonl(path.join(ROOT, 'cases', 'route1', 'fastdec-en.jsonl'), fast);
console.log(`fast-decisions: ${fast.length} rows, ${fast.reduce((s, r) => s + r.tasks.length, 0)} single-label tasks, ${dropped} multi-label tasks dropped`);

// --- ex1: stratified sample of 1,000 test ids, proportional per scenario (largest remainder)
const counts = {};
for (const r of massive.en) counts[r.scenario] = (counts[r.scenario] || 0) + 1;
const total = massive.en.length;
const quota = Object.fromEntries(Object.entries(counts).map(([s, c]) => [s, Math.floor(c * SAMPLE / total)]));
let left = SAMPLE - Object.values(quota).reduce((a, b) => a + b, 0);
for (const [s] of Object.entries(counts).sort((a, b) => (b[1] * SAMPLE / total % 1) - (a[1] * SAMPLE / total % 1) || a[0].localeCompare(b[0]))) {
  if (left-- <= 0) break;
  quota[s] += 1;
}
const rnd = rng(SEED);
const ids = massive.en.map(r => r.id);
for (let i = ids.length - 1; i > 0; i -= 1) { const j = Math.floor(rnd() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
const taken = {}; const pick = new Set();
for (const id of ids) { const s = en.get(id).scenario; if ((taken[s] || 0) < quota[s]) { pick.add(id); taken[s] = (taken[s] || 0) + 1; } }
const types = new Set();
for (const lang of ['de', 'en']) {
  for (const r of massive[lang]) for (const s of parseSlots(r.annot_utt)) types.add(s.type);
  const rows = massive[lang].filter(r => pick.has(r.id)).map(r => ({ ...r, slots: parseSlots(r.annot_utt) }));
  writeJsonl(path.join(ROOT, 'cases', 'ex1', `sample-${lang}.jsonl`), rows);
}
fs.writeFileSync(path.join(ROOT, 'cases', 'ex1', 'slot-types.json'), JSON.stringify({ source: 'MASSIVE 1.1 test split, de-DE and en-US, all slot types in annot_utt', seed: SEED, sample: SAMPLE, quota, types: [...types].sort() }, null, 2) + '\n');
console.log(`ex1: ${pick.size} ids, ${types.size} slot types, quota ${JSON.stringify(quota)}`);
