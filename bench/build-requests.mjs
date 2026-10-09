#!/usr/bin/env node
/**
 * Turns the frozen cases of a bench version into the exact requests the
 * service would send, one JSON line per request.
 *
 *   node bench/build-requests.mjs [--pv pv1]
 *   -> requests/<pv>/requests.jsonl
 *
 * Two kinds of unit, as in the service (bench/match.ts):
 *   project  one per posting and language: demand per area, role
 *   profile  one per CV and language: depth per area, the fallback the
 *            score uses for an area the posting asks for without own lines
 *   line     one per requirement line, pair and language: is_req, must,
 *            axis, evidence, level, with the pair's CV in the state
 *
 * Line ids are stable: <pair>-<lang>-l<NN> in the order splitRequirements()
 * returns them. The German and English line of the same index are the same
 * line only when both splits agree in length. They do not always: the
 * service's splitter drops one-word lines, so "ISTQB-Zertifizierung" vanishes
 * in German while "ISTQB certification" stays (found 2026-10-01). The bench
 * keeps what the service does and reports the mismatch; the de/en comparison
 * aligns lines through the reference judgment, not through the index.
 */
import fs from 'node:fs';
import path from 'node:path';
import YAML from 'yaml';
import { cvEntries, splitRequirements, requirementRequest, projectRequest, profileRequest } from './match.ts';

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const PV = arg('pv', 'pv1');
const ROOT = path.resolve(import.meta.dirname, '..');
const CASES = path.join(ROOT, 'cases', PV);
const OUT = path.join(ROOT, 'requests', PV);
const LANGS = ['en', 'de'];

const pairs = YAML.parse(fs.readFileSync(path.join(CASES, 'pairs.yaml'), 'utf8'));
const list = Array.isArray(pairs) ? pairs : pairs.pairs;
const posting = (id, lang) => fs.readFileSync(path.join(CASES, 'postings', `${id}.${lang}.md`), 'utf8');
const cv = (id, lang) => JSON.parse(fs.readFileSync(path.join(CASES, 'cvs', `${id}.${lang}.json`), 'utf8'));

const rows = [];
const seenPostings = new Set();
const seenCvs = new Set();
for (const p of list) {
  const split = Object.fromEntries(LANGS.map(l => [l, splitRequirements(posting(p.posting, l))]));
  if (split.en.length !== split.de.length) {
    console.warn(`warning ${p.id}: posting ${p.posting} splits into ${split.en.length} en and ${split.de.length} de lines`);
  }
  for (const lang of LANGS) {
    if (!seenPostings.has(`${p.posting}:${lang}`)) {
      seenPostings.add(`${p.posting}:${lang}`);
      rows.push({ pv: PV, unit: 'project', pair: null, posting: p.posting, cv: null, lang, line: `${p.posting}-project`, text: null, request: projectRequest(posting(p.posting, lang)) });
    }
    const entries = cvEntries(cv(p.cv, lang));
    if (!seenCvs.has(`${p.cv}:${lang}`)) {
      seenCvs.add(`${p.cv}:${lang}`);
      rows.push({ pv: PV, unit: 'profile', pair: null, posting: null, cv: p.cv, lang, line: `${p.cv}-profile`, text: null, request: profileRequest(entries) });
    }
    split[lang].forEach((text, i) => {
      const line = `${p.id}-${lang}-l${String(i + 1).padStart(2, '0')}`;
      rows.push({ pv: PV, unit: 'line', pair: p.id, posting: p.posting, cv: p.cv, lang, line, text, request: requirementRequest(text, entries) });
    });
  }
}

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'requests.jsonl'), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
const lines = rows.filter(r => r.unit === 'line');
console.log(`${PV}: ${list.length} pairs, ${lines.length / LANGS.length} lines per language, ${rows.length} requests -> requests/${PV}/requests.jsonl`);
