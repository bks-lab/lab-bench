#!/usr/bin/env node
/**
 * Builds the requests and the gold reference of the route1 and ex1 families
 * from the frozen cases (bench/freeze-route-ex.mjs).
 *
 *   node bench/build-route-ex.mjs
 *   -> requests/route1/requests.jsonl       one request per utterance or fast-decisions row
 *                                           (every arm, Jev included, asks its questions
 *                                           one call each, see run-jev.mjs)
 *      reference/route1/gold.<lang>.jsonl   gold label per (line, question), pv1 reference row format
 *      reference/ex1/gold.<lang>.jsonl      gold slots per utterance
 *      cases/ex1/slot-labels.json           label text per slot type, the one mapping both
 *                                           ex1 arms read (underscores -> spaces, see slotLabel)
 *
 * Every arm gets the same label names and the same label texts. A label text
 * is the label name made readable ("alarm_set" -> "alarm: set"), nothing
 * hand-written. fast-decisions labels are used as given.
 *
 * Questions per MASSIVE utterance (state `utterance`):
 *   scenario   choice over the 18 scenarios
 *   intent     choice over the intents of the GOLD scenario only (oracle
 *              scenario), 1 to 9 options
 * Questions per fast-decisions row (state `input`): one choice per
 * single-label task, id <domain>:<task>, options = the task's labels.
 */
import fs from 'node:fs';
import path from 'node:path';
import { humanise, slotLabel } from './lib/route-ex.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
const writeJsonl = (f, rows) => { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, rows.map(r => JSON.stringify(r)).join('\n') + '\n'); };
const choice = (instructions, names, text = humanise) => ({ type: 'choice', instructions, criteria: Object.fromEntries(names.map(n => [n, text(n)])) });

const massive = { de: readJsonl(path.join(ROOT, 'cases/route1/massive-de.jsonl')), en: readJsonl(path.join(ROOT, 'cases/route1/massive-en.jsonl')) };
const scenarios = [...new Set(massive.en.map(r => r.scenario))].sort();
const intentsOf = {};
for (const r of massive.en) (intentsOf[r.scenario] ||= new Set()).add(r.intent);
for (const s of scenarios) intentsOf[s] = [...intentsOf[s]].sort();

const requests = [];
const gold = { de: [], en: [] };
const base = (lang, line, text) => ({ pv: 'route1', unit: 'utt', pair: null, posting: null, cv: null, lang, line, text });
const goldRow = (lang, line, question, c) => ({ pv: 'route1', unit: 'utt', pair: null, posting: null, cv: null, line, lang, arm: 'gold', question, choice: c, note: '' });

for (const lang of ['de', 'en']) {
  for (const r of massive[lang]) {
    const line = `massive-${r.id}-${lang}`;
    requests.push({
      ...base(lang, line, r.utt),
      request: {
        model: 'jev-latest',
        state: { utterance: r.utt },
        questions: {
          scenario: choice('Which scenario does `utterance` belong to?', scenarios),
          intent: choice('Which intent does `utterance` express?', intentsOf[r.scenario]),
        },
      },
    });
    gold[lang].push(goldRow(lang, line, 'scenario', r.scenario), goldRow(lang, line, 'intent', r.intent));
  }
}

for (const r of readJsonl(path.join(ROOT, 'cases/route1/fastdec-en.jsonl'))) {
  const line = `fastdec-${r.domain}-${r.row}-en`;
  const questions = {};
  for (const t of r.tasks) {
    questions[`${r.domain}:${t.task}`] = choice(`Which label fits \`input\` for the task ${t.task}?`, t.labels, x => x);
    gold.en.push(goldRow('en', line, `${r.domain}:${t.task}`, t.gold));
  }
  requests.push({ ...base('en', line, r.input), request: { model: 'jev-latest', state: { input: r.input }, questions } });
}

writeJsonl(path.join(ROOT, 'requests/route1/requests.jsonl'), requests);
for (const lang of ['de', 'en']) writeJsonl(path.join(ROOT, `reference/route1/gold.${lang}.jsonl`), gold[lang]);
const maxOpts = Math.max(...requests.flatMap(r => Object.values(r.request.questions).map(q => Object.keys(q.criteria).length)));
console.log(`route1: ${requests.length} requests, ${gold.de.length + gold.en.length} gold rows, at most ${maxOpts} options per question`);

for (const lang of ['de', 'en']) {
  const rows = readJsonl(path.join(ROOT, `cases/ex1/sample-${lang}.jsonl`)).map(r => ({
    pv: 'ex1', unit: 'utt', pair: null, posting: null, cv: null, line: `massive-${r.id}-${lang}`, lang, arm: 'gold', question: 'slots', slots: r.slots, note: '',
  }));
  writeJsonl(path.join(ROOT, `reference/ex1/gold.${lang}.jsonl`), rows);
  console.log(`ex1 ${lang}: ${rows.length} utterances, ${rows.reduce((s, r) => s + r.slots.length, 0)} gold slots`);
}

{
  const types = JSON.parse(fs.readFileSync(path.join(ROOT, 'cases/ex1/slot-types.json'), 'utf8')).types;
  const labels = Object.fromEntries(types.map(t => [t, slotLabel(t)]));
  if (new Set(Object.values(labels)).size !== types.length) throw new Error('ex1: two slot types map to the same label text');
  fs.writeFileSync(path.join(ROOT, 'cases/ex1/slot-labels.json'), JSON.stringify({
    source: 'generated by bench/build-route-ex.mjs from cases/ex1/slot-types.json',
    rule: 'label text = MASSIVE slot type with every underscore replaced by a space; nothing hand-written',
    labels,
  }, null, 2) + '\n');
  console.log(`ex1: ${types.length} slot labels, ${types.filter(t => labels[t] !== t).length} changed by the rule`);
}
