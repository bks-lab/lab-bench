#!/usr/bin/env node
/**
 * Raw one-token decision prompts for the C8 `decide` workload (plan/c8.md),
 * exactly as bench/run-local.mjs builds them for route1 (system line of the
 * route family, state as named sections, lettered options, prefix ''),
 * rendered with bench/lib/templates.mjs for each template a C8 arm uses.
 *
 *   node bench/c8/decide-prompts.mjs < lines.json > prompts.json
 *
 * stdin: JSON list of route1 line ids; stdout: JSON list of
 * {line, question, options, prompts: {template: raw prompt}}.
 *
 * `harmony` (gpt-oss) is not one of the bench's templates: gpt-oss has no
 * switch to turn reasoning off, so for this throughput workload the assistant
 * turn is opened directly in the final channel. Its decisions are timed, not
 * scored.
 */
import fs from 'node:fs';
import path from 'node:path';
import { rawPrompt } from '../lib/templates.mjs';

const ROOT = path.resolve(import.meta.dirname, '..', '..');
const SYSTEM = 'You classify a short text. Answer each multiple-choice question with the letter of exactly one option.';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const TEMPLATES = ['chatml', 'gemma4', 'mistral', 'granite'];

// The next three functions are bench/run-local.mjs's stateText, lettered and prompt.
function stateText(state) {
  const parts = [];
  for (const [k, v] of Object.entries(state)) {
    if (v && typeof v === 'object') parts.push(`\`${k}\`:\n${Object.entries(v).map(([id, t]) => `${k}.${id}: ${t}`).join('\n')}`);
    else parts.push(`\`${k}\`:\n${v}`);
  }
  return parts.join('\n\n');
}
function lettered(q) {
  if (q.type === 'noul') return [{ key: 'yes', label: 'Yes' }, { key: 'no', label: 'No' }];
  if (q.type === 'score') return q.criteria.map((c, i) => ({ key: String(i), label: c }));
  return Object.entries(q.criteria).map(([k, v]) => ({ key: k, label: v }));
}
function userText(state, q, opts) {
  return `${stateText(state)}\n\nQuestion: ${q.instructions}\n${opts.map((o, i) => `${LETTERS[i]}) ${o.label}`).join('\n')}`;
}
const harmony = (system, user) => `<|start|>system<|message|>${system}<|end|><|start|>user<|message|>${user}<|end|><|start|>assistant<|channel|>final<|message|>`;

const want = JSON.parse(fs.readFileSync(0, 'utf8'));
const byLine = new Map(fs.readFileSync(path.join(ROOT, 'requests', 'route1', 'requests.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l)).map(r => [r.line, r]));
const out = want.map(line => {
  const r = byLine.get(line);
  const q = r.request.questions.scenario;
  const opts = lettered(q);
  const user = userText(r.request.state, q, opts);
  const prompts = Object.fromEntries(TEMPLATES.map(t => [t, rawPrompt(t, SYSTEM, user, '')]));
  prompts.harmony = harmony(SYSTEM, user);
  return { line, question: 'scenario', options: opts.map(o => o.key), prompts };
});
process.stdout.write(JSON.stringify(out));
