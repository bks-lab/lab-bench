#!/usr/bin/env node
/**
 * Local arm: asks an open-weight model the same questions Jev gets, through
 * Ollama, and reads the probabilities from the logprobs of the answer letter.
 *
 *   node bench/run-local.mjs --model qwen3:32b [--template chatml|gemma4|gemma] [--prefix 'Answer: **']
 *        [--host http://localhost:11434] [--arm qwen3-32b] [--pv pv1]
 *        [--runs 5] [--only p01] [--unit line|project|profile] [--run-ids 1,5] [--limit N]
 *        [--lang de] [--per-group N] [--tag smoke] [--skip-over-26] [--host-label local-gpu]
 *   -> results/<pv>/<arm>/<date>-run<k>[-<unit>][-<tag>].jsonl (a unit run never overwrites the full run)
 *
 * --only: in pv1 a list of pair ids, in the other families (route1) a list of
 * line prefixes. --per-group N keeps the first N requests of every (source,
 * language) group, for smoke runs. The system line depends on the family
 * (SYSTEMS below). A question with more than 26 options cannot be lettered:
 * the run refuses to start unless --skip-over-26 is given, which writes
 * those questions as rows with an error and no answer.
 *
 * How a typed question becomes a raw model call:
 *   - the state goes into the user turn as named sections (`requirement`, and
 *     `cv` with one line per entry as cv.<id>), so the question's backtick
 *     references and the evidence options (`cv.s-x`) mean what they mean to Jev
 *   - the instruction is Jev's wording, unchanged
 *   - options are lettered A, B, C ...: noul is yes/no, score is its levels
 *     0..3, choice is its criteria in definition order
 *   - the assistant turn is pre-filled up to "Answer: **" (unprompted, Qwen writes the letter in bold, so the bare prefix put 28 of 29 first tokens on " **", measured 2026-10-01) (thinking switched off
 *     by an empty think block for chatml), so the next token is the letter
 *   - the probability of an option is the summed probability of its letter
 *     among the top logprobs, renormalised over the options; `mass` records
 *     how much probability the letters had before that, a low mass means the
 *     model wanted to say something else. Outside pv1, a mass of 0 (no letter
 *     in the top 20) is written as an error with no answer
 *   - `latency_ms` is the whole HTTP call as seen by this script (network
 *     included when Ollama runs on another machine), `compute_ms` is Ollama's
 *     own total_duration; `arm_digest` is the Ollama digest of the model
 *
 * One call per question, the question last, so Ollama reuses the cached prompt
 * prefix (state) across the five questions of a line.
 */
import fs from 'node:fs';
import path from 'node:path';
import { optionsOf, withShuffledOptions } from './normalize.mjs';
import { perGroup } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const MODEL = arg('model');
if (!MODEL) { console.error('--model is required'); process.exit(1); }
const TEMPLATE = arg('template', 'chatml');
// What the answer turn starts with; the next token should be the letter.
const PREFIX = arg('prefix', 'Answer: **');
// Default: OLLAMA_HOST if set, else the local Ollama.
const HOST = arg('host', process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/$/, '');
const ARM = arg('arm', MODEL.replace(/[:/]/g, '-'));
// What the `host` field of every row says; default: the Ollama address without scheme.
const HOST_LABEL = arg('host-label', HOST.replace(/^https?:\/\//, ''));
const PV = arg('pv', 'pv1');
const RUNS = Number(arg('runs', '5'));
const ONLY = arg('only', '')?.split(',').filter(Boolean);
const LIMIT = Number(arg('limit', '0'));
const UNIT = arg('unit', '');
const LANG = arg('lang', '');
const PER_GROUP = Number(arg('per-group', '0'));
const TAG = arg('tag', '');
const SKIP_OVER_26 = process.argv.includes('--skip-over-26');
// --run-ids 1,5: only these runs (the shuffled one stays the last of --runs)
const RUN_IDS = (arg('run-ids', '') || '').split(',').filter(Boolean).map(Number);
const NUM_CTX = Number(arg('ctx', '16384'));
const ROOT = path.resolve(import.meta.dirname, '..');
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
// The only domain text in this file: one system line per family.
const SYSTEMS = {
  pv: 'You judge a project posting and a candidate CV. Answer each multiple-choice question with the letter of exactly one option.',
  route: 'You classify a short text. Answer each multiple-choice question with the letter of exactly one option.',
};
const SYSTEM = arg('system', SYSTEMS[Object.keys(SYSTEMS).find(f => PV.startsWith(f))]);
if (!SYSTEM) { console.error(`no system line for ${PV}, pass --system`); process.exit(1); }

let reqs = fs.readFileSync(path.join(ROOT, 'requests', PV, 'requests.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
const all = reqs;
if (ONLY.length && !PV.startsWith('pv')) reqs = reqs.filter(r => ONLY.some(o => r.line.startsWith(o)));
else if (ONLY.length) reqs = reqs.filter(r => r.unit === 'project' ? all.some(x => x.posting === r.posting && ONLY.includes(x.pair)) : r.unit === 'profile' ? all.some(x => x.cv === r.cv && ONLY.includes(x.pair)) : ONLY.includes(r.pair));
if (UNIT) reqs = reqs.filter(r => r.unit === UNIT);
if (LANG) reqs = reqs.filter(r => r.lang === LANG);
if (PER_GROUP) reqs = perGroup(reqs, PER_GROUP);
if (LIMIT) reqs = reqs.slice(0, LIMIT);
{
  const over = reqs.flatMap(r => Object.entries(r.request.questions).filter(([, q]) => optionsOf(q).length > 26).map(([qid, q]) => `${r.line} ${qid} (${optionsOf(q).length})`));
  if (over.length && !SKIP_OVER_26) { console.error(`${over.length} questions have more than 26 options, e.g. ${over.slice(0, 3).join(', ')}. Pass --skip-over-26 to write them as unanswered rows.`); process.exit(1); }
  if (over.length) console.error(`${over.length} questions with more than 26 options are written as unanswered rows`);
}

function stateText(state) {
  const parts = [];
  for (const [k, v] of Object.entries(state)) {
    if (v && typeof v === 'object') parts.push(`\`${k}\`:\n${Object.entries(v).map(([id, t]) => `${k}.${id}: ${t}`).join('\n')}`);
    else parts.push(`\`${k}\`:\n${v}`);
  }
  return parts.join('\n\n');
}

/** [{key, label}] in the order the question is asked, keys as in normalize.optionsOf. */
function lettered(q) {
  if (q.type === 'noul') return [{ key: 'yes', label: 'Yes' }, { key: 'no', label: 'No' }];
  if (q.type === 'score') return q.criteria.map((c, i) => ({ key: String(i), label: c }));
  return Object.entries(q.criteria).map(([k, v]) => ({ key: k, label: v }));
}

function prompt(state, q, opts) {
  const user = `${stateText(state)}\n\nQuestion: ${q.instructions}\n${opts.map((o, i) => `${LETTERS[i]}) ${o.label}`).join('\n')}`;
  // Gemma 4 (2026): <|turn>role ... <turn|>, thinking off by an empty thought channel
  if (TEMPLATE === 'gemma4') return `<|turn>system\n${SYSTEM}<turn|>\n<|turn>user\n${user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>${PREFIX}`;
  if (TEMPLATE === 'gemma') return `<start_of_turn>user\n${SYSTEM}\n\n${user}<end_of_turn>\n<start_of_turn>model\n${PREFIX}`;
  return `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n${PREFIX}`;
}

async function generate(p) {
  const t0 = performance.now();
  const res = await fetch(`${HOST}/api/generate`, {
    method: 'POST',
    body: JSON.stringify({ model: MODEL, raw: true, stream: false, prompt: p, logprobs: true, top_logprobs: 20, keep_alive: '30m', options: { temperature: 0, num_predict: 1, num_ctx: NUM_CTX } }),
  });
  if (!res.ok) return { error: `http ${res.status} ${(await res.text()).slice(0, 200)}`, ms: Math.round(performance.now() - t0) };
  const d = await res.json();
  return { ...d, ms: Math.round(performance.now() - t0) };
}

async function askQuestion(state, q) {
  const opts = lettered(q);
  if (opts.length > LETTERS.length) return { error: `too many options (${opts.length})` };
  const d = await generate(prompt(state, q, opts));
  if (d.error) return d;
  const top = d.logprobs?.[0]?.top_logprobs || [];
  const byLetter = {};
  for (const t of top) {
    const s = t.token.trim();
    if (s.length === 1 && LETTERS.includes(s)) byLetter[s] = (byLetter[s] || 0) + Math.exp(t.logprob);
  }
  const raw = opts.map((_, i) => byLetter[LETTERS[i]] || 0);
  const mass = raw.reduce((a, b) => a + b, 0);
  const pByKey = Object.fromEntries(opts.map((o, i) => [o.key, mass ? raw[i] / mass : 0]));
  return { pByKey, mass, ms: d.ms, compute_ms: d.total_duration != null ? Math.round(d.total_duration / 1e6) : null, tokens_in: d.prompt_eval_count, first: d.response };
}

const date = new Date().toISOString().slice(0, 10);
// Recorded per row: the runtime version can change answers, and the run files
// of 2026-10-01 did not carry it (0.30.6, Qwen3.8 on 0.35.0, from the work log).
const OLLAMA_VERSION = await fetch(`${HOST}/api/version`).then(r => r.json()).then(j => j.version).catch(() => null);
// The weights behind the tag, so a later run can tell whether they changed.
const MODEL_DIGEST = await fetch(`${HOST}/api/tags`).then(r => r.json()).then(j => j.models.find(m => m.name === MODEL || m.model === MODEL)?.digest ?? null).catch(() => null);
const dir = path.join(ROOT, 'results', PV, ARM);
fs.mkdirSync(dir, { recursive: true });
const r4 = x => Math.round(x * 1e4) / 1e4;

for (const run of RUN_IDS.length ? RUN_IDS : Array.from({ length: RUNS }, (_, i) => i + 1)) {
  const shuffle = RUNS > 1 && run === RUNS;
  const file = path.join(dir, `${date}-run${run}${UNIT ? `-${UNIT}` : ''}${TAG ? `-${TAG}` : ''}.jsonl`);
  const out = fs.createWriteStream(file);
  let errors = 0; let lowMass = 0; let n = 0; const t0 = Date.now();
  for (const req of reqs) {
    const request = shuffle ? withShuffledOptions(req.request, 1000 + run) : req.request;
    for (const [qid, q] of Object.entries(request.questions)) {
      const a = await askQuestion(request.state, q);
      const options = optionsOf(req.request.questions[qid]);
      // route1: no option letter among the top logprobs is no answer, not option A
      if (!PV.startsWith('pv') && a.mass === 0) { a.error = 'no option letter in the top logprobs'; delete a.pByKey; }
      const probs = a.pByKey ? options.map(o => r4(a.pByKey[o] ?? 0)) : null;
      const choice = probs ? options[probs.indexOf(Math.max(...probs))] : null;
      if (a.error) errors += 1; else if (a.mass < 0.5) lowMass += 1;
      n += 1;
      out.write(JSON.stringify({
        pv: req.pv, unit: req.unit, pair: req.pair, posting: req.posting, cv: req.cv, line: req.line,
        question: qid, qtype: q.type, lang: req.lang,
        arm: ARM, arm_version: MODEL, arm_digest: MODEL_DIGEST, runtime: OLLAMA_VERSION ? `ollama ${OLLAMA_VERSION}` : null, host: HOST_LABEL, run, shuffled: shuffle,
        latency_ms: a.ms ?? null, compute_ms: a.compute_ms ?? null, tokens_in: a.tokens_in ?? null, cost_usd: 0, ts: new Date().toISOString(),
        options, probs, choice, mass: a.mass != null ? r4(a.mass) : undefined, first_token: a.first, error: a.error,
      }) + '\n');
      if (n % 50 === 0) process.stderr.write(`run ${run}: ${n} questions, ${Math.round((Date.now() - t0) / 1000)} s\n`);
    }
  }
  await new Promise(r => out.end(r));
  console.log(`run ${run}${shuffle ? ' (shuffled options)' : ''}: ${n} questions, ${errors} errors, ${lowMass} with letter mass < 0.5, ${Math.round((Date.now() - t0) / 1000)} s -> ${path.relative(ROOT, file)}`);
}
