#!/usr/bin/env node
/**
 * Jev arm: sends every request of a bench version to the TypeSafe API and
 * writes one normalized row per question.
 *
 *   node bench/run-jev.mjs [--pv pv1] [--runs 5] [--only p01,p02] [--unit line|project|profile] [--run-ids 1,5] [--limit N]
 *        [--lang de] [--per-group N] [--tag smoke]
 *   -> results/<pv>/jev/<date>-run<k>[-<unit>][-<tag>].jsonl (a unit run never overwrites the full run)
 *
 * --only: in pv1 a list of pair ids. In the other families (route1) a list of
 * line prefixes, e.g. massive or fastdec-news_topic. --per-group N keeps the
 * first N requests of every (source, language) group, for smoke runs.
 *
 * Run k = runs is the shuffled run: the options of every choice question go
 * out in a different (seeded) order, to see whether the order moves answers.
 *
 * pv1 sends a whole request (all questions of a line) in one call. The other
 * families (route1) send one call per question, like the local and GLiNER
 * arms: a joint call would let one question see the options of another (the
 * MASSIVE intent options name the gold scenario). Their rows carry
 * `call_scope: "question"`, latency and tokens per call, and `attempts`, the
 * HTTP attempts of the call (more than 1 after a 429 or 5xx; `latency_ms` is
 * the last attempt only). pv1 rows keep their format.
 *
 * The key never sits in this repository. It comes from TYPESAFE_API_KEY. On
 * macOS it can come from a login keychain entry instead: set
 * TYPESAFE_KEYCHAIN_SERVICE and TYPESAFE_KEYCHAIN_ACCOUNT to its service and
 * account names. Some networks get a 403 before any key check. JEV_URL can
 * point to a proxy in that case.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { rowsFor, withShuffledOptions } from './normalize.mjs';
import { perGroup } from './lib/route-ex.mjs';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'pv1');
const RUNS = Number(arg('runs', '5'));
const ONLY = arg('only', '')?.split(',').filter(Boolean);
const LIMIT = Number(arg('limit', '0'));
const UNIT = arg('unit', '');
const LANG = arg('lang', '');
const PER_GROUP = Number(arg('per-group', '0'));
const TAG = arg('tag', '');
// --run-ids 1,5: only these runs (the shuffled one stays the last of --runs)
const RUN_IDS = (arg('run-ids', '') || '').split(',').filter(Boolean).map(Number);
const PARALLEL = 4;
const URL_API = process.env.JEV_URL || 'https://api.typesafe.ai/v1/systemone';
const ROOT = path.resolve(import.meta.dirname, '..');

function apiKey() {
  const env = process.env.TYPESAFE_API_KEY?.trim();
  if (env) return env;
  const service = process.env.TYPESAFE_KEYCHAIN_SERVICE;
  const account = process.env.TYPESAFE_KEYCHAIN_ACCOUNT;
  if (service && account && process.platform === 'darwin') {
    try {
      return execFileSync('security', ['find-generic-password', '-s', service, '-a', account, '-w'], { encoding: 'utf8' }).trim();
    } catch {
      console.error(`No keychain entry for service "${service}" and account "${account}".`);
      process.exit(1);
    }
  }
  console.error('No API key: set TYPESAFE_API_KEY (or, on macOS, TYPESAFE_KEYCHAIN_SERVICE and TYPESAFE_KEYCHAIN_ACCOUNT).');
  process.exit(1);
}
const KEY = apiKey();

let reqs = fs.readFileSync(path.join(ROOT, 'requests', PV, 'requests.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
const all = reqs;
if (ONLY.length && !PV.startsWith('pv')) reqs = reqs.filter(r => ONLY.some(o => r.line.startsWith(o)));
else if (ONLY.length) reqs = reqs.filter(r => r.unit === 'project' ? all.some(x => x.posting === r.posting && ONLY.includes(x.pair)) : r.unit === 'profile' ? all.some(x => x.cv === r.cv && ONLY.includes(x.pair)) : ONLY.includes(r.pair));
if (UNIT) reqs = reqs.filter(r => r.unit === UNIT);
if (LANG) reqs = reqs.filter(r => r.lang === LANG);
if (PER_GROUP) reqs = perGroup(reqs, PER_GROUP);
if (LIMIT) reqs = reqs.slice(0, LIMIT);

async function ask(body) {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const t0 = performance.now();
    const res = await fetch(URL_API, { method: 'POST', headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const ms = Math.round(performance.now() - t0);
    if (res.ok) return { ...(await res.json()), _ms: ms, _attempts: attempt + 1 };
    if (res.status === 403) throw new Error('403 from TypeSafe before the key check: this network may be blocked, set JEV_URL to a proxy (see header comment)');
    if ([429, 500, 502, 503, 504].includes(res.status)) { await new Promise(r => setTimeout(r, 1000 * 2 ** attempt)); continue; }
    return { error: `http ${res.status}`, _ms: ms, _attempts: attempt + 1 };
  }
  return { error: 'retries exhausted', _attempts: 5 };
}

// One call per question outside pv1 (see header).
const PER_QUESTION = !PV.startsWith('pv');
const callsOf = req => (PER_QUESTION
  ? Object.entries(req.request.questions).map(([qid, q]) => ({ ...req, request: { ...req.request, questions: { [qid]: q } } }))
  : [req]);

const date = new Date().toISOString().slice(0, 10);
const dir = path.join(ROOT, 'results', PV, 'jev');
fs.mkdirSync(dir, { recursive: true });

for (const run of RUN_IDS.length ? RUN_IDS : Array.from({ length: RUNS }, (_, i) => i + 1)) {
  const shuffle = RUNS > 1 && run === RUNS;
  const file = path.join(dir, `${date}-run${run}${UNIT ? `-${UNIT}` : ''}${TAG ? `-${TAG}` : ''}.jsonl`);
  const out = fs.createWriteStream(file);
  let next = 0; let done = 0; let tokens = 0; let errors = 0;
  const worker = async () => {
    while (next < reqs.length) {
      const req = reqs[next++];
      for (const call of callsOf(req)) {
        const body = shuffle ? withShuffledOptions(call.request, 1000 + run) : call.request;
        const a = await ask(body);
        if (a.error) errors += 1;
        tokens += a.usage?.input_tokens ?? 0;
        const meta = { arm: 'jev', arm_version: a.model ?? null, host: 'api.typesafe.ai', run, shuffled: shuffle, latency_ms: a._ms ?? null, tokens_in: a.usage?.input_tokens ?? null, tokens_out: a.usage?.output_tokens ?? null, cost_usd: null, ts: new Date().toISOString() };
        if (PER_QUESTION) Object.assign(meta, { call_scope: 'question', attempts: a._attempts ?? null });
        for (const row of rowsFor(call, a, meta)) out.write(JSON.stringify(row) + '\n');
      }
      done += 1;
      if (done % 25 === 0) process.stderr.write(`run ${run}: ${done}/${reqs.length}\n`);
    }
  };
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  await new Promise(r => out.end(r));
  console.log(`run ${run}${shuffle ? ' (shuffled options)' : ''}: ${reqs.length} requests, ${errors} errors, ${tokens} input tokens -> ${path.relative(ROOT, file)}`);
}
