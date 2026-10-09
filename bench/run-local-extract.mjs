#!/usr/bin/env node
/**
 * Local extraction arm (family ex1): asks an open-weight model through Ollama
 * to list the MASSIVE slots of an utterance as JSON.
 *
 *   node bench/run-local-extract.mjs --model hf.co/EldanRing/Winnow-12B:Q8_0 --arm winnow-12b
 *        [--template gemma4|chatml|gemma] [--host http://localhost:11434] [--pv ex1]
 *        [--lang de] [--limit N] [--tag smoke] [--run 1] [--host-label local-gpu]
 *   -> results/<pv>/<arm>/<date>-run<k>[-<lang>][-<tag>].jsonl, one row per utterance
 *
 * How the call is made:
 *   - generation mode: raw prompt in the model's chat template, thinking off
 *     as in run-local.mjs (empty thought channel for gemma4, empty think
 *     block for chatml), temperature 0, Ollama `format: "json"` so the output
 *     is constrained to JSON
 *   - the prompt lists every slot type by its label text from
 *     cases/<pv>/slot-labels.json (MASSIVE name with underscores replaced by
 *     spaces, "place_name" -> "place name"; the same texts GLiNER gets),
 *     nothing hand-written per type, and asks for a JSON object
 *     {"slots": [{"type", "text"}]} with the text copied verbatim from the
 *     utterance. A list wrapped in an object, because Ollama's JSON mode
 *     expects an object at the top level
 *   - parsing is lenient: a bare list, any list under any key, items with
 *     other key spellings (slot, value) are all read. Output that is not
 *     JSON, or items without type and text, give an empty prediction and an
 *     `error` field
 *   - the answered label is mapped back to the MASSIVE type before it is
 *     stored (`type`; the answered text is kept as `label`). An answer that
 *     is already a MASSIVE name is kept as that type. Types outside the list
 *     are kept as answered and count as wrong
 *   - `latency_ms` is the whole HTTP call (network included when Ollama runs
 *     on another machine), `compute_ms` is Ollama's own total_duration,
 *     `arm_digest` the Ollama digest of the model
 */
import fs from 'node:fs';
import path from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const MODEL = arg('model');
if (!MODEL) { console.error('--model is required'); process.exit(1); }
const TEMPLATE = arg('template', 'gemma4');
const HOST = arg('host', process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/$/, '');
const ARM = arg('arm', MODEL.replace(/[:/]/g, '-'));
// What the `host` field of every row says; default: the Ollama address without scheme.
const HOST_LABEL = arg('host-label', HOST.replace(/^https?:\/\//, ''));
const PV = arg('pv', 'ex1');
const LANG = arg('lang', '');
const LIMIT = Number(arg('limit', '0'));
const TAG = arg('tag', '');
const RUN = Number(arg('run', '1'));
const NUM_PREDICT = Number(arg('num-predict', '512'));
const ROOT = path.resolve(import.meta.dirname, '..');
const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

const LABELS = JSON.parse(fs.readFileSync(path.join(ROOT, 'cases', PV, 'slot-labels.json'), 'utf8')).labels;  // MASSIVE type -> label text
const TYPE_OF = new Map(Object.entries(LABELS).map(([t, l]) => [l.toLowerCase(), t]));
const SYSTEM = 'You extract slots from a short text. Answer with JSON only.';
const user = utt => `Slot types: ${Object.values(LABELS).join(', ')}\n\nText: ${utt}\n\nList every slot in the text as JSON: {"slots": [{"type": "<one of the slot types>", "text": "<the words, copied verbatim from the text>"}]}. Use only the slot types listed. If the text has no slot, answer {"slots": []}.`;

function prompt(utt) {
  const u = user(utt);
  if (TEMPLATE === 'gemma4') return `<|turn>system\n${SYSTEM}<turn|>\n<|turn>user\n${u}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`;
  if (TEMPLATE === 'gemma') return `<start_of_turn>user\n${SYSTEM}\n\n${u}<end_of_turn>\n<start_of_turn>model\n`;
  return `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n${u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

/** Label text answered by the model -> MASSIVE slot type; a MASSIVE name or an unknown text stays as it is. */
export const typeOfLabel = label => (Object.hasOwn(LABELS, label) ? label : TYPE_OF.get(label.toLowerCase()) ?? label);

/** Reads the slots out of the model output. Returns {slots, error}. */
export function parseSlotsOutput(text) {
  let j;
  try { j = JSON.parse(text); } catch {
    const m = String(text).match(/[[{][\s\S]*[\]}]/);
    try { j = m ? JSON.parse(m[0]) : undefined; } catch { /* fall through */ }
    if (j === undefined) return { slots: [], error: 'not json' };
  }
  let list = Array.isArray(j) ? j : null;
  if (!list && j && typeof j === 'object') list = Array.isArray(j.slots) ? j.slots : Object.values(j).find(Array.isArray) || null;
  if (!list) return { slots: [], error: 'no list in json' };
  const slots = []; let bad = 0;
  for (const it of list) {
    const type = it?.type ?? it?.slot ?? it?.slot_type ?? it?.label;
    const t = it?.text ?? it?.value ?? it?.span;
    if (typeof type === 'string' && typeof t === 'string' && t.trim()) slots.push({ type: typeOfLabel(type.trim()), label: type.trim(), text: t.trim() });
    else bad += 1;
  }
  return { slots, error: bad ? `${bad} items without type and text` : undefined };
}

async function generate(p) {
  const t0 = performance.now();
  const res = await fetch(`${HOST}/api/generate`, {
    method: 'POST',
    body: JSON.stringify({ model: MODEL, raw: true, stream: false, prompt: p, format: 'json', keep_alive: '30m', options: { temperature: 0, num_predict: NUM_PREDICT, num_ctx: 4096 } }),
  });
  if (!res.ok) return { error: `http ${res.status} ${(await res.text()).slice(0, 200)}`, ms: Math.round(performance.now() - t0) };
  return { ...(await res.json()), ms: Math.round(performance.now() - t0) };
}

if (import.meta.main ?? process.argv[1] === import.meta.filename) {
  let cases = [];
  for (const lang of LANG ? [LANG] : ['de', 'en']) cases.push(...readJsonl(path.join(ROOT, 'cases', PV, `sample-${lang}.jsonl`)).map(r => ({ ...r, lang })));
  if (LIMIT) cases = (LANG ? [LANG] : ['de', 'en']).flatMap(l => cases.filter(c => c.lang === l).slice(0, LIMIT));

  const OLLAMA_VERSION = await fetch(`${HOST}/api/version`).then(r => r.json()).then(j => j.version).catch(() => null);
  const MODEL_DIGEST = await fetch(`${HOST}/api/tags`).then(r => r.json()).then(j => j.models.find(m => m.name === MODEL || m.model === MODEL)?.digest ?? null).catch(() => null);
  const date = new Date().toISOString().slice(0, 10);
  const dir = path.join(ROOT, 'results', PV, ARM);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${date}-run${RUN}${LANG ? `-${LANG}` : ''}${TAG ? `-${TAG}` : ''}.jsonl`);
  await generate(prompt('wake me up at five am'));  // warm-up: model load and JSON grammar, not recorded
  const out = fs.createWriteStream(file);
  let n = 0; let errors = 0; const t0 = Date.now();
  for (const c of cases) {
    const d = await generate(prompt(c.utt));
    const parsed = d.error ? { slots: [], error: d.error } : parseSlotsOutput(d.response);
    if (parsed.error) errors += 1;
    n += 1;
    out.write(JSON.stringify({
      pv: PV, unit: 'utt', pair: null, posting: null, cv: null, line: `massive-${c.id}-${c.lang}`, question: 'slots', lang: c.lang,
      arm: ARM, arm_version: MODEL, arm_digest: MODEL_DIGEST, runtime: OLLAMA_VERSION ? `ollama ${OLLAMA_VERSION}` : null, host: HOST_LABEL, run: RUN,
      latency_ms: d.ms ?? null, compute_ms: d.total_duration != null ? Math.round(d.total_duration / 1e6) : null, tokens_in: d.prompt_eval_count ?? null, tokens_out: d.eval_count ?? null, cost_usd: 0, ts: new Date().toISOString(),
      slots: parsed.slots, raw: d.response ?? null, error: parsed.error,
    }) + '\n');
    if (n % 50 === 0) process.stderr.write(`${n}/${cases.length} utterances, ${Math.round((Date.now() - t0) / 1000)} s\n`);
  }
  await new Promise(r => out.end(r));
  console.log(`${ARM}: ${n} utterances, ${errors} with a parse or call error, ${Math.round((Date.now() - t0) / 1000)} s -> ${path.relative(ROOT, file)}`);
}
