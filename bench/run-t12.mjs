#!/usr/bin/env node
/**
 * T12 driver (plan/t12.md): runs the local model series one model after the
 * other on one Ollama host, with the machine measurements of the plan.
 *
 *   OLLAMA_HOST=http://<workstation>:11434 HOST_LABEL=local-rtx4090:11434 \
 *   VRAM_CMD='nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' \
 *   node bench/run-t12.mjs [--only qwen3-4b,qwen3-14b] [--skip-ex1] [--probe-only] [--load-only]
 *
 * Per model:
 *   1. unload every loaded model (keep_alive 0), read idle VRAM (VRAM_CMD,
 *      default: local nvidia-smi; it may be an ssh command)
 *   2. template check: the first probe question once through /api/chat
 *      (Ollama renders its own template) and once as the raw prompt of the
 *      runner, each from a cold load, prompt token counts recorded
 *   3. load time: empty /api/generate after an unload, Ollama load_duration
 *   4. letter-mass probe: run-local.mjs --per-group 4 --tag probe (20
 *      questions) with prefix ''; median mass < 0.5 -> once more with
 *      'Answer: **'; still < 0.5 -> route1 format failure, no full route1 run
 *   5. full route1 run, VRAM sampled every 20 s; /api/ps size and size_vram
 *   6. full ex1 run, VRAM sampled the same way
 *   7. unload
 * Writes results/t12/machine.json (one entry per arm, merged) and logs the
 * runners' output to stdout.
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execSync } from 'node:child_process';
import { rawPrompt } from './lib/templates.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const ROOT = path.resolve(import.meta.dirname, '..');
const HOST = (process.env.OLLAMA_HOST || 'http://localhost:11434').replace(/\/$/, '');
const HOST_LABEL = process.env.HOST_LABEL || 'local';
const VRAM_CMD = process.env.VRAM_CMD || 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits';
const ONLY = (arg('only', '') || '').split(',').filter(Boolean);
const SKIP_EX1 = process.argv.includes('--skip-ex1');
const PROBE_ONLY = process.argv.includes('--probe-only');
// --load-only: per model, three cold loads (unload, then an empty generate), wall time from the Mac
const LOAD_ONLY = process.argv.includes('--load-only');

// plan/t12.md, in the order of the plan
export const SERIES = [
  { arm: 'qwen3-4b', model: 'qwen3:4b-q4_K_M', template: 'chatml', cls: 'small' },
  { arm: 'qwen3-14b', model: 'qwen3:14b', template: 'chatml', cls: 'medium' },
  { arm: 'qwen3-30b-a3b', model: 'qwen3:30b-a3b-q4_K_M', template: 'chatml', cls: 'moe' },
  { arm: 'qwen3-32b', model: 'qwen3:32b', template: 'chatml', cls: 'large' },
  { arm: 'gemma4-e4b', model: 'gemma4:e4b', template: 'gemma4', cls: 'small' },
  { arm: 'granite4-micro', model: 'granite4:micro', template: 'granite', cls: 'small' },
  { arm: 'phi4-14b', model: 'phi4:14b', template: 'phi4', cls: 'medium' },
  { arm: 'gemma4-12b', model: 'gemma4:12b', template: 'gemma4', cls: 'medium' },
  { arm: 'mistral-small3.2-24b', model: 'mistral-small3.2:24b', template: 'mistral', cls: 'large' },
  { arm: 'qwen3.8-27b', model: 'qwen3.8:27b', template: 'chatml', cls: 'large' },
  { arm: 'shisa-de-1', model: 'hf.co/mradermacher/shisa-de-1-GGUF:Q4_K_M', template: 'gemma4', cls: 'german-moe' },
];
// The arm measured before T12 (results/route1/method.md), listed for the series table only.
export const MEASURED = [{ arm: 'winnow-12b', model: 'hf.co/EldanRing/Winnow-12B:Q8_0', template: 'gemma4', cls: 'medium' }];

const post = (p, body) => fetch(`${HOST}${p}`, { method: 'POST', body: JSON.stringify(body) }).then(r => r.json());
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const vram = () => { try { return Math.max(...execSync(VRAM_CMD, { encoding: 'utf8', timeout: 30000 }).trim().split(/\s+/).map(Number).filter(Number.isFinite)); } catch { return null; } };
async function unloadAll() {
  const ps = await fetch(`${HOST}/api/ps`).then(r => r.json());
  for (const m of ps.models || []) await post('/api/generate', { model: m.name, keep_alive: 0 });
  for (let i = 0; i < 30; i++) { const p = await fetch(`${HOST}/api/ps`).then(r => r.json()); if (!(p.models || []).length) return; await new Promise(r => setTimeout(r, 1000)); }
}
const median = xs => { const a = xs.filter(x => x != null).sort((p, q) => p - q); const m = Math.floor(a.length / 2); return a.length ? (a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2) : null; };

function run(script, args) {
  return new Promise((resolve, reject) => {
    const c = spawn('node', [path.join(ROOT, 'bench', script), ...args], { cwd: ROOT, stdio: ['ignore', 'inherit', 'inherit'], env: { ...process.env, OLLAMA_HOST: HOST } });
    c.on('exit', code => (code === 0 ? resolve() : reject(new Error(`${script} exited ${code}`))));
  });
}
async function sampled(fn) {
  const samples = [];
  const t = setInterval(() => { const v = vram(); if (v != null) samples.push(v); }, 20000);
  const t0 = Date.now();
  try { await fn(); } finally { clearInterval(t); }
  return { wall_s: Math.round((Date.now() - t0) / 1000), vram_max_mib: samples.length ? Math.max(...samples) : null, vram_samples: samples.length };
}
const today = () => new Date().toISOString().slice(0, 10);
const probeFile = (arm, tag) => path.join(ROOT, 'results', 'route1', arm, `${today()}-run1-${tag}.jsonl`);
const massOf = f => median(fs.readFileSync(f, 'utf8').trim().split('\n').map(l => JSON.parse(l)).filter(r => !String(r.error || '').startsWith('too many')).map(r => r.mass ?? 0));

// The first probe question as the runner would send it, for the template check.
async function templateCheck(m) {
  const req = JSON.parse(fs.readFileSync(path.join(ROOT, 'requests', 'route1', 'requests.jsonl'), 'utf8').split('\n')[0]);
  const [qid, q] = Object.entries(req.request.questions)[0];
  const system = 'You classify a short text. Answer each multiple-choice question with the letter of exactly one option.';
  const opts = Object.values(q.criteria);
  const L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const state = Object.entries(req.request.state).map(([k, v]) => `\`${k}\`:\n${v}`).join('\n\n');
  const user = `${state}\n\nQuestion: ${q.instructions}\n${opts.map((o, i) => `${L[i]}) ${o}`).join('\n')}`;
  await unloadAll();
  const chat = await post('/api/chat', { model: m.model, stream: false, messages: [{ role: 'system', content: system }, { role: 'user', content: user }], options: { temperature: 0, num_predict: 1, num_ctx: 16384 } });
  await unloadAll();
  const raw = await post('/api/generate', { model: m.model, raw: true, stream: false, prompt: rawPrompt(m.template, system, user), options: { temperature: 0, num_predict: 1, num_ctx: 16384 } });
  await unloadAll();
  return { question: `${req.line} ${qid}`, chat_prompt_tokens: chat.prompt_eval_count ?? null, raw_prompt_tokens: raw.prompt_eval_count ?? null, diff: (raw.prompt_eval_count ?? 0) - (chat.prompt_eval_count ?? 0), chat_first: chat.message?.content ?? null, raw_first: raw.response ?? null };
}

const outFile = path.join(ROOT, 'results', 't12', 'machine.json');
fs.mkdirSync(path.dirname(outFile), { recursive: true });
const machine = fs.existsSync(outFile) ? JSON.parse(fs.readFileSync(outFile, 'utf8')) : { note: 'plan/t12.md, written by bench/run-t12.mjs', arms: {} };
const save = () => fs.writeFileSync(outFile, JSON.stringify(machine, null, 1) + '\n');

if (import.meta.main ?? process.argv[1] === import.meta.filename) {
  const version = await fetch(`${HOST}/api/version`).then(r => r.json()).then(j => j.version);
  const tags = await fetch(`${HOST}/api/tags`).then(r => r.json());
  if (LOAD_ONLY) {
    for (const m of [...SERIES, ...MEASURED].filter(x => !ONLY.length || ONLY.includes(x.arm))) {
      const walls = [];
      for (let i = 0; i < 3; i++) {
        await unloadAll();
        const t0 = Date.now();
        const r = await post('/api/generate', { model: m.model, prompt: '', keep_alive: '5m', options: { num_ctx: 16384 } });
        if (r.error) { log(m.arm, r.error); break; }
        walls.push(Date.now() - t0);
      }
      const loadedVram = vram();
      await unloadAll();
      const idleVram = vram();
      const e = machine.arms[m.arm] ||= { arm: m.arm, model: m.model, class: m.cls, template: m.template };
      e.load_wall_ms = walls; e.load_wall_ms_median = walls.length ? [...walls].sort((a, b) => a - b)[Math.floor(walls.length / 2)] : null;
      if (!e.route1) { e.loaded_vram_mib = loadedVram; e.idle_vram_mib = idleVram; e.vram_note = 'not run by the T12 driver: VRAM after loading with num_ctx 16384, no run peak'; }
      e.load_note = 'three loads after an unload, empty prompt, num_ctx 16384, wall time of the HTTP call from the Mac; the file is in the OS cache after the run, so this is a warm-disk load';
      log(m.arm, 'load ms', walls.join(', '));
      save();
    }
    process.exit(0);
  }
  for (const m of SERIES.filter(x => !ONLY.length || ONLY.includes(x.arm))) {
    const info = tags.models.find(t => t.name === m.model || t.model === m.model);
    if (!info) { log(`${m.arm}: ${m.model} not installed, skipped`); continue; }
    const e = machine.arms[m.arm] = { ...(machine.arms[m.arm] || {}), arm: m.arm, model: m.model, class: m.cls, template: m.template, digest: info.digest, quantization: info.details?.quantization_level, parameter_size: info.details?.parameter_size, file_bytes: info.size, runtime: `ollama ${version}`, host: HOST_LABEL, started: new Date().toISOString() };
    log(`== ${m.arm} (${m.model})`);
    const show = await post('/api/show', { model: m.model });
    fs.mkdirSync(path.join(ROOT, 'results', 't12', 'templates'), { recursive: true });
    fs.writeFileSync(path.join(ROOT, 'results', 't12', 'templates', `${m.arm}.txt`), `# Ollama template of ${m.model} (digest ${info.digest}), /api/show, read ${new Date().toISOString()}\n# runner template: ${m.template}\n${show.template}\n`);
    await unloadAll();
    e.idle_vram_mib = vram();
    e.template_check = await templateCheck(m);
    log('template check', JSON.stringify(e.template_check));
    if (Math.abs(e.template_check.diff) > 6) { e.stopped = 'template check: raw and chat prompt token counts differ by more than 6 (plan/t12.md)'; save(); log(e.stopped); continue; }
    const ld = await post('/api/generate', { model: m.model, prompt: '', keep_alive: '30m', options: { num_ctx: 16384 } });
    e.load_ms = ld.load_duration != null ? Math.round(ld.load_duration / 1e6) : null;
    const ps = (await fetch(`${HOST}/api/ps`).then(r => r.json())).models?.find(x => x.name === m.model || x.model === m.model);
    e.ps_size_bytes = ps?.size ?? null; e.ps_size_vram_bytes = ps?.size_vram ?? null; e.ps_context_length = ps?.context_length ?? null;
    e.loaded_vram_mib = vram();
    save();
    const common = ['--pv', 'route1', '--model', m.model, '--arm', m.arm, '--template', m.template, '--host', HOST, '--host-label', HOST_LABEL, '--runs', '1', '--skip-over-26'];
    e.probe = [];
    let prefix = null;
    for (const [pre, tag] of [['', 'probe'], ['Answer: **', 'probe-answer']]) {
      await run('run-local.mjs', [...common, '--prefix', pre, '--per-group', '4', '--tag', tag]);
      const f = probeFile(m.arm, tag);
      const mm = massOf(f);
      e.probe.push({ prefix: pre, median_mass: mm, file: path.relative(ROOT, f) });
      log(`probe prefix ${JSON.stringify(pre)}: median letter mass ${mm}`);
      if (mm >= 0.5) { prefix = pre; break; }
    }
    e.prefix = prefix; e.route1_format_failure = prefix == null;
    save();
    if (PROBE_ONLY) { await unloadAll(); continue; }
    if (prefix != null) {
      e.route1 = await sampled(() => run('run-local.mjs', [...common, '--prefix', prefix]));
      const ps2 = (await fetch(`${HOST}/api/ps`).then(r => r.json())).models?.find(x => x.name === m.model || x.model === m.model);
      e.route1.ps_size_bytes = ps2?.size ?? null; e.route1.ps_size_vram_bytes = ps2?.size_vram ?? null;
      log('route1', JSON.stringify(e.route1)); save();
    }
    if (!SKIP_EX1) {
      e.ex1 = await sampled(() => run('run-local-extract.mjs', ['--model', m.model, '--arm', m.arm, '--template', m.template, '--host', HOST, '--host-label', HOST_LABEL]));
      const ps3 = (await fetch(`${HOST}/api/ps`).then(r => r.json())).models?.find(x => x.name === m.model || x.model === m.model);
      e.ex1.ps_size_bytes = ps3?.size ?? null; e.ex1.ps_size_vram_bytes = ps3?.size_vram ?? null;
      log('ex1', JSON.stringify(e.ex1));
    }
    e.finished = new Date().toISOString();
    save();
    await unloadAll();
  }
  log('series done');
}
