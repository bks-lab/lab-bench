#!/usr/bin/env node
/**
 * T12 series report (plan/t12.md): one row per local model with the route1
 * and ex1 numbers of the score reports, the letter mass, latency, throughput
 * and the machine measurements of bench/run-t12.mjs.
 *
 *   node bench/score-t12.mjs        (after score-route.mjs and score-extract.mjs)
 *   -> results/t12/series.md, results/t12/probes.md
 *
 * Accuracies and F1 are read from results/route1/score.md and
 * results/ex1/score.md, so they are the reports' numbers. Throughput is rows
 * per wall time of the run (first to last row timestamp), one stream.
 */
import fs from 'node:fs';
import path from 'node:path';
import { runRows, readJsonl } from './lib/results.mjs';
import { median } from './lib/route-ex.mjs';
import { SERIES, MEASURED } from './run-t12.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const rd = f => fs.readFileSync(path.join(ROOT, f), 'utf8');
function sections(md) {
  const out = {}; let cur = null;
  for (const line of md.split('\n')) {
    if (line.startsWith('## ')) { cur = line.slice(3).trim(); out[cur] = []; continue; }
    if (cur && line.startsWith('|') && !line.startsWith('|---')) out[cur].push(line.split('|').slice(1, -1).map(c => c.trim()));
  }
  for (const k of Object.keys(out)) { const [h, ...rows] = out[k]; out[k] = (rows || []).map(r => Object.fromEntries(h.map((c, i) => [c, r[i]]))); }
  return out;
}
const route = sections(rd('results/route1/score.md'));
const ex = sections(rd('results/ex1/score.md'));
const machine = fs.existsSync(path.join(ROOT, 'results/t12/machine.json')) ? JSON.parse(rd('results/t12/machine.json')).arms : {};
const pick = (sec, arm, col) => route[sec]?.find(r => r.arm === arm)?.[col] ?? '';
const pickEx = (lang, arm, col) => ex[lang]?.find(r => r.arm === arm)?.[col] ?? '';
const perHour = rows => { const ts = rows.map(r => Date.parse(r.ts)).filter(Number.isFinite).sort((a, b) => a - b); return ts.length > 1 ? Math.round(rows.length / ((ts.at(-1) - ts[0]) / 3600000)) : null; };
const gib = b => (b == null ? '' : (b / 2 ** 30).toFixed(1));
const fmt = x => (x == null || x === '' ? '' : String(x));
const ROUTE_SECS = [['massive scenario, de', 'scen de'], ['massive scenario, en', 'scen en'], ['massive intent, 2+ options, de', 'int2 de'], ['massive intent, 2+ options, en', 'int2 en'], ['fastdec, tasks with 26 options or fewer, en', 'fd26 en']];

const all = [...SERIES, ...MEASURED];
const summary = [];
const num = x => { const m = String(x ?? '').match(/-?[\d.]+/); return m ? Number(m[0]) : null; };
const wallS = rows => { const ts = rows.map(r => Date.parse(r.ts)).filter(Number.isFinite).sort((a, b) => a - b); return ts.length > 1 ? Math.round((ts.at(-1) - ts[0]) / 1000) : null; };
const goldRoute = new Map();
for (const lang of ['de', 'en']) for (const g of readJsonl(path.join(ROOT, 'reference', 'route1', `gold.${lang}.jsonl`))) goldRoute.set(`${g.line}|${g.question}`, g.choice);
const FAMILY = { 'qwen3-4b': 'Qwen3', 'qwen3-14b': 'Qwen3', 'qwen3-32b': 'Qwen3', 'qwen3-30b-a3b': 'Qwen3 (MoE)', 'qwen3.8-27b': 'Qwen3.8', 'gemma4-e4b': 'Gemma 4', 'gemma4-12b': 'Gemma 4', 'winnow-12b': 'Gemma 4 (LoRA)', 'shisa-de-1': 'Gemma 4 (MoE, German-tuned)', 'granite4-micro': 'Granite 4', 'phi4-14b': 'Phi-4', 'mistral-small3.2-24b': 'Mistral Small 3.2' };
const PARAMS = { 'winnow-12b': 11.9 };
const QUANT = { 'winnow-12b': 'Q8_0' };
const rowsOut = []; const probeOut = [];
for (const m of all) {
  const has = pv => fs.existsSync(path.join(ROOT, 'results', pv, m.arm));
  const r1 = has('route1') ? runRows(ROOT, 'route1', m.arm, { run: 1 }) : [];
  const e1 = has('ex1') ? runRows(ROOT, 'ex1', m.arm, { run: 1 }) : [];
  const mc = machine[m.arm] || {};
  const mass = median(r1.filter(r => r.mass != null).map(r => r.mass));
  const fail = mc.route1_format_failure === true || (mass != null && mass < 0.5);
  const acc = s => (r1.length ? (fail ? `(${pick(s, m.arm, 'acc')})` : pick(s, m.arm, 'acc')) : 'not run');
  rowsOut.push({
    cls: m.cls, arm: m.arm, model: m.model,
    quant: mc.quantization || (r1[0] ? '' : ''), digest: (mc.digest || r1[0]?.arm_digest || '').slice(0, 12),
    route: ROUTE_SECS.map(([s]) => acc(s)),
    f1de: pickEx('de', m.arm, 'F1'), f1en: pickEx('en', m.arm, 'F1'), frde: pickEx('de', m.arm, 'exact frame'), fren: pickEx('en', m.arm, 'exact frame'),
    exErr: [pickEx('de', m.arm, 'errors'), pickEx('en', m.arm, 'errors')].filter(Boolean).join('/'),
    mass: mass == null ? '' : mass.toFixed(3), prefix: mc.prefix ?? (m.arm === 'winnow-12b' ? '' : null),
    p50r: median(r1.map(r => r.latency_ms)), p50e: median(e1.map(r => r.latency_ms)),
    dph: perHour(r1), uph: perHour(e1),
    vram: mc.route1?.vram_max_mib != null && mc.idle_vram_mib != null ? ((Math.max(mc.route1.vram_max_mib, mc.ex1?.vram_max_mib ?? 0) - mc.idle_vram_mib) / 1024).toFixed(1) : '',
    // /api/ps under Ollama 0.35.0 reports a few hundred MB to 1.3 GB for the Gemma 4 builds, far below their VRAM use: not usable for them
    ps: m.template === 'gemma4' && mc.route1 ? 'n/a' : mc.route1?.ps_size_bytes ? `${gib(mc.route1.ps_size_vram_bytes)}/${gib(mc.route1.ps_size_bytes)}` : '',
    load: mc.load_wall_ms_median != null ? (mc.load_wall_ms_median / 1000).toFixed(1) : '',
    fail,
  });
  if (r1.length || e1.length) {
    // route1 pooled over the questions that were asked: the 100 support_intent:intent questions (28 options) are unanswered by design and left out
    const asked = r1.filter(r => !String(r.error || '').startsWith('too many'));
    const right = asked.filter(r => r.choice != null && r.choice === goldRoute.get(`${r.line}|${r.question}`)).length;
        summary.push({
      arm: m.arm, model: m.model, family: FAMILY[m.arm] || null, size_class: m.cls,
      params_b: mc.parameter_size ? Number(String(mc.parameter_size).replace(/B$/, '')) : PARAMS[m.arm] ?? null,
      quant: mc.quantization || QUANT[m.arm] || null, digest: mc.digest || r1[0]?.arm_digest || null,
      route1: r1.length ? {
        accuracy: +(100 * right / asked.length).toFixed(1), n: asked.length, excluded_unanswered_by_design: r1.length - asked.length,
        accuracy_groups: Object.fromEntries(ROUTE_SECS.map(([sec]) => [sec, fail ? null : num(pick(sec, m.arm, 'acc'))])),
        letter_mass_below_05_share: +(asked.filter(r => (r.mass ?? 0) < 0.5).length / asked.length).toFixed(4),
        letter_mass_median: mass, format_failure: fail,
        wall_s: mc.route1?.wall_s ?? wallS(r1), items_per_s: +(asked.length / (mc.route1?.wall_s ?? wallS(r1))).toFixed(2), p50_ms: median(r1.map(r => r.latency_ms)),
      } : null,
      ex1: e1.length ? {
        slot_f1_de: num(pickEx('de', m.arm, 'F1')), slot_f1_en: num(pickEx('en', m.arm, 'F1')), exact_frame_de: num(pickEx('de', m.arm, 'exact frame')), exact_frame_en: num(pickEx('en', m.arm, 'exact frame')),
        n: e1.length, wall_s: mc.ex1?.wall_s ?? wallS(e1), items_per_s: +(e1.length / (mc.ex1?.wall_s ?? wallS(e1))).toFixed(2), p50_ms: median(e1.map(r => r.latency_ms)),
      } : null,
      vram_max_mib: mc.route1 ? Math.max(mc.route1.vram_max_mib ?? 0, mc.ex1?.vram_max_mib ?? 0) : mc.loaded_vram_mib ?? null,
      ...(mc.vram_note ? { vram_note: mc.vram_note } : {}),
      vram_idle_mib: mc.idle_vram_mib ?? null,
      fits_in_vram: mc.route1?.ps_size_bytes == null || m.template === 'gemma4' ? (mc.route1 ? (Math.max(mc.route1.vram_max_mib ?? 0, mc.ex1?.vram_max_mib ?? 0) < 24000) : mc.loaded_vram_mib != null ? mc.loaded_vram_mib < 24000 : null) : mc.route1.ps_size_vram_bytes >= mc.route1.ps_size_bytes,
      load_s: mc.load_wall_ms_median != null ? +(mc.load_wall_ms_median / 1000).toFixed(1) : null,
      measured_by_t12_driver: !!mc.route1,
    });
  }
  for (const p of mc.probe || []) {
    const rows = readJsonl(path.join(ROOT, p.file));
    probeOut.push(`| ${m.arm} | \`${JSON.stringify(p.prefix)}\` | ${rows.length} | ${p.median_mass} | ${rows.filter(r => (r.mass ?? 0) < 0.5).length} | ${[...new Set(rows.map(r => r.first_token))].slice(0, 4).map(t => `\`${JSON.stringify(t)}\``).join(' ')} | \`${p.file}\` |`);
  }
  if (mc.template_check) probeOut.push(`| ${m.arm} | template check | | | | chat ${mc.template_check.chat_prompt_tokens} tokens, raw ${mc.template_check.raw_prompt_tokens} (diff ${mc.template_check.diff}) | |`);
}

const md = [];
md.push('# T12: local open model series, one table\n');
md.push('Generated by `bench/score-t12.mjs` from `results/route1/score.md`, `results/ex1/score.md`, the run rows and `results/t12/machine.json`. Design: `plan/t12.md`. Run 1 of every arm, Ollama on one RTX 4090 (24 GB), called from a Mac over the LAN.\n');
md.push('Accuracy columns: route1 groups of the score report (scen = MASSIVE scenario, int2 = MASSIVE intent with 2+ options, fd26 = fast-decisions tasks with 26 labels or fewer). A value in parentheses belongs to a model with a format failure (median letter mass below 0.5) and is not a result. F1 and frame = ex1 slot micro F1 and exact frame. mass = median letter mass over the full route1 run. prefix = assistant prefix chosen by the probe. p50 = median ms per call as seen from the Mac (LAN included). per h = route1 decisions or ex1 utterances per hour, one stream, wall time first to last row. VRAM = peak nvidia-smi memory.used during the runs minus the idle value before loading (GiB). Ollama = size_vram/size of the loaded model per /api/ps at the end of route1 (GiB; equal means fully on the GPU). load = median of three loads after an unload, wall time from the Mac, file already in the OS cache (s). Ollama n/a: /api/ps under Ollama 0.35.0 reports 0.3 to 1.3 GiB for the Gemma 4 builds, far below their measured VRAM, so the column is not used for them.\n');
md.push(`| class | arm | quant | digest | ${ROUTE_SECS.map(([, h]) => h).join(' | ')} | F1 de | F1 en | frame de | frame en | ex1 errors | mass | prefix | p50 ms route1 | p50 ms ex1 | decisions per h | utterances per h | VRAM GiB | Ollama GiB | load s |`);
md.push(`|---|---|---|---|${ROUTE_SECS.map(() => '---').join('|')}|---|---|---|---|---|---|---|---|---|---|---|---|---|---|`);
for (const r of rowsOut) md.push(`| ${r.cls} | ${r.arm} | ${fmt(r.quant)} | ${r.digest} | ${r.route.join(' | ')} | ${r.f1de} | ${r.f1en} | ${r.frde} | ${r.fren} | ${r.exErr} | ${r.mass} | ${r.prefix == null ? '' : `\`${JSON.stringify(r.prefix)}\``} | ${fmt(r.p50r)} | ${fmt(r.p50e)} | ${fmt(r.dph)} | ${fmt(r.uph)} | ${r.vram} | ${r.ps} | ${r.load} |`);
md.push('');
md.push('winnow-12b was measured before T12 (2026-10-09, `results/route1/method.md`); its machine columns are empty because the T12 driver did not run it.\n');
fs.mkdirSync(path.join(ROOT, 'results/t12'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'results/t12/series.md'), md.join('\n') + '\n');
fs.writeFileSync(path.join(ROOT, 'results/t12/summary.json'), JSON.stringify({
  schema: 'jev-match-bench t12 summary v1', design: 'plan/t12.md', report: 'results/t12-findings.md',
  note: 'One entry per local model, chart-ready. route1.accuracy is pooled over every route1 question that was asked (MASSIVE scenario and intent de/en, fast-decisions en): the 100 support_intent:intent questions have 28 options, more than the 26 letters, and are written as unanswered by design for every local model, so they are left out (route1.excluded_unanswered_by_design). accuracy_groups are the score report groups. letter_mass_below_05_share: share of asked questions whose answer letters had less than half of the probability among the top 20 tokens. wall_s: run time of the task from the T12 driver (first to last row for winnow-12b, measured before T12); items_per_s = items / wall_s, one stream, from a Mac over the LAN to Ollama 0.35.0 on one RTX 4090 (24 GB). vram_max_mib: peak nvidia-smi memory.used during the runs, including about 1.7 to 3.3 GiB of the desktop (vram_idle_mib). fits_in_vram: Ollama /api/ps size_vram equals size (all layers on the GPU); for the Gemma 4 builds /api/ps is not usable under Ollama 0.35.0, so it is peak VRAM below 24,000 MiB. ex1 F1 in percent, micro over slots.',
  models: summary,
}, null, 1) + '\n');

const pm = ['# T12: letter-mass probes and template checks\n', 'Per model before its full route1 run (`plan/t12.md`): 20 route1 questions (`--per-group 4`), first with prefix `""`, then, if the median letter mass was below 0.5, once with `"Answer: **"`. The template check sends the first probe question once through `/api/chat` (Ollama renders its own template) and once as the runner\'s raw prompt, each from a cold load, and compares the prompt token counts.\n', '| arm | prefix | questions | median mass | mass < 0.5 | first tokens (up to 4 distinct) | file |', '|---|---|---|---|---|---|---|', ...probeOut, ''];
fs.writeFileSync(path.join(ROOT, 'results/t12/probes.md'), pm.join('\n') + '\n');
console.log(md.join('\n'));
