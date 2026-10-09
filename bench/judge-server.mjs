#!/usr/bin/env node
/**
 * The human reference judgment, blind: a local page that shows each posting
 * line with the pair's CV and asks the service's questions, never a model's
 * answer.
 *
 *   node bench/judge-server.mjs [--pv pv1] [--judge human-1] [--port 4350]
 *   open http://localhost:4350/?lang=de (German cases and labels; without it, English)
 *
 * Listens on 127.0.0.1 only: /api/judge writes files and has no login.
 *
 * Every answer is written at once to reference/<pv>/<judge>.<lang>.jsonl, one
 * row per question in the same keys as the results (pair, line, question), so
 * the report can lay the judgment next to every arm. Stop any time; the page
 * resumes at the first unjudged line.
 *
 * Asked per line: is_req, and only if it is a requirement: must, axis,
 * evidence, level. Per posting once: role and demand per area.
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { AXES, ROLES } from './match.ts';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const PV = arg('pv', 'pv1');
const JUDGE = arg('judge', 'human-1');
const PORT = Number(arg('port', '4350'));
const ROOT = path.resolve(import.meta.dirname, '..');
const OUT = path.join(ROOT, 'reference', PV);
fs.mkdirSync(OUT, { recursive: true });

const reqs = fs.readFileSync(path.join(ROOT, 'requests', PV, 'requests.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));

function items(lang) {
  const out = [];
  for (const r of reqs.filter(x => x.lang === lang && x.unit !== 'profile')) {
    if (r.unit === 'project') {
      out.push({ unit: 'project', key: r.line, posting: r.posting, text: r.request.state.project });
    } else {
      out.push({ unit: 'line', key: r.line, pair: r.pair, posting: r.posting, cv: r.cv, text: r.text, cvEntries: r.request.state.cv });
    }
  }
  // project question first, then its pairs' lines
  const order = [];
  for (const p of out.filter(i => i.unit === 'project')) {
    order.push(p);
    order.push(...out.filter(i => i.unit === 'line' && i.posting === p.posting));
  }
  return order;
}

const file = lang => path.join(OUT, `${JUDGE}.${lang}.jsonl`);
function readJudgments(lang) {
  if (!fs.existsSync(file(lang))) return [];
  return fs.readFileSync(file(lang), 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
}
function saveItem(lang, key, rows) {
  const keep = readJudgments(lang).filter(r => r.line !== key);
  fs.writeFileSync(file(lang), [...keep, ...rows].map(r => JSON.stringify(r)).join('\n') + '\n');
}

const page = fs.readFileSync(path.join(import.meta.dirname, 'judge.html'), 'utf8');

http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const lang = url.searchParams.get('lang') === 'de' ? 'de' : 'en';
  const send = (code, body, type = 'application/json; charset=utf-8') => { res.writeHead(code, { 'Content-Type': type }); res.end(typeof body === 'string' ? body : JSON.stringify(body)); };
  if (url.pathname === '/') return send(200, page, 'text/html; charset=utf-8');
  if (url.pathname === '/api/state') {
    return send(200, { pv: PV, judge: JUDGE, lang, items: items(lang), judgments: readJudgments(lang),
      axes: [...AXES.map(a => ({ key: a.key, label: a.label[lang], hint: a.jev })), { key: 'other', label: lang === 'de' ? 'Keiner dieser Bereiche' : 'None of these areas', hint: '' }],
      roles: ROLES });
  }
  if (url.pathname === '/api/judge' && req.method === 'POST') {
    let body = ''; for await (const c of req) body += c;
    const { item, answers, seconds } = JSON.parse(body);
    const ts = new Date().toISOString();
    const base = { pv: PV, unit: item.unit, pair: item.pair ?? null, posting: item.posting, cv: item.cv ?? null, line: item.key, lang, arm: JUDGE, seconds, ts };
    const rows = Object.entries(answers).map(([question, choice]) => ({ ...base, question, choice: choice == null ? null : String(choice) }));
    saveItem(lang, item.key, rows);
    return send(200, { ok: true, saved: rows.length });
  }
  send(404, { error: 'not found' });
}).listen(PORT, '127.0.0.1', () => console.log(`judge ${JUDGE} for ${PV}: http://localhost:${PORT}/?lang=de or ?lang=en  (writes reference/${PV}/${JUDGE}.<lang>.jsonl)`));
