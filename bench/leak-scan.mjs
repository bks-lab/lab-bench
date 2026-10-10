#!/usr/bin/env node
/**
 * Leak scan of this repository. CI runs it on every push and pull request
 * (.github/workflows/ci.yml); run it locally before a push:
 *
 *   node bench/leak-scan.mjs [dir] [--config bench/leak-scan.json]
 *   node bench/leak-scan.mjs --hash <word>     hash line for leak-scan.json
 *
 * Without <dir> it scans the repository it lives in. In a git work tree it
 * reads the tracked files plus untracked ones that are not ignored (so a
 * gitignored bench/local.env or .cache/ is not scanned, but a new file is);
 * elsewhere it walks the folder. Seven checks:
 *
 * 0. paths: every file must match `paths.include` of the config, so a new
 *    kind of file is noticed before it is published.
 * 1. terms: every token of every text file, lower-cased, and every part of
 *    it split at '-', '_' and '.', is hashed with SHA-256 and compared with
 *    `terms.sha256` (host, account, tailnet and keychain names of the lab
 *    machines). The config holds only the hashes, so it does not publish
 *    the words it guards. Words from LEAK_SCAN_TERMS (environment, or
 *    bench/local.env) are checked as well, in plain text, on the local
 *    machine only. No allow entry lifts a term hit.
 * 2. text: every line of every text file against the patterns below
 *    (tailnet domains, private IP ranges, home paths, mail domains, keychain
 *    lookups, Windows account SIDs, API key shapes). Allowed only by a
 *    `leaks` entry with the same pattern, a path regex, optionally a regex
 *    for the matched text, and a reason.
 * 3. structured: every .json, .jsonl, .csv, .yaml and .yml file is parsed,
 *    and every record is walked. A record fails when one of the fields arm,
 *    arm_version, arm_revision, host, model, provider, endpoint or base_url
 *    names Jev or TypeSafe, or when any key itself does ("jev": {...}).
 *    requests/ is exempt (the request bodies are the questions, not
 *    answers). Other hits only by a `structured` entry with a path regex,
 *    optionally a regex `at` for the place in the record ("$.scripts.x"),
 *    and a reason. A file that does not parse fails.
 * 4. binary: any binary file under results/ or reference/ fails, since it
 *    cannot be checked.
 * 5. Jev rows: every row of a file matching `jevRows.path` of the config
 *    (results/<family>/jev/*.jsonl, public since TypeSafe's permission of
 *    2026-10-10) may only carry the fields of `jevRows.fields`. A request
 *    id, header, account or raw response field fails.
 * 6. Jev notice: LICENSE-JEV-OUTPUTS must exist at the root, and every
 *    directory with Jev rows must also hold NOTICE.md.
 *
 * Exits 1 on any hit that is not allowed.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { parseAllDocuments } from 'yaml';

const args = process.argv.slice(2);
const sha256 = w => crypto.createHash('sha256').update(w).digest('hex');
const hi = args.indexOf('--hash');
if (hi >= 0) {
  const w = (args[hi + 1] || '').trim().toLowerCase();
  if (!w) { console.error('usage: node bench/leak-scan.mjs --hash <word>'); process.exit(2); }
  console.log(JSON.stringify({ sha256: sha256(w), reason: '<why this word must not be published>' }));
  process.exit(0);
}
const HERE = path.dirname(new URL(import.meta.url).pathname);
const ci = args.indexOf('--config');
const CONFIG_FILE = ci >= 0 ? args[ci + 1] : path.join(HERE, 'leak-scan.json');
const DIR = args.find((a, k) => !a.startsWith('--') && args[k - 1] !== '--config') || path.join(HERE, '..');

/** LEAK_SCAN_TERMS from the environment or bench/local.env (never committed). */
function localTerms() {
  let raw = process.env.LEAK_SCAN_TERMS || '';
  const env = path.join(HERE, 'local.env');
  if (!raw && fs.existsSync(env)) {
    const m = fs.readFileSync(env, 'utf8').match(/^\s*LEAK_SCAN_TERMS=(.*)$/m);
    if (m) raw = m[1].trim().replace(/^['"]|['"]$/g, '');
  }
  return raw.split(',').map(x => x.trim().toLowerCase()).filter(Boolean);
}

export const PATTERNS = {
  host: /\.ts\.net\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b/i,
  'home-path': /\/Users\/|\/home\/[a-z]|\/private\/tmp\/|C:\\Users\\|~\/Developer/i,
  mail: /gmail|googlemail|@bks-lab\.com/i,
  address: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/,
  keychain: /TYPESAFE_KEYCHAIN_(SERVICE|ACCOUNT)=\S|find-generic-password\s+-s\s+\S/i,
  sid: /\bS-1-5-21-\d+-\d+-\d+-\d+\b/,
  'api-key': /\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{20,}|\bgho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|\bhf_[A-Za-z0-9]{20,}|\bAKIA[0-9A-Z]{16}\b|\bxox[abps]-[A-Za-z0-9-]{10,}|Bearer\s+[A-Za-z0-9._-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY|tskey-[A-Za-z0-9-]{10,}|(api[_-]?key|secret|token)["']?\s*[:=]\s*["'][A-Za-z0-9_\-]{16,}["']/,
};

// structured check: a word "jev" (jev, jev-1.13.0, jev_latest) or typesafe
const NAMES_JEV = s => typeof s === 'string' && /(^|[^a-z])jev([^a-z]|$)|typesafe/i.test(s);
const ARM_FIELDS = new Set(['arm', 'arm_version', 'arm_revision', 'host', 'model', 'provider', 'endpoint', 'base_url']);
const TEXT_EXT = /\.(md|json|jsonl|csv|tsv|ya?ml|txt|mjs|js|ts|py|sh|bat|html|xml|example|gitignore|gitattributes)$|(^|\/)(LICENSE[^/]*|\.gitignore|\.gitattributes)$/;

const cfg = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
const compile = (list, kind) => (list || []).map(a => {
  if (!a.reason || !a.path) { console.error(`${kind} entry without path or reason: ${JSON.stringify(a)}`); process.exit(2); }
  return { ...a, pathRe: new RegExp(a.path), matchRe: a.match ? new RegExp(a.match, 'i') : null };
});
const leakAllow = compile(cfg.leaks, 'leaks');
for (const a of leakAllow) if (!PATTERNS[a.pattern]) { console.error(`leaks entry with unknown pattern: ${JSON.stringify(a)}`); process.exit(2); }
const structAllow = compile(cfg.structured, 'structured');
const pathInclude = compile(cfg.paths?.include, 'paths.include');
if (!pathInclude.length) { console.error('config without paths.include'); process.exit(2); }
const termHashes = new Set((cfg.terms?.sha256 || []).map(t => {
  if (!/^[0-9a-f]{64}$/.test(t.sha256 || '') || !t.reason) { console.error(`terms entry without sha256 or reason: ${JSON.stringify(t)}`); process.exit(2); }
  return t.sha256;
}));
const plainTerms = new Set(localTerms());
const jevRows = cfg.jevRows;
if (!jevRows?.path || !Array.isArray(jevRows.fields)) { console.error('config without jevRows.path and jevRows.fields'); process.exit(2); }
const JEV_ROWS = new RegExp(jevRows.path);
const JEV_FIELDS = new Set(jevRows.fields);
const jevDirs = new Set();

const tokenSeen = new Map();
/** True when a token (or a part of it) is a guarded word. */
function isTerm(tok) {
  let hit = tokenSeen.get(tok);
  if (hit !== undefined) return hit;
  const parts = new Set([tok, ...tok.split(/[-_.]+/).filter(Boolean)]);
  hit = false;
  for (const p of parts) if (plainTerms.has(p) || termHashes.has(sha256(p))) { hit = true; break; }
  if (tokenSeen.size < 2e6) tokenSeen.set(tok, hit);
  return hit;
}

/** Files to scan: git-tracked plus untracked-not-ignored in a work tree, else all. */
function listFiles(dir) {
  try {
    const out = execFileSync('git', ['-C', dir, 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], { encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'ignore'] });
    const top = execFileSync('git', ['-C', dir, 'rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
    if (fs.realpathSync(top) === fs.realpathSync(dir)) return out.split('\0').filter(Boolean).map(f => path.join(dir, f)).filter(f => fs.existsSync(f));
  } catch { /* not a git work tree */ }
  return [...walk(dir)];
}

function* walk(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    if (e.name === '.git' || e.name === 'node_modules') continue;
    const p = path.join(d, e.name);
    if (e.isDirectory()) yield* walk(p); else if (e.isFile()) yield p;
  }
}

const bad = [];
const allowedCount = new Map();
const count = a => allowedCount.set(a, (allowedCount.get(a) || 0) + 1);
let files = 0, parsed = 0;

/** Walks a parsed value; yields [where, reason] for every Jev field or key. */
function* jevFields(v, where) {
  if (Array.isArray(v)) { for (let k = 0; k < v.length; k++) yield* jevFields(v[k], `${where}[${k}]`); return; }
  if (!v || typeof v !== 'object') return;
  for (const [k, x] of Object.entries(v)) {
    if (NAMES_JEV(k)) yield [`${where}.${k}`, `key "${k}"`];
    if (ARM_FIELDS.has(k) && NAMES_JEV(x)) yield [`${where}.${k}`, `${k} = ${JSON.stringify(x)}`];
    yield* jevFields(x, `${where}.${k}`);
  }
}

function csvRecords(text) {
  const rows = []; let row = [], cell = '', q = false;
  for (let k = 0; k < text.length; k++) {
    const ch = text[k];
    if (q) { if (ch === '"' && text[k + 1] === '"') { cell += '"'; k++; } else if (ch === '"') q = false; else cell += ch; continue; }
    if (ch === '"') q = true; else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; } else cell += ch;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [head, ...body] = rows;
  return body.filter(r => r.some(Boolean)).map(r => Object.fromEntries(head.map((h, k) => [h, r[k]])));
}

/** Parsed records of a structured file: [[where, value]]; throws when it does not parse. */
function records(rel, text) {
  if (rel.endsWith('.jsonl')) return text.split('\n').map((l, n) => [n + 1, l]).filter(([, l]) => l.trim()).map(([n, l]) => [`line ${n}`, JSON.parse(l)]);
  if (rel.endsWith('.json')) return [['$', JSON.parse(text)]];
  if (rel.endsWith('.csv')) return csvRecords(text).map((r, n) => [`row ${n + 1}`, r]);
  return parseAllDocuments(text).map((d, n) => { if (d.errors.length) throw d.errors[0]; return [`doc ${n + 1}`, d.toJS()]; });
}

for (const f of listFiles(DIR)) {
  const rel = path.relative(DIR, f).split(path.sep).join('/');
  // 0. positive path list
  if (!pathInclude.some(a => a.pathRe.test(rel))) bad.push(`${rel}: [path] not on paths.include of the config; add a pattern with a reason if it belongs in the repository`);
  const buf = fs.readFileSync(f);
  const binary = buf.subarray(0, 8000).includes(0) || !TEXT_EXT.test(rel);
  if (binary) {
    if (/^(results|reference)\//.test(rel)) bad.push(`${rel}: [binary] not a known text file under results/ or reference/, cannot be checked`);
    continue;
  }
  files++;
  const text = buf.toString('utf8');
  const lines = text.split('\n');
  // 1. guarded words, by hash
  for (let n = 0; n < lines.length; n++) {
    for (const tok of lines[n].toLowerCase().split(/[^a-z0-9_.-]+/)) {
      if (tok && isTerm(tok.replace(/^[.-]+|[.-]+$/g, ''))) bad.push(`${rel}:${n + 1}: [term] a guarded word (host, account, tailnet or keychain name); value not printed`);
    }
  }
  // 2. text patterns
  for (let n = 0; n < lines.length; n++) {
    for (const [id, re] of Object.entries(PATTERNS)) {
      const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g');
      for (const m of lines[n].matchAll(g)) {
        const a = leakAllow.find(x => x.pattern === id && x.pathRe.test(rel) && (!x.matchRe || x.matchRe.test(m[0])));
        if (a) { count(a); continue; }
        const s = Math.max(0, m.index - 40);
        bad.push(`${rel}:${n + 1}: [${id}] ${lines[n].slice(s, m.index + m[0].length + 40)}`);
      }
    }
  }
  // 3. structured records
  if (/\.(jsonl?|csv|ya?ml)$/.test(rel) && !rel.startsWith('requests/')) {
    let recs;
    try { recs = records(rel, text); parsed++; } catch (e) { bad.push(`${rel}: [structured] does not parse (${String(e.message).split('\n')[0]}), cannot be checked`); recs = []; }
    for (const [where, v] of recs) {
      for (const [at, why] of jevFields(v, where)) {
        const a = structAllow.find(x => x.pathRe.test(rel) && (!x.at || new RegExp(x.at).test(at)));
        if (a) { count(a); continue; }
        bad.push(`${rel}: [structured] ${at}: ${why}`);
      }
    }
  }
  // 5. Jev rows: known fields only
  if (JEV_ROWS.test(rel)) {
    jevDirs.add(path.posix.dirname(rel));
    let n = 0;
    for (const l of lines) {
      n++;
      if (!l.trim()) continue;
      let o; try { o = JSON.parse(l); } catch { continue; } // reported by check 2
      const extra = Object.keys(o).filter(k => !JEV_FIELDS.has(k));
      if (extra.length) bad.push(`${rel}:${n}: [jev-row] fields outside jevRows.fields: ${extra.join(', ')}`);
    }
  }
}

// 6. Jev usage notice next to the rows and at the root
for (const d of jevDirs) if (!fs.existsSync(path.join(DIR, d, 'NOTICE.md'))) bad.push(`${d}/: [jev-notice] Jev rows without NOTICE.md`);
if (!fs.existsSync(path.join(DIR, 'LICENSE-JEV-OUTPUTS'))) bad.push('LICENSE-JEV-OUTPUTS: [jev-notice] missing at the repository root');

console.log(`leak scan: ${files} text files under ${DIR}, ${parsed} structured files parsed, ${termHashes.size} hashed and ${plainTerms.size} local guarded words`);
for (const [a, c] of allowedCount) console.log(`  allowed ${String(c).padStart(5)} x [${a.pattern || 'structured'}] ${a.path}: ${a.reason}`);
if (bad.length) {
  console.error(`leak scan: ${bad.length} hits not allowed:`);
  const max = Number(process.env.LEAK_SCAN_MAX || 200);
  for (const b of bad.slice(0, max)) console.error('  ' + b);
  if (bad.length > max) console.error(`  ... ${bad.length - max} more`);
  process.exit(1);
}
console.log('leak scan: clean');
