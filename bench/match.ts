// VENDORED from mboiman/mboiman.github.io src/lib/match.ts at commit 6d57850.
// The bench asks exactly the questions the service asks (README, principle 3).
// Do not edit here: update by copying again and recording the new commit.

/**
 * The project match (/match/): does Michael fit a project?
 *
 * Built the way TypeSafe says Jev wants to be used: many small, typed
 * questions, and the verdict composed in code. Jev never writes the score. It
 * answers, per requirement, which CV entry shows it, how well (a level from 0
 * to 3), whether the line is mandatory and which area it belongs to. Per
 * project it says how central each area is, and once per CV how deep Michael
 * is in each area. Everything after that is this file: plain arithmetic whose
 * weights the page lets a visitor move without asking Jev again.
 *
 * One module for three callers, so they cannot drift apart:
 *   scripts/jev-measure.mjs   measures the demo projects and the profile
 *   workers/jev-match/        answers live requests for pasted text
 *   CVMatch.astro             replays and scores in the browser
 *
 * The questions are English over German or English text. The CV goes into the
 * state once, and the evidence options point at it by path (`cv.s-dvag`), so
 * the entries are not repeated per option (measured 2026-09-26: 540 input
 * tokens for a two-entry probe, 0.3 s).
 */

export type Lang = 'de' | 'en';

export interface Axis { key: string; label: Record<Lang, string>; jev: string }

/** The eight areas of the net diagram. `jev` is what Jev reads. */
export const AXES: Axis[] = [
  { key: 'testauto', label: { de: 'Testautomatisierung', en: 'Test automation' },
    jev: 'test automation: automated UI, API and end-to-end tests with frameworks such as Playwright, Selenium, pytest or BDD' },
  { key: 'perf', label: { de: 'Last und Performance', en: 'Load and performance' },
    jev: 'load and performance testing: JMeter, Gatling, LoadRunner, performance analysis' },
  { key: 'qm', label: { de: 'Qualitätsstrategie', en: 'Quality strategy' },
    jev: 'quality strategy and test management: test strategy, test process, test planning, acceptance, defect management' },
  { key: 'ai', label: { de: 'KI und LLMs', en: 'AI and LLMs' },
    jev: 'AI and LLM engineering: building with large language models, RAG, prompt design, AI-assisted testing or evaluation' },
  { key: 'agents', label: { de: 'Agenten und MCP', en: 'Agents and MCP' },
    jev: 'AI agents and orchestration: multi-agent systems, agentic coding tools, MCP, A2A, tool use' },
  { key: 'cloud', label: { de: 'Cloud und CI/CD', en: 'Cloud and CI/CD' },
    jev: 'cloud, DevOps and CI/CD: Azure or AWS, containers, pipelines, infrastructure, monitoring' },
  { key: 'dev', label: { de: 'Entwicklung', en: 'Development' },
    jev: 'software development and integration: production code in Python, TypeScript, Java or Kotlin, APIs and system interfaces' },
  { key: 'lead', label: { de: 'Beratung und Führung', en: 'Consulting and lead' },
    jev: 'consulting and leadership: advising clients, workshops, coaching, leading or building teams, working with stakeholders' },
];

/** What kind of role a posting describes; labels live in i18n (match.roles). */
export const ROLES: Record<string, string> = {
  test_automation: 'A test automation engineer or SDET',
  qa_lead: 'A QA lead, test manager or quality engineering lead',
  performance: 'A performance or load test engineer',
  ai_engineer: 'An AI, LLM or agent engineer',
  developer: 'A software developer or engineer',
  consultant: 'A consultant or architect',
  other: 'Something else',
};

export const MODEL = 'jev-latest';

const DEMAND_LEVELS = ['Not asked for', 'Mentioned in passing or nice to have', 'Clearly required', 'Central to the role'];
// Anchored on time and number of roles: with softer levels ("some exposure",
// "deep") Jev put every area of this CV at the top (measured 2026-09-26, all
// eight between 0.88 and 1.0), and a net diagram with every spoke at full
// length says nothing.
const DEPTH_LEVELS = [
  'Not shown in `cv`',
  'Shown once or briefly, in one role or side project',
  'Several roles or projects, under five years in total',
  'Main work for more than five years, across many roles',
];
const COVER_LEVELS = ['Not at all', 'Adjacent or transferable experience', 'Done in at least one role or project', 'Done repeatedly, across several roles or projects'];

export interface JevRequest {
  model: string;
  state: Record<string, unknown>;
  questions: Record<string, { type: 'noul' | 'choice' | 'score'; instructions: string; criteria?: unknown }>;
}

// ── The CV as Jev sees it ──────────────────────────────────────────────────

export interface CvEntry {
  /** s- station or talk, p- project, x- profile. Also a choice option key. */
  id: string;
  kind: 'station' | 'project' | 'profile';
  /** The page anchor: #exp-<anchor> or #project-<anchor>; empty for the profile. */
  anchor: string;
  label: string;
  text: string;
}

/** Markdown of config.cv.toml to one line of plain text, cut at a word. */
export function plain(md: string, max = 600): string {
  const s = (md || '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_`#]/g, '')
    .replace(/^\s*[-•·]\s*/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (s.length <= max) return s;
  const cut = s.slice(0, max);
  return `${cut.slice(0, cut.lastIndexOf(' ') > max * 0.6 ? cut.lastIndexOf(' ') : max).trim()} …`;
}

interface CvParams {
  profile: { tagline: string };
  summary?: { summary?: string };
  education?: { list: { degree: string; college?: string; dates: string }[] };
  language?: { list: { language: string; level: string }[] };
  experiences: { list: { position: string; company: string; dates: string; details: string; anchor?: string }[] };
  projects: { list: { title: string; tagline: string; tech_stack?: string[]; anchor?: string; year?: number; category?: string }[] };
}

export function cvEntries(p: CvParams): CvEntry[] {
  const out: CvEntry[] = [];
  const profile = [
    p.profile.tagline,
    p.summary?.summary ? plain(p.summary.summary, 300) : '',
    (p.education?.list || []).map(e => [e.degree, e.college, e.dates].filter(Boolean).join(', ')).join('; '),
    (p.language?.list || []).map(l => `${l.language} (${l.level})`).join(', '),
  ].filter(Boolean).join('. ');
  out.push({ id: 'x-profile', kind: 'profile', anchor: '', label: p.profile.tagline, text: plain(profile, 690) });
  for (const e of p.experiences.list) {
    if (!e.anchor) continue;
    const head = `${e.position}, ${e.company.split(' · ')[0]} (${e.dates})`;
    out.push({ id: `s-${e.anchor}`, kind: 'station', anchor: e.anchor, label: `${e.position} · ${e.company.split(' · ')[0]}`, text: plain(`${head}: ${e.details}`) });
  }
  for (const pr of p.projects.list) {
    if (!pr.anchor) continue;
    const tech = pr.tech_stack?.length ? ` Stack: ${pr.tech_stack.join(', ')}.` : '';
    out.push({ id: `p-${pr.anchor}`, kind: 'project', anchor: pr.anchor, label: pr.title, text: plain(`${pr.title}${pr.year ? ` (${pr.year})` : ''}: ${plain(pr.tagline, 520)}${tech}`) });
  }
  return out;
}

/** FNV-1a over the entries: tells a stored measurement it is older than the CV. */
export function entriesHash(entries: CvEntry[]): string {
  let h = 0x811c9dc5;
  const s = JSON.stringify(entries.map(e => [e.id, e.text]));
  for (let i = 0; i < s.length; i += 1) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

// ── A posting to requirement lines ─────────────────────────────────────────

const BULLET = /^\s*(?:[-*•·▪►–]|\d{1,2}[.)])\s+/;
const HEADING = /^(?:ihre?\s+)?(?:profil|aufgaben|anforderungen|qualifikation(?:en)?|kenntnisse|das bringen sie mit|was sie mitbringen|wir bieten|das bieten wir|benefits|über uns|about us|(?:your\s+)?(?:profile|tasks|requirements|responsibilities|qualifications)|what we offer|what you bring)$/i;

/**
 * Lines as the posting wrote them. A bullet is a line; a paragraph without
 * bullets splits into sentences. Anything ending in a colon is a heading, and
 * so is a short line that names a section ("Ihr Profil") or introduces a
 * bullet list. Other short lines stay: a Word list or a list copied from a web
 * page arrives without bullet glyphs, and "Fließend Englisch" is a requirement
 * (review 2026-09-26). Whether a line is a requirement at all is Jev's call
 * (is_req), not a keyword list's.
 *
 * Over `max`, bullets win: a long company introduction must not use up the
 * budget before the list of requirements at the end. The kept lines stay in
 * the posting's order.
 */
export function splitRequirements(text: string, max = 16): string[] {
  const found: { text: string; bullet: boolean }[] = [];
  const seen = new Set<string>();
  const add = (raw: string, bullet: boolean) => {
    const s = raw.replace(/\s+/g, ' ').trim();
    if (s.length < 8 || s.split(' ').length < 2) return;
    const cut = s.length > 300 ? `${s.slice(0, 297).replace(/\s+\S*$/, '')} …` : s;
    const key = cut.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    found.push({ text: cut, bullet });
  };
  const lines = (text || '').replace(/\r/g, '').split('\n').map(l => l.trim()).filter(Boolean);
  lines.forEach((t, i) => {
    if (BULLET.test(t)) { add(t.replace(BULLET, ''), true); return; }
    if (t.endsWith(':')) return;
    const short = t.split(/\s+/).length <= 3 && !/[.!?]$/.test(t);
    if (short && (HEADING.test(t) || BULLET.test(lines[i + 1] ?? ''))) return;
    const sentences = t.length > 160 ? t.split(/(?<=[.!?])\s+(?=[A-ZÄÖÜ])/) : [t];
    sentences.forEach(x => add(x, false));
  });
  if (found.length <= max) return found.map(f => f.text);
  const bullets = found.filter(f => f.bullet).length;
  let prose = Math.max(0, max - bullets);
  let kept = 0;
  const out: string[] = [];
  for (const f of found) {
    if (kept >= max) break;
    if (!f.bullet) { if (prose <= 0) continue; prose -= 1; }
    out.push(f.text); kept += 1;
  }
  return out;
}

// ── Requests ───────────────────────────────────────────────────────────────

/** One per posting: how central each area is, and what kind of role it is. */
export function projectRequest(text: string): JevRequest {
  const questions: JevRequest['questions'] = {};
  for (const a of AXES) {
    questions[`demand:${a.key}`] = { type: 'score', instructions: `How important is ${a.jev} for the work described in \`project\`?`, criteria: DEMAND_LEVELS };
  }
  questions.role = { type: 'choice', instructions: 'What kind of role does `project` describe?', criteria: ROLES };
  return { model: MODEL, state: { project: text.slice(0, 12000) }, questions };
}

/** One per requirement line, with the whole CV in the state. */
export function requirementRequest(requirement: string, entries: CvEntry[]): JevRequest {
  const cv = Object.fromEntries(entries.map(e => [e.id, e.text]));
  const evidence: Record<string, string> = Object.fromEntries(entries.map(e => [e.id, `\`cv.${e.id}\``]));
  evidence.none = 'No entry of `cv` shows it';
  const axis: Record<string, string> = Object.fromEntries(AXES.map(a => [a.key, a.jev]));
  axis.other = 'None of these areas';
  return {
    model: MODEL,
    state: { requirement, cv },
    questions: {
      is_req: { type: 'noul', instructions: 'Does `requirement` describe something the candidate must know, have done or will do (a skill, experience, qualification or task), rather than describing the company, its benefits or the application process?' },
      must: { type: 'noul', instructions: 'Is `requirement` phrased as mandatory rather than optional or nice to have?' },
      axis: { type: 'choice', instructions: 'Which area does `requirement` belong to?', criteria: axis },
      evidence: { type: 'choice', instructions: 'Which entry of `cv` best shows that the candidate meets `requirement`?', criteria: evidence },
      level: { type: 'score', instructions: 'How well does `cv` show that the candidate has done what `requirement` asks for?', criteria: COVER_LEVELS },
    },
  };
}

/** Once per CV: how deep Michael is in each area. Measured, then stored. */
export function profileRequest(entries: CvEntry[]): JevRequest {
  const questions: JevRequest['questions'] = {};
  for (const a of AXES) {
    questions[`depth:${a.key}`] = { type: 'score', instructions: `How much hands-on experience with ${a.jev} does \`cv\` show?`, criteria: DEPTH_LEVELS };
  }
  return { model: MODEL, state: { cv: Object.fromEntries(entries.map(e => [e.id, e.text])) }, questions };
}

// ── Answers ────────────────────────────────────────────────────────────────

/** What the API returns, plus the wall time the caller measured. */
export interface JevAnswer {
  answers: Record<string, { noul?: number; choice?: string; score?: number; probabilities?: Record<string, number> }>;
  usage?: { input_tokens?: number };
  model?: string;
  _seconds?: number;
}

export interface RequirementResult {
  text: string;
  isReq: number;
  must: number;
  axis: string;
  evidence: string | null;
  evidenceP: number;
  /** Jev's level, 0..3 */
  level: number;
  /** level / 3 */
  cover: number;
  /** Counted in the score: Jev says it is a requirement. */
  counts: boolean;
  seconds: number;
  tokens: number;
}

const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;

export function readRequirement(text: string, a: JevAnswer): RequirementResult {
  const x = a.answers;
  const ev = x.evidence?.choice ?? 'none';
  const level = Math.max(0, Math.min(3, x.level?.score ?? 0));
  const isReq = x.is_req?.noul ?? 0;
  return {
    text,
    isReq: round(isReq),
    must: round(x.must?.noul ?? 0),
    axis: x.axis?.choice ?? 'other',
    evidence: ev === 'none' ? null : ev,
    evidenceP: round(x.evidence?.probabilities?.[ev] ?? 0),
    level,
    cover: level / 3,
    counts: isReq >= 0.5,
    seconds: a._seconds ?? 0,
    tokens: a.usage?.input_tokens ?? 0,
  };
}

export function readProject(a: JevAnswer): { demand: Record<string, number>; role: string; seconds: number; tokens: number } {
  const demand = Object.fromEntries(AXES.map(ax => [ax.key, round((a.answers[`demand:${ax.key}`]?.score ?? 0) / 3)]));
  return { demand, role: a.answers.role?.choice ?? 'other', seconds: a._seconds ?? 0, tokens: a.usage?.input_tokens ?? 0 };
}

export function readProfile(a: JevAnswer): Record<string, number> {
  return Object.fromEntries(AXES.map(ax => [ax.key, round((a.answers[`depth:${ax.key}`]?.score ?? 0) / 3)]));
}

// ── The score ──────────────────────────────────────────────────────────────

export interface Weights {
  /** Share of the requirement coverage in the total; the rest is topic fit. */
  reqShare: number;
  /** How much more a mandatory line weighs than an optional one. */
  mustFactor: number;
}
export const DEFAULT_WEIGHTS: Weights = { reqShare: 0.7, mustFactor: 2 };

/** Below this demand an area does not count toward topic fit. */
const DEMAND_FLOOR = 0.2;
/** From this level on a mandatory line counts as met: "done in one role". */
export const MET_LEVEL = 1.5;

export interface AxisScore {
  /** How central the area is to the project, 0..1 (Jev, project request). */
  demand: number;
  /** How well the project's lines in this area are shown in the CV, 0..1. */
  cover: number;
  /** false: no line of the posting fell into this area, so `cover` is the
   *  stored profile depth instead of this project's evidence. */
  fromLines: boolean;
}

export interface Score {
  total: number;
  coverage: number;
  topicFit: number;
  mustMet: number;
  mustTotal: number;
  counted: number;
  perAxis: Record<string, AxisScore>;
  gaps: RequirementResult[];
}

/**
 * Two numbers, then a weighted mean of them.
 *
 * coverage: every counted line weighs 1, a mandatory one `mustFactor`; the
 * line's value is Jev's level over 3.
 *
 * topicFit: per area, how well the lines of that area are covered, weighted by
 * how central the area is to the project. An area the project asks for without
 * any line of its own falls back to the profile depth. The first version used
 * the profile depth everywhere, and Jev put this CV near the top in all eight
 * areas, so every project scored full marks on topics (2026-09-26).
 */
export function scoreMatch(reqs: RequirementResult[], demand: Record<string, number>, profile: Record<string, number>, w: Weights): Score {
  const counted = reqs.filter(r => r.counts);
  const weight = (r: RequirementResult) => (r.must >= 0.5 ? w.mustFactor : 1);
  const wSum = counted.reduce((s, r) => s + weight(r), 0);
  const coverage = wSum ? counted.reduce((s, r) => s + weight(r) * r.cover, 0) / wSum : 0;

  const perAxis: Score['perAxis'] = {};
  let dSum = 0; let fSum = 0;
  for (const a of AXES) {
    const d = demand[a.key] ?? 0;
    const lines = counted.filter(r => r.axis === a.key);
    const cover = lines.length ? lines.reduce((s, r) => s + r.cover, 0) / lines.length : (profile[a.key] ?? 0);
    perAxis[a.key] = { demand: d, cover, fromLines: lines.length > 0 };
    if (d >= DEMAND_FLOOR) { dSum += d; fSum += d * cover; }
  }
  const topicFit = dSum ? fSum / dSum : 0;

  const hasReqs = wSum > 0; const hasTopics = dSum > 0;
  const share = hasReqs && hasTopics ? w.reqShare : hasReqs ? 1 : 0;
  const total = hasReqs || hasTopics ? Math.round(100 * (share * coverage + (1 - share) * topicFit)) : 0;

  const musts = counted.filter(r => r.must >= 0.5);
  const gaps = counted.filter(r => r.level < MET_LEVEL).sort((a, b) => weight(b) - weight(a) || a.level - b.level);
  return { total, coverage, topicFit, mustMet: musts.filter(r => r.level >= MET_LEVEL).length, mustTotal: musts.length, counted: counted.length, perAxis, gaps };
}

export type Verdict = 'strong' | 'good' | 'partial' | 'weak';
export function verdictOf(total: number): Verdict {
  return total >= 85 ? 'strong' : total >= 70 ? 'good' : total >= 50 ? 'partial' : 'weak';
}

/** A stored or live run, the shape the page replays. */
export interface Measurement {
  lang: Lang;
  measuredAt: string;
  model: string;
  project: { demand: Record<string, number>; role: string; seconds: number; tokens: number };
  requirements: RequirementResult[];
}
