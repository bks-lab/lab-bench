/**
 * Shared pieces of the route1 and ex1 families: label texts, the slot
 * parser, value normalisation and a seeded random generator. Kept in one
 * place so the freeze, build, run and score scripts agree.
 */

/** "alarm_set" -> "alarm: set", "iot_hue_lightchange" -> "iot: hue lightchange". */
export function humanise(name) {
  const i = name.indexOf('_');
  return i < 0 ? name : `${name.slice(0, i)}: ${name.slice(i + 1).replace(/_/g, ' ')}`;
}

/**
 * Label text of a MASSIVE slot type for ex1: underscores become spaces,
 * nothing else ("place_name" -> "place name", "timeofday" stays). The
 * mapping is written to cases/ex1/slot-labels.json by build-route-ex.mjs and
 * read from there by both extraction arms.
 */
export const slotLabel = name => name.replace(/_/g, ' ');

/** MASSIVE slot annotation: "[date : morgen]" -> {type: "date", text: "morgen"}. */
export const SLOT_RE = /\[([^\]:]+?) : ([^\]]+?)\]/g;
export function parseSlots(annotUtt) {
  return [...annotUtt.matchAll(SLOT_RE)].map(m => ({ type: m[1].trim(), text: m[2].trim() }));
}

/** Lowercase, trim, collapse spaces, strip trailing punctuation. */
export function normValue(s) {
  return String(s).toLowerCase().replace(/\s+/g, ' ').trim().replace(/[\s.,;:!?¿¡'"`)\]]+$/u, '').trim();
}

/** mulberry32, the same generator as normalize.mjs shuffled(). */
export function rng(seed) {
  let t = seed >>> 0;
  return () => { t += 0x6d2b79f5; let x = t; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
}

/** Source of a route1 line id: "massive-12-de" -> "massive", "fastdec-news_topic-7-en" -> "fastdec". */
export const sourceOf = line => line.split('-')[0];

/** First n requests of every (source, lang) group, in file order. For smoke runs. */
export function perGroup(reqs, n) {
  const seen = new Map();
  return reqs.filter(r => { const k = `${sourceOf(r.line)}|${r.lang}`; const c = seen.get(k) || 0; seen.set(k, c + 1); return c < n; });
}

/** Median of numbers, null for an empty list. */
export function median(xs) {
  const a = xs.filter(x => x != null && Number.isFinite(x)).sort((p, q) => p - q);
  if (!a.length) return null;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
