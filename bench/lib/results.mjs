/**
 * Loading run files of a family. A run is the newest date that has files for
 * that run; every file of that date is merged (split runs per language or
 * unit). A file with any other suffix (-smoke, -probe ...) is a tagged run:
 * it is read only when its tag is asked for, and then only tagged files are
 * read.
 */
import fs from 'node:fs';
import path from 'node:path';

export const readJsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

export function arms(root, pv) {
  const dir = path.join(root, 'results', pv);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(a => fs.statSync(path.join(dir, a)).isDirectory()).sort();
}

/** File-name suffixes a full run may be split by. Any other suffix is a tag. */
export const SPLITS = ['de', 'en', 'line', 'project', 'profile', 'utt'];

export function runRows(root, pv, arm, { run = 1, tag = '' } = {}) {
  const dir = path.join(root, 'results', pv, arm);
  const re = new RegExp(`^\\d{4}-\\d{2}-\\d{2}-run${run}(-[\\w.]+)*\\.jsonl$`);
  // without a tag, only the split suffixes of a full run: language or unit
  const full = new RegExp(`^\\d{4}-\\d{2}-\\d{2}-run${run}(-(${SPLITS.join('|')}))*\\.jsonl$`);
  const files = fs.readdirSync(dir).filter(f => re.test(f)).filter(f => (tag ? f.endsWith(`-${tag}.jsonl`) : full.test(f))).sort();
  if (!files.length) return [];
  const latest = files.at(-1).slice(0, 10);
  return files.filter(f => f.startsWith(latest)).flatMap(f => readJsonl(path.join(dir, f)));
}
