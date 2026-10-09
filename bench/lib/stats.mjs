/** Exact two-sided binomial McNemar, computed in log space. */
export function mcnemar(b, c) {
  const n = b + c;
  if (n === 0) return 1;
  const lf = k => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; };
  let tail = 0;
  for (let i = 0; i <= Math.min(b, c); i++) tail += Math.exp(lf(n) - lf(i) - lf(n - i) - n * Math.log(2));
  return Math.min(1, 2 * tail);
}
