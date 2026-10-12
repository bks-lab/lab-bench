"""C5 scorer (plan/c5.md, Metrics), on the Mac after fetching the runs.

  python bench/c5/score_c5.py --data .cache/c5 arm=<out dir> ... [--comet <comet out dir>] [--md results/c5/score.md] [--json ...]

Per arm and direction: chrF++ (primary) and BLEU (13a) with sacrebleu, the
copy baseline (source as translation), the mean COMET from bench/c5/comet_score.py
when given, and sacrebleu's paired bootstrap (1,000 resamples, seed 20261009
via SACREBLEU_SEED) of every arm against the best arm of that direction,
which also gives the 95 % interval. Segments per second at batch 1 and, for
dedicated MT, at batch 32 come from meta.json; GPU numbers from the queue's
gpu.csv are added by the C scorers' shared code later.
"""
import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("SACREBLEU_SEED", "20261009")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from run_c5 import DIRS, SRC, TGT, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, ".cache", "c5"))
    ap.add_argument("--comet")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    from sacrebleu.metrics import BLEU, CHRF  # noqa: PLC0415
    from sacrebleu.significance import PairedTest  # noqa: PLC0415
    _, ids, text = load(a.data, ROOT, 0)
    runs = {}
    for spec in a.arms:
        arm, d = spec.split("=", 1)
        runs[arm] = ([json.loads(x) for x in open(os.path.join(d, "result.jsonl"), encoding="utf-8")],
                     json.load(open(os.path.join(d, "meta.json"), encoding="utf-8")))
    comet = {}
    if a.comet:
        for x in open(os.path.join(a.comet, "comet.jsonl"), encoding="utf-8"):
            r = json.loads(x)
            comet.setdefault((r["arm"], r["dir"]), []).append(r["comet"])
    out = {"design": "plan/c5.md", "seed": os.environ["SACREBLEU_SEED"], "directions": {}}
    lines = ["# C5 score", ""]
    for d in DIRS:
        refs = [text[TGT[d]]]
        systems = {"copy": text[SRC[d]]}
        for arm, (rows, _) in runs.items():
            hyp = {r["id"]: r["hyp"] for r in rows if r["dir"] == d}
            systems[arm] = [hyp.get(i, "") for i in ids]
        chrf = {k: CHRF(word_order=2).corpus_score(v, refs).score for k, v in systems.items()}
        best = max((k for k in systems if k != "copy"), key=lambda k: chrf[k])
        order = [best] + [k for k in systems if k != best]
        pt = PairedTest([(k, systems[k]) for k in order], {"chrF++": CHRF(word_order=2), "BLEU": BLEU()}, refs,
                        n_samples=1000, test_type="bs")
        sigs, scores = pt()
        res = {}
        mk = [m for m in scores if m != "System"]  # sacrebleu names them itself (chrF2++, BLEU), in the order given
        for k in order:
            c, b = scores[mk[0]][order.index(k)], scores[mk[1]][order.index(k)]
            res[k] = {"chrf++": c.score, "chrf++_mean": c.mean, "chrf++_ci": c.ci, "chrf++_p_vs_best": c.p_value,
                      "bleu": b.score, "bleu_p_vs_best": b.p_value,
                      "comet": statistics.mean(comet[(k, d)]) if (k, d) in comet else None}
            if k in runs:
                secs = [r["seconds"] for r in runs[k][0] if r["dir"] == d]
                res[k]["segments_per_s_batch1"] = len(secs) / sum(secs) if sum(secs) else None
                b32 = next((s for s in runs[k][1]["sections"] if s["name"] == f"batch32-{d}"), None)
                res[k]["segments_per_s_batch32"] = b32.get("segments_per_s") if b32 else None
                res[k]["errors"] = sum(1 for r in runs[k][0] if r["dir"] == d and r.get("error"))
        out["directions"][d] = {"best": best, "signatures": {k: str(v) for k, v in sigs.items()}, "arms": res}
        lines += [f"## {d}", "", "| arm | chrF++ | 95 % CI | p vs best | BLEU | COMET | seg/s b1 | seg/s b32 |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted(res, key=lambda k: -res[k]["chrf++"]):
            r = res[k]
            f = lambda v, n=1: "" if v is None else f"{v:.{n}f}"  # noqa: E731
            lines.append(f"| {k} | {r['chrf++']:.1f} | ±{f(r['chrf++_ci'])} | {f(r['chrf++_p_vs_best'], 3)} | {r['bleu']:.1f} | "
                         f"{f(r['comet'], 3)} | {f(r.get('segments_per_s_batch1'), 2)} | {f(r.get('segments_per_s_batch32'), 2)} |")
        lines.append("")
    text_md = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text_md)
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, default=float)
    print(text_md)


if __name__ == "__main__":
    main()
