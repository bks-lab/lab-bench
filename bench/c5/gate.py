"""C5 sanity gate (plan/c5.md, Sanity gate): plain chrF of the opus-mt arm per direction.

  python bench/c5/gate.py --run <opus-mt out dir> --data D:\\bench-data\\c5 --out <dir>

Reference: OPUS-MT leaderboard, opus-2020-02-26 on flores101-devtest, chrF
66.70 (de to en) and 63.63 (en to de); both must lie within 1.5 points,
else exit 3 and the other C5 arms do not start (they depend on this job).
Writes <out>/gate.json. Needs sacrebleu (in the C5 venv).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_c5 import DIRS, TGT, load  # noqa: E402

REF = {"de-en": 66.70, "en-de": 63.63}
TOL = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    import sacrebleu  # noqa: PLC0415
    _, ids, text = load(a.data, a.repo, 0)
    rows = [json.loads(x) for x in open(os.path.join(a.run, "result.jsonl"), encoding="utf-8")]
    out = {"reference": REF, "tolerance": TOL, "sacrebleu": sacrebleu.__version__, "directions": {}}
    ok = True
    for d in DIRS:
        hyp = {r["id"]: r["hyp"] for r in rows if r["dir"] == d}
        if len(hyp) != len(ids):
            sys.exit(f"{d}: {len(hyp)} hypotheses for {len(ids)} sentences")
        chrf = sacrebleu.metrics.CHRF()
        s = chrf.corpus_score([hyp[i] for i in ids], [text[TGT[d]]])
        passed = abs(s.score - REF[d]) <= TOL
        ok &= passed
        out["directions"][d] = {"chrf": s.score, "signature": str(chrf.get_signature()), "passed": passed}
    out["passed"] = ok
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "gate.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
