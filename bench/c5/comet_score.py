"""C5 COMET pass (plan/c5.md, Metrics): Unbabel/wmt22-comet-da on the GPU, after all arms.

  python bench/c5/comet_score.py --data D:\\bench-data\\c5 --out <dir> arm=<out dir> [arm=<out dir> ...]

One segment-level COMET score per arm, direction and sentence, written to
<out>/comet.jsonl, plus the copy baseline (source as hypothesis). Batch 64,
the library's defaults otherwise. Needs unbabel-comet (in the C5 venv).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_c5 import DIRS, SRC, TGT, load  # noqa: E402

MODEL = ("Unbabel/wmt22-comet-da", "2760a223ac957f30acfb18c8aa649b01cf1d75f2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    from comet import load_from_checkpoint  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    _, ids, text = load(a.data, a.repo, 0)
    path = snapshot_download(MODEL[0], revision=MODEL[1])
    model = load_from_checkpoint(os.path.join(path, "checkpoints", "model.ckpt"))
    os.makedirs(a.out, exist_ok=True)
    meta = {"model": MODEL, "batch": a.batch, "arms": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(a.out, "comet.jsonl"), "w", encoding="utf-8") as fo:
        for spec in a.arms + ["copy=copy"]:
            arm, d0 = spec.split("=", 1)
            rows = [] if arm == "copy" else [json.loads(x) for x in open(os.path.join(d0, "result.jsonl"), encoding="utf-8")]
            for d in DIRS:
                hyp = {r["id"]: r["hyp"] for r in rows if r["dir"] == d}
                src = text[SRC[d]]
                data = [{"src": s, "mt": (s if arm == "copy" else hyp.get(i, "")), "ref": r}
                        for i, s, r in zip(ids, src, text[TGT[d]])]
                t0 = time.perf_counter()
                res = model.predict(data, batch_size=a.batch, gpus=1, progress_bar=False)
                meta["arms"].setdefault(arm, {})[d] = {"system": res.system_score, "seconds": time.perf_counter() - t0}
                for i, s in zip(ids, res.scores):
                    fo.write(json.dumps({"arm": arm, "dir": d, "id": i, "comet": s}) + "\n")
                print(arm, d, res.system_score, flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
