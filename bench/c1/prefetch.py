"""Download every C1 model at its pinned revision into HF_HOME (no GPU).

  set HF_HOME=D:\\hf
  python bench/c1/prefetch.py [--load]

--load also instantiates each model once on the CPU, which caches the
remote code modules that trust_remote_code models pull from other repos
(needed because the jobs run with HF_HUB_OFFLINE=1).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from run_c1 import EMBEDDERS, RERANKERS, repair_buffers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of arms not to load (still downloaded)")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    for arm, c in {**EMBEDDERS, **RERANKERS}.items():
        p = snapshot_download(c["model"], revision=c["rev"])
        print(arm, p, flush=True)
        if a.load and c.get("remote") and arm not in a.skip.split(","):
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(c["model"], revision=c["rev"], device="cpu", trust_remote_code=True)
            print(arm, "repaired", repair_buffers(m), flush=True)
            e = m.encode(["Wie hoch ist die Zugspitze?", "Die Zugspitze ist 2962 Meter hoch.",
                          "Der Rhein fliesst durch Basel."], normalize_embeddings=True)
            print(arm, "dim", e.shape, "sim", float(e[0] @ e[1]), float(e[0] @ e[2]), "head", e[0][:4].tolist(),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
