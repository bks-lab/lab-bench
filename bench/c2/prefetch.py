"""Download every C2 model at its pinned revision into HF_HOME and convert
the two primeline transformers checkpoints to CTranslate2 fp16 (no GPU).

  set HF_HOME=D:\\hf
  python bench/c2/prefetch.py --models D:\\bench-data\\c2\\models

Writes MODELS/<arm>-ct2/ and MODELS/<arm>-ct2/conversion.json (converter
and library versions, source revision).
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--skip-nemo", action="store_true")
    a = ap.parse_args()
    import run_c2
    from huggingface_hub import snapshot_download
    from importlib import metadata
    arms = dict(run_c2.FW)
    if not a.skip_nemo:
        arms.update(run_c2.NEMO)
    for arm, c in arms.items():
        p = snapshot_download(c["model"], revision=c["rev"])
        print(arm, p, flush=True)
        if c.get("local"):
            dst = os.path.join(a.models, c["local"])
            if os.path.exists(os.path.join(dst, "model.bin")):
                continue
            conv = os.path.join(os.path.dirname(sys.executable), "ct2-transformers-converter")
            cmd = [conv, "--model", p, "--output_dir", dst, "--quantization", "float16",
                   "--copy_files", "preprocessor_config.json"]
            subprocess.check_call(cmd)
            # The primeline repos ship no tokenizer.json; faster-whisper would fetch
            # one from the hub at load time, which fails offline. Write it here.
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(p, use_fast=True).backend_tokenizer.save(os.path.join(dst, "tokenizer.json"))
            with open(os.path.join(dst, "conversion.json"), "w", encoding="utf-8") as f:
                json.dump({"source": c["model"], "revision": c["rev"], "cmd": cmd[1:],
                           "ctranslate2": metadata.version("ctranslate2"),
                           "transformers": metadata.version("transformers")}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
