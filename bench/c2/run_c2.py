"""C2 runner: German speech to text on FLEURS de_de test, one arm per call
(plan/c2.md).

Pass 1, batch 1: one utterance per call after three warm-up items, timed
per utterance; these transcripts are scored. Pass 2, batch 16: throughput
only (audio hours per wall hour), transcripts kept but not scored.

  python bench/c2/run_c2.py --arm canary-1b-v2 --data D:\\bench-data\\c2 --out OUT --gate 4.40 --tol 1.5

Writes OUT/result.jsonl (one row per utterance), OUT/meta.json. Exit 3 when
the sanity gate fails, 1 on any other error.
"""
import argparse, csv, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "bench", "c1"))
sys.path.insert(0, os.path.join(REPO, "bench", "lib"))
import capcommon as cc  # noqa: E402
from norm_de import norm_de  # noqa: E402

FW = {
    "whisper-large-v3": dict(model="Systran/faster-whisper-large-v3", rev="edaa852ec7e145841d8ffdb056a99866b5f0a478"),
    "whisper-large-v3-turbo": dict(model="deepdml/faster-whisper-large-v3-turbo-ct2",
                                   rev="4df90f75321148c3a29a9e2351b7ddf8f5b115a8"),
    # Converted to CTranslate2 beforehand by bench/c2/convert_ct2.py (fp16).
    "whisper-turbo-de": dict(model="primeline/whisper-large-v3-turbo-german",
                             rev="9e7012dad9247f07553aac8a99c4d50ad7c3d36d", local="whisper-turbo-de-ct2"),
    "distil-whisper-de": dict(model="primeline/distil-whisper-large-v3-german",
                              rev="95779fc83382d3749b20c5992299e75e0c269346", local="distil-whisper-de-ct2"),
}
NEMO = {
    "canary-1b-v2": dict(model="nvidia/canary-1b-v2", rev="d455706339a6b32e1aa40f82c713a482a0c938e2"),
    "parakeet-v3": dict(model="nvidia/parakeet-tdt-0.6b-v3", rev="541d1f99c6b0c3cd0b11a95167540bb8edefd82b"),
}
BEAM = 5
BATCH = 16
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def load_utts(data):
    m = json.load(open(os.path.join(REPO, "cases", "c2", "manifest.json"), encoding="utf-8"))
    cols = ["id", "file", "raw_transcription", "transcription", "chars", "num_samples", "gender"]
    with open(os.path.join(data, "test.tsv"), encoding="utf-8", newline="") as f:
        rows = [dict(zip(cols, r)) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    for r in rows:
        r["path"] = os.path.join(data, "audio", "test", r["file"])
        r["dur"] = int(r["num_samples"]) / 16000
    return m, rows


def read16k(path):
    """FLEURS wavs are 16 kHz; read them with soundfile instead of PyAV, whose
    current release no longer matches faster-whisper's decode_audio."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(f"{path}: {sr} Hz, expected 16000")
    return np.ascontiguousarray(x.mean(axis=1))


def spell_numbers(t):
    from num2words import num2words

    def rep(mo):
        s = mo.group(0).replace(".", "").replace(",", ".")
        try:
            return " " + num2words(float(s) if "." in s else int(s), lang="de") + " "
        except Exception:
            return mo.group(0)
    return _NUM.sub(rep, t)


def score(refs, hyps, clusters):
    import jiwer
    rn = [norm_de(r) for r in refs]
    hn = [norm_de(h) for h in hyps]
    per = []
    for r, h in zip(rn, hn):
        o = jiwer.process_words(r, h) if r else None
        per.append((o.substitutions + o.deletions + o.insertions, len(r.split())) if o else (0, 0))
    wer = jiwer.wer(rn, hn)
    cer = jiwer.cer(rn, hn)
    rs = [norm_de(spell_numbers(r)) for r in refs]
    hs = [norm_de(spell_numbers(h)) for h in hyps]
    digits = sum(1 for r, h in zip(refs, hyps) if re.search(r"\d", r + h))
    return dict(wer=wer, wer_ci95=cc.bootstrap_ratio([e for e, _ in per], [n for _, n in per], clusters),
                cer=cer, wer_num2words=jiwer.wer(rs, hs), utterances_with_digits=digits,
                share_with_digits=digits / len(refs)), per, rn, hn


class FasterWhisperArm:
    def __init__(self, cfg, device, models_dir):
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        if cfg.get("local"):
            path = os.path.join(models_dir, cfg["local"])
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(cfg["model"], revision=cfg["rev"])
        self.path = path
        self.compute = "float16" if device == "cuda" else "int8"
        self.m = WhisperModel(path, device=device, compute_type=self.compute)
        self.tok = Tokenizer(self.m.hf_tokenizer, self.m.model.is_multilingual, task="transcribe", language="de")

    def one(self, path):
        segs, _ = self.m.transcribe(read16k(path), language="de", task="transcribe", beam_size=BEAM, vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()

    def batch(self, paths):
        """Batch of whole utterances through the CTranslate2 model, each padded to
        one 30 s window. Utterances longer than 30 s (18 of 862) go through the
        normal long-form path inside the same timed pass."""
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        audio = [read16k(p) for p in paths]
        short = [i for i, x in enumerate(audio) if len(x) <= 30 * 16000]
        out = [None] * len(paths)
        if short:
            feats = [pad_or_trim(self.m.feature_extractor(audio[i])) for i in short]
            enc = self.m.encode(np.stack(feats))
            prompt = list(self.tok.sot_sequence) + [self.tok.no_timestamps]
            res = self.m.model.generate(enc, [prompt] * len(short), beam_size=BEAM, max_length=448)
            for i, r in zip(short, res):
                out[i] = self.tok.decode([t for t in r.sequences_ids[0] if t < self.tok.eot]).strip()
        for i in range(len(paths)):
            if out[i] is None:
                out[i] = self.one(paths[i])
        return out

    def info(self):
        return {"runtime": "faster-whisper", "path": self.path, "compute_type": self.compute, "beam": BEAM,
                "language": "de", "vad": False}


class NemoArm:
    def __init__(self, cfg, arm, device):
        from huggingface_hub import snapshot_download
        from nemo.collections.asr.models import ASRModel
        d = snapshot_download(cfg["model"], revision=cfg["rev"])
        nemo_file = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".nemo")][0]
        self.m = ASRModel.restore_from(nemo_file, map_location=device).eval()
        self.arm, self.path = arm, nemo_file
        # canary-1b-v2 ignores a pnc="no" request (the CPU probe still returned
        # punctuation and case); the normaliser removes both, as the card's
        # number does not count them.
        self.kw = dict(source_lang="de", target_lang="de") if arm == "canary-1b-v2" else {}

    def _run(self, paths, bs):
        out = self.m.transcribe(paths, batch_size=bs, verbose=False, **self.kw)
        if isinstance(out, tuple):
            out = out[0]
        return [o.text if hasattr(o, "text") else str(o) for o in out]

    def one(self, path):
        return self._run([path], 1)[0].strip()

    def batch(self, paths):
        return [x.strip() for x in self._run(paths, BATCH)]

    def info(self):
        return {"runtime": "nemo", "path": self.path, "decode_kwargs": self.kw, "dtype": "float32"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(FW) + list(NEMO))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="dir of converted CTranslate2 models (default DATA/models)")
    ap.add_argument("--gate", type=float)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="probe only")
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.skip_verify:
        import subprocess
        rc = subprocess.call([sys.executable, os.path.join(REPO, "cases", "c2", "fetch.py"), "--out", a.data,
                              "--verify-only"])
        if rc:
            print("data verification failed", file=sys.stderr)
            return 1
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest, utts = load_utts(a.data)
    if a.limit:
        utts = utts[:a.limit]
    sec = cc.Sections()
    pk = ["torch", "torchaudio", "faster-whisper", "ctranslate2", "jiwer", "num2words", "nemo_toolkit", "numpy"]
    meta = {"test": "C2", "arm": a.arm, "started": cc.now_iso(), "device": device, "job": cc.job_env(),
            "argv": sys.argv, "probe": bool(a.limit), "versions": cc.versions(pk),
            "data": {k: manifest[k] for k in ("repo", "config", "split", "revision")}}
    cfg = FW.get(a.arm) or NEMO[a.arm]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with sec("model_load"):
        arm = (FasterWhisperArm(cfg, device, a.models or os.path.join(a.data, "models")) if a.arm in FW
               else NemoArm(cfg, a.arm, device))
    for u in utts[:3]:
        arm.one(u["path"])
    hyps, times = [], []
    with sec("batch1", utterances=len(utts), audio_s=sum(u["dur"] for u in utts)):
        for u in utts:
            t0 = time.perf_counter()
            hyps.append(arm.one(u["path"]))
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    batch_hyps = None
    if not a.skip_batch:
        batch_hyps = []
        with sec("batch16", utterances=len(utts), audio_s=sum(u["dur"] for u in utts)) as b16:
            for i in range(0, len(utts), BATCH):
                batch_hyps += arm.batch([u["path"] for u in utts[i:i + BATCH]])
            if device == "cuda":
                torch.cuda.synchronize()
    summ, per, rn, hn = score([u["transcription"] for u in utts], hyps, [u["id"] for u in utts])
    audio = sum(u["dur"] for u in utts)
    rtf = sorted(t / u["dur"] for t, u in zip(times, utts))
    summ.update(utterances=len(utts), sentences=len({u["id"] for u in utts}), audio_hours=audio / 3600,
                rtf_median=rtf[len(rtf) // 2],
                rtf_median_ci95=cc.bootstrap_median([t / u["dur"] for t, u in zip(times, utts)]),
                rtf_pooled=sum(times) / audio,
                peak_vram_gb_torch=(torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
                utterances_over_30s=sum(1 for u in utts if u["dur"] > 30))
    if batch_hyps is not None:
        summ["batch16_audio_h_per_wall_h"] = audio / b16["seconds"]
        bs, _, _, _ = score([u["transcription"] for u in utts], batch_hyps, [u["id"] for u in utts])
        summ["batch16_wer_unscored_check"] = bs["wer"]
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            f.write(json.dumps(dict(id=u["id"], file=u["file"], duration_s=u["dur"], seconds=round(times[i], 4),
                                    ref=u["transcription"], hyp=hyps[i], ref_norm=rn[i], hyp_norm=hn[i],
                                    errors=per[i][0], ref_words=per[i][1],
                                    hyp_batch16=batch_hyps[i] if batch_hyps else None), ensure_ascii=False) + "\n")
    gate = cc.gate_check(summ["wer"] * 100, a.gate, a.tol)
    if gate:
        gate.update(metric="wer_percent")
    meta.update(arm_config={"model": cfg["model"], "revision": cfg["rev"], **arm.info(), "batch_pass": BATCH},
                summary=summ, gate=gate, sections=sec.items, finished=cc.now_iso())
    cc.write_json(os.path.join(a.out, "meta.json"), meta)
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, list)}))
    if gate and not gate["passed"] and not a.limit:
        print(f"SANITY GATE FAILED: {gate}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
