"""Small helpers shared by the C1 and C2 runners (plan/c0-common.md)."""
import contextlib, datetime, json, os, platform, sys, time

SEED = 20261009
RESAMPLES = 1000


def now_iso():
    """Local time with offset, comparable with the nvidia-smi sampler CSV."""
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def versions(names):
    from importlib import metadata
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = None
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


class Sections:
    """Records named timed sections (start/end as local ISO, seconds)."""

    def __init__(self):
        self.items = []

    @contextlib.contextmanager
    def __call__(self, name, **extra):
        rec = {"name": name, "start": now_iso(), **extra}
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            rec["end"] = now_iso()
            self.items.append(rec)


def bootstrap_mean(values, clusters=None):
    """95 % percentile interval of the mean, resampling items or clusters."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    v = np.asarray(values, dtype=float)
    if clusters is None:
        idx = rng.integers(0, len(v), size=(RESAMPLES, len(v)))
        stats = v[idx].mean(axis=1)
    else:
        keys = sorted(set(clusters))
        groups = {k: [] for k in keys}
        for x, c in zip(v, clusters):
            groups[c].append(x)
        sums = np.array([sum(groups[k]) for k in keys]); cnts = np.array([len(groups[k]) for k in keys])
        idx = rng.integers(0, len(keys), size=(RESAMPLES, len(keys)))
        stats = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def bootstrap_ratio(num, den, clusters):
    """95 % interval of sum(num)/sum(den) with cluster resampling (corpus WER)."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    keys = sorted(set(clusters))
    pos = {k: i for i, k in enumerate(keys)}
    n = np.zeros(len(keys)); d = np.zeros(len(keys))
    for a, b, c in zip(num, den, clusters):
        n[pos[c]] += a; d[pos[c]] += b
    idx = rng.integers(0, len(keys), size=(RESAMPLES, len(keys)))
    stats = n[idx].sum(axis=1) / d[idx].sum(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def job_env():
    return {k: os.environ.get(k) for k in ("JMB_JOB_ID", "JMB_OUT", "JMB_GPU_CSV", "CUDA_VISIBLE_DEVICES")}


def gate_check(value, ref, tol):
    if ref is None:
        return None
    return {"reference": ref, "tolerance": tol, "value": value, "passed": abs(value - ref) <= tol}


def bootstrap_median(values):
    """95 % percentile interval of the median, resampling items."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(RESAMPLES, len(v)))
    stats = np.median(v[idx], axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
