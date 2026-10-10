"""C1 runner: German retrieval, one arm per call (plan/c1.md).

Embedder arms rank the full corpus by exact inner product over normalised
vectors and keep the top 100 per query. Reranker arms rescore the top 100
of a base embedder run (--base-run, the out dir of that job).

  python bench/c1/run_c1.py --arm bge-m3 --data D:\\bench-data\\c1 --out OUT \\
      --gate 0.5759 --tol 0.015

Writes OUT/result.jsonl (one row per query), OUT/meta.json. Exit 3 when the
sanity gate fails, 1 on any other error.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "cases", "c1"))
import capcommon as cc  # noqa: E402

QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"

EMBEDDERS = {
    "e5-large": dict(model="intfloat/multilingual-e5-large", rev="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
                     q="query: ", d="passage: "),
    "e5-large-instruct": dict(model="intfloat/multilingual-e5-large-instruct",
                              rev="274baa43b0e13e37fafa6428dbc7938e62e5c439",
                              q=f"Instruct: {QWEN_TASK}\nQuery: ", d=""),
    "bge-m3": dict(model="BAAI/bge-m3", rev="5617a9f61b028005a4858fdac845db406aefb181", q="", d=""),
    "gte-multi": dict(model="Alibaba-NLP/gte-multilingual-base", rev="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
                      q="", d="", remote=True),
    "nomic-v2-moe": dict(model="nomic-ai/nomic-embed-text-v2-moe", rev="1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
                         q="search_query: ", d="search_document: ", remote=True),
    "qwen3-emb-0.6b": dict(model="Qwen/Qwen3-Embedding-0.6B", rev="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                           q=f"Instruct: {QWEN_TASK}\nQuery:", d="", left_pad=True),
    "qwen3-emb-8b": dict(model="Qwen/Qwen3-Embedding-8B", rev="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                         q=f"Instruct: {QWEN_TASK}\nQuery:", d="", left_pad=True),
}
RERANKERS = {
    "+bge-rerank": dict(model="BAAI/bge-reranker-v2-m3", rev="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "+qwen3-rerank": dict(model="Qwen/Qwen3-Reranker-0.6B", rev="e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}
BM25 = dict(k1=0.9, b=0.4, stemmer="german (PyStemmer, Snowball)", stopwords=None)
MAX_LEN = 512
TOPK = 100


def repair_buffers(model):
    """Rebuild non-persistent buffers of remote-code models.

    transformers 5 builds models on the meta device and then loads only the
    weights in the checkpoint, so buffers that the remote code computes in
    __init__ with persistent=False (position_ids, rotary inv_freq and cos/sin
    caches in gte-multilingual's modeling.py) come back uninitialised. Returns
    the names of the rebuilt buffers, recorded in meta.json.
    """
    import torch
    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        return []
    fixed = []
    for name, mod in model.named_modules():
        bufs = dict(mod.named_buffers(recurse=False))
        if "position_ids" in bufs and bufs["position_ids"].dim() == 1:
            n = bufs["position_ids"].shape[0]
            mod.position_ids = torch.arange(n, device=bufs["position_ids"].device)
            fixed.append(name + ".position_ids")
        if "inv_freq" in bufs and hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "base") and hasattr(mod, "dim"):
            dev = bufs["inv_freq"].device
            mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=dev).float() / mod.dim))
            seq = mod.max_position_embeddings
            if getattr(mod, "scaling_factor", None):
                seq = seq * mod.scaling_factor
            mod._set_cos_sin_cache(seq, dev, torch.get_default_dtype())
            fixed.append(name + ".rotary")
    return fixed


def local_path(cfg):
    """Snapshot dir of the pinned revision. sentence-transformers 6 does not pass
    `revision` on to AutoConfig, which then asks the hub for `main` and fails
    offline; loading from the pinned snapshot dir avoids that. Remote-code
    models keep repo id + revision (their auto_map points at other repos)."""
    if cfg.get("remote"):
        return cfg["model"]
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg["model"], revision=cfg["rev"])


def load_sets(data, sets, limit_q, limit_c):
    import pandas as pd
    out = {}
    for s in sets:
        c = pd.read_parquet(os.path.join(data, s, "corpus.parquet"))
        q = pd.read_parquet(os.path.join(data, s, "queries.parquet"))
        r = pd.read_parquet(os.path.join(data, s, "qrels.parquet"))
        qrels = {}
        for qid, did, sc in zip(r["query-id"], r["corpus-id"], r["score"]):
            qrels.setdefault(str(qid), {})[str(did)] = int(sc)
        queries = [(str(i), t) for i, t in zip(q["id"], q["text"]) if str(i) in qrels]
        # MTEB joins title and text for the corpus.
        corpus = [(str(i), (f"{ti} {tx}" if ti else tx).strip()) for i, tx, ti in zip(c["id"], c["text"], c["title"])]
        if limit_q:
            queries = queries[:limit_q]
            keep = {d for qid, _ in queries for d in qrels[qid]}
            corpus = [x for x in corpus if x[0] in keep] + [x for x in corpus if x[0] not in keep][:limit_c]
        out[s] = dict(queries=queries, corpus=corpus, qrels=qrels)
    return out


def metrics(qrels, run):
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    res = ev.evaluate(run)
    out = {}
    for qid, ranked in run.items():
        order = sorted(ranked.items(), key=lambda x: -x[1])[:10]
        rel = qrels.get(qid, {})
        mrr = 0.0
        for k, (d, _) in enumerate(order, 1):
            if rel.get(d, 0) > 0:
                mrr = 1.0 / k
                break
        r = res.get(qid, {})  # pytrec_eval drops queries with an empty ranking
        out[qid] = dict(ndcg10=r.get("ndcg_cut_10", 0.0), recall100=r.get("recall_100", 0.0), mrr10=mrr)
    return out


def summarise(per_q):
    s = {}
    for m in ("ndcg10", "recall100", "mrr10"):
        vals = [v[m] for v in per_q.values()]
        s[m] = sum(vals) / len(vals)
        s[m + "_ci95"] = cc.bootstrap_mean(vals)
    s["queries"] = len(per_q)
    return s


def run_embedder(arm, cfg, sets, device, batch, sec):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    dtype = torch.float16 if device == "cuda" else torch.float32
    tk = {"padding_side": "left"} if cfg.get("left_pad") else {}
    with sec("model_load"):
        src = local_path(cfg)
        model = SentenceTransformer(src, revision=cfg["rev"] if cfg.get("remote") else None, device=device,
                                    trust_remote_code=bool(cfg.get("remote")),
                                    model_kwargs={"torch_dtype": dtype}, tokenizer_kwargs=tk)
        model.max_seq_length = MAX_LEN
        repaired = repair_buffers(model) if cfg.get("remote") else []
        if repaired:
            model.to(device)
    info = {"repaired_buffers": repaired, "model": cfg["model"], "revision": cfg["rev"], "query_prefix": cfg["q"], "doc_prefix": cfg["d"],
            "max_seq_length": MAX_LEN, "dtype": str(dtype), "index_batch": batch, "query_batch": 1}
    rows, per_set = [], {}
    for s, D in sets.items():
        docs = [cfg["d"] + t for _, t in D["corpus"]]
        ids = [i for i, _ in D["corpus"]]
        lens = [len(x) for x in model.tokenizer(docs, add_special_tokens=True, truncation=False)["input_ids"]]
        trunc = sum(1 for n in lens if n > MAX_LEN)
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with sec("index", set=s, passages=len(docs)) as rec:
            emb = model.encode(docs, batch_size=batch, normalize_embeddings=True, convert_to_tensor=True,
                               show_progress_bar=False)
            if device == "cuda":
                torch.cuda.synchronize()
        idx_s = rec["seconds"]
        emb = emb.to(torch.float32)
        qs = D["queries"]
        warm = [cfg["q"] + t for _, t in qs[:3]]
        for w in warm:
            model.encode([w], batch_size=1, normalize_embeddings=True, convert_to_tensor=True)
        run, lat = {}, {}
        with sec("queries_batch1", set=s, queries=len(qs)):
            for qid, text in qs:
                t0 = time.perf_counter()
                qe = model.encode([cfg["q"] + text], batch_size=1, normalize_embeddings=True,
                                  convert_to_tensor=True, show_progress_bar=False).to(torch.float32)
                sc = qe @ emb.T
                top = torch.topk(sc[0], k=min(TOPK, emb.shape[0]))
                vals, inds = top.values.tolist(), top.indices.tolist()
                if device == "cuda":
                    torch.cuda.synchronize()
                lat[qid] = (time.perf_counter() - t0) * 1000
                run[qid] = {ids[i]: float(v) for i, v in zip(inds, vals)}
        per_q = metrics({q: D["qrels"][q] for q, _ in qs}, run)
        summ = summarise(per_q)
        lats = sorted(lat.values())
        summ.update(passages=len(docs), truncated_passages=trunc, index_seconds=idx_s,
                    passages_per_s=len(docs) / idx_s if idx_s else None,
                    query_ms_median=lats[len(lats) // 2],
                    query_ms_median_ci95=cc.bootstrap_median(list(lat.values())),
                    peak_vram_gb_torch=(torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None))
        per_set[s] = summ
        for qid, _ in qs:
            top = sorted(run[qid].items(), key=lambda x: -x[1])
            rows.append(dict(set=s, qid=qid, **per_q[qid], latency_ms=round(lat[qid], 3),
                             top100=[[d, round(v, 6)] for d, v in top]))
        del emb
    return info, per_set, rows


class QwenReranker:
    """Qwen3-Reranker per its model card: yes/no logits after a fixed prompt."""

    def __init__(self, cfg, device, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        src = local_path(cfg)
        self.tok = AutoTokenizer.from_pretrained(src, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
                       "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                       "<|im_end|>\n<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.pre_ids = self.tok.encode(self.prefix, add_special_tokens=False)
        self.suf_ids = self.tok.encode(self.suffix, add_special_tokens=False)
        self.max_length = 8192

    def last_logits(self, enc):
        """Logits of the last position only (left padding, so it is every pair's last token).

        The plain forward call runs the LM head over every position (pairs x tokens x
        151k vocabulary) and the first run of 2026-10-10 spent its whole timeout doing
        that. Same scores: only the last position was ever read.
        """
        try:
            return self.model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:  # transformers without logits_to_keep
            h = self.model.model(**enc).last_hidden_state[:, -1, :]
            return self.model.lm_head(h)

    def score(self, query, docs, batch):
        import torch
        out = []
        for i in range(0, len(docs), batch):
            pairs = [f"<Instruct>: {QWEN_TASK}\n<Query>: {query}\n<Document>: {d}" for d in docs[i:i + batch]]
            enc = self.tok(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                           max_length=self.max_length - len(self.pre_ids) - len(self.suf_ids))
            enc["input_ids"] = [self.pre_ids + x + self.suf_ids for x in enc["input_ids"]]
            enc = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.last_logits(enc)
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()
        return out


def run_reranker(arm, cfg, sets, device, base_run, batch, sec):
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = {}
    with open(os.path.join(base_run, "result.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            base[(r["set"], r["qid"])] = r["top100"]
    base_meta = json.load(open(os.path.join(base_run, "meta.json"), encoding="utf-8"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with sec("model_load"):
        if arm == "+bge-rerank":
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(local_path(cfg), device=device, max_length=MAX_LEN,
                              model_kwargs={"torch_dtype": dtype})
            scorer = lambda q, docs: [float(x) for x in ce.predict([(q, d) for d in docs], batch_size=batch,
                                                                    show_progress_bar=False)]
        else:
            qr = QwenReranker(cfg, device, dtype)
            scorer = lambda q, docs: qr.score(q, docs, batch)
    info = {"model": cfg["model"], "revision": cfg["rev"], "base_arm": base_meta.get("arm"),
            "base_run": base_run, "rerank_depth": TOPK, "pair_batch": batch, "dtype": str(dtype)}
    rows, per_set = [], {}
    for s, D in sets.items():
        text = dict(D["corpus"])
        qs = [(q, t) for q, t in D["queries"] if (s, q) in base]
        for q, t in qs[:3]:
            scorer(t, [text[d] for d, _ in base[(s, q)] if d in text][:8])
        run, lat = {}, {}
        with sec("rerank", set=s, queries=len(qs)):
            for k, (qid, qtext) in enumerate(qs, 1):
                if k % 50 == 0 or k == len(qs):
                    print(f"rerank {s} {k}/{len(qs)}", flush=True)
                cand = [d for d, _ in base[(s, qid)] if d in text]
                t0 = time.perf_counter()
                sc = scorer(qtext, [text[d] for d in cand])
                if device == "cuda":
                    torch.cuda.synchronize()
                lat[qid] = (time.perf_counter() - t0) * 1000
                run[qid] = {d: float(v) for d, v in zip(cand, sc)}
        per_q = metrics({q: D["qrels"][q] for q, _ in qs}, run)
        summ = summarise(per_q)
        lats = sorted(lat.values())
        summ.update(rerank_ms_per_query_median=lats[len(lats) // 2],
                    peak_vram_gb_torch=(torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None))
        per_set[s] = summ
        for qid, _ in qs:
            top = sorted(run[qid].items(), key=lambda x: -x[1])
            rows.append(dict(set=s, qid=qid, **per_q[qid], latency_ms=round(lat[qid], 3),
                             top100=[[d, round(v, 6)] for d, v in top]))
    return info, per_set, rows


def run_bm25(sets, sec):
    import bm25s, Stemmer
    stem = Stemmer.Stemmer("german")
    info = dict(BM25, library="bm25s")
    rows, per_set = [], {}
    for s, D in sets.items():
        ids = [i for i, _ in D["corpus"]]
        with sec("index", set=s, passages=len(ids)) as rec:
            tok = bm25s.tokenize([t for _, t in D["corpus"]], stopwords=None, stemmer=stem, show_progress=False)
            r = bm25s.BM25(k1=BM25["k1"], b=BM25["b"])
            r.index(tok, show_progress=False)
        run, lat = {}, {}
        with sec("queries_batch1", set=s, queries=len(D["queries"])):
            for qid, text in D["queries"]:
                t0 = time.perf_counter()
                qt = bm25s.tokenize([text], stopwords=None, stemmer=stem, show_progress=False)
                docs, scores = r.retrieve(qt, k=min(TOPK, len(ids)), show_progress=False)
                lat[qid] = (time.perf_counter() - t0) * 1000
                run[qid] = {ids[int(d)]: float(v) for d, v in zip(docs[0], scores[0])}
        per_q = metrics({q: D["qrels"][q] for q, _ in D["queries"]}, run)
        summ = summarise(per_q)
        lats = sorted(lat.values())
        summ.update(passages=len(ids), index_seconds=rec["seconds"], query_ms_median=lats[len(lats) // 2])
        per_set[s] = summ
        for qid, _ in D["queries"]:
            top = sorted(run[qid].items(), key=lambda x: -x[1])
            rows.append(dict(set=s, qid=qid, **per_q[qid], latency_ms=round(lat[qid], 3),
                             top100=[[d, round(v, 6)] for d, v in top]))
    return info, per_set, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(EMBEDDERS) + list(RERANKERS) + ["bm25"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="miracl,germandpr")
    ap.add_argument("--gate", type=float)
    ap.add_argument("--gate-set", default="miracl")
    ap.add_argument("--tol", type=float, default=0.015)
    ap.add_argument("--base-run", help="out dir of the embedder job whose top 100 is reranked")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-queries", type=int, default=0, help="probe only")
    ap.add_argument("--limit-corpus", type=int, default=2000, help="probe only: extra non-relevant passages")
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import fetch  # cases/c1/fetch.py
    if not a.skip_verify:
        m = json.load(open(fetch.MANIFEST, encoding="utf-8"))
        if not fetch.verify(a.data, m):
            print("data verification failed", file=sys.stderr)
            return 1
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sec = cc.Sections()
    meta = {"test": "C1", "arm": a.arm, "started": cc.now_iso(), "device": device, "job": cc.job_env(),
            "argv": sys.argv, "probe": bool(a.limit_queries),
            "versions": cc.versions(["torch", "transformers", "sentence-transformers", "pytrec-eval-terrier",
                                     "bm25s", "PyStemmer", "numpy", "pandas"]),
            "data_manifest": "cases/c1/manifest.json"}
    sets = load_sets(a.data, a.sets.split(","), a.limit_queries, a.limit_corpus)
    if a.arm == "bm25":
        info, per_set, rows = run_bm25(sets, sec)
    elif a.arm in EMBEDDERS:
        info, per_set, rows = run_embedder(a.arm, EMBEDDERS[a.arm], sets, device, a.batch, sec)
    else:
        if not a.base_run:
            print("reranker arms need --base-run", file=sys.stderr)
            return 1
        info, per_set, rows = run_reranker(a.arm, RERANKERS[a.arm], sets, device, a.base_run,
                                           min(a.batch, 32), sec)
    with open(os.path.join(a.out, "result.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta.update(arm_config=info, summary=per_set, sections=sec.items, finished=cc.now_iso())
    gate = None
    if a.gate is not None and a.gate_set in per_set:
        gate = cc.gate_check(per_set[a.gate_set]["ndcg10"], a.gate, a.tol)
        gate.update(set=a.gate_set, metric="ndcg10")
    meta["gate"] = gate
    cc.write_json(os.path.join(a.out, "meta.json"), meta)
    print(json.dumps({s: {k: round(v, 4) for k, v in d.items() if isinstance(v, float)} for s, d in per_set.items()}))
    if gate is not None and not gate["passed"] and not a.limit_queries:
        print(f"SANITY GATE FAILED: {gate}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
