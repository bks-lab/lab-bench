"""Freezes the C4 inputs (plan/c4.md) into cases/c4/.

    BFCL_SRC=.cache/c4/gorilla/berkeley-function-call-leaderboard \
      .cache/c4/venv/bin/python bench/c4/build_c4.py --jsb .cache/c4/jsb

Part A: the four non-live AST categories of BFCL at gorilla commit
6ea57973c7a6 (simple_python 400, multiple 200, parallel 200,
parallel_multiple 200), loaded with BFCL's own dataset loader (which adds the
"Python 3 syntax" hint to every function description, as the leaderboard
does). For every item the exact request of both modes is written:

- fc: the user turn and the tool list from BFCL's convert_to_tool in the
  OpenAI completions style (dots in names become "_", as the leaderboard's
  "Qwen3-14B (FC)" row ran it), to be sent as Ollama's native `tools` field.
- prompt: BFCL's prompt-mode system prompt (system_prompt_pre_processing_chat_model)
  plus the user turn; the answer is parsed as text.

Part B: 100 schemas from each of six JSONSchemaBench test subsets at
revision 5bd0f4640bad, sampled per subset with a fresh
numpy.random.default_rng(20261009): rows sorted by their index, a
permutation of the row count, the first 100 positions (all rows when fewer).

Writes cases/c4/bfcl.jsonl, cases/c4/bfcl-answers.jsonl, cases/c4/schemas.jsonl
and cases/c4/manifest.json.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bfcl_shim  # noqa: E402

COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple"]
JSB_REPO = "epfl-dlab/JSONSchemaBench"
JSB_REV = "5bd0f4640badc6f3f02df796421d21cb0ca0b141"
SUBSETS = ["Github_easy", "Github_medium", "Github_hard", "Glaiveai2K", "Kubernetes", "JsonSchemaStore"]
SEED = 20261009
PER_SUBSET = 100


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return sha_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsb", required=True, help="folder with <subset>/test.parquet at revision " + JSB_REV)
    ap.add_argument("--out", default=os.path.join(ROOT, "cases", "c4"))
    a = ap.parse_args()
    b = bfcl_shim.load()
    src = os.environ["BFCL_SRC"]
    head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        sys.exit(f"gorilla checkout is at {head}, plan/c4.md pins {COMMIT}")
    os.makedirs(a.out, exist_ok=True)

    items, answers, upstream = [], [], {}
    for cat in CATEGORIES:
        fname = f"BFCL_v4_{cat}.json"
        upstream[fname] = sha_file(os.path.join(src, "bfcl_eval", "data", fname))
        upstream["possible_answer/" + fname] = sha_file(os.path.join(src, "bfcl_eval", "data", "possible_answer", fname))
        entries = b.utils.load_dataset_entry(cat)
        gold = {g["id"]: g["ground_truth"] for g in b.utils.load_file(b.utils.POSSIBLE_ANSWER_PATH / fname)}
        for e in entries:
            assert len(e["question"]) == 1, e["id"]  # single turn
            funcs = e["function"]
            tools = b.mutils.convert_to_tool(copy.deepcopy(funcs), b.GORILLA_TO_OPENAPI, b.ModelStyle.OPENAI_COMPLETIONS)
            prompt_msgs = b.mutils.system_prompt_pre_processing_chat_model(copy.deepcopy(e["question"][0]), funcs, e["id"])
            items.append({"id": e["id"], "category": cat, "function": funcs,
                          "fc": {"messages": e["question"][0], "tools": tools},
                          "prompt": {"messages": prompt_msgs}})
            answers.append({"id": e["id"], "category": cat, "ground_truth": gold[e["id"]]})
    counts = {c: sum(1 for i in items if i["category"] == c) for c in CATEGORIES}
    assert counts == {"simple_python": 400, "multiple": 200, "parallel": 200, "parallel_multiple": 200}, counts

    import pandas as pd  # noqa: PLC0415
    schemas, sample = [], {}
    for s in SUBSETS:
        p = os.path.join(a.jsb, s, "test.parquet")
        upstream[f"{s}/test-00000-of-00001.parquet"] = sha_file(p)
        df = pd.read_parquet(p).reset_index(drop=True)
        n = len(df)
        perm = np.random.default_rng(SEED).permutation(n)
        pick = sorted(int(i) for i in perm[: min(PER_SUBSET, n)])
        sample[s] = {"rows": n, "picked": len(pick)}
        for i in pick:
            sch = df.loc[i, "json_schema"]
            schemas.append({"id": f"{s}:{i}", "subset": s, "row": i, "unique_id": df.loc[i, "unique_id"],
                            "schema": sch, "schema_sha256": sha_text(sch)})

    files = {
        "bfcl.jsonl": write_jsonl(os.path.join(a.out, "bfcl.jsonl"), items),
        "bfcl-answers.jsonl": write_jsonl(os.path.join(a.out, "bfcl-answers.jsonl"), answers),
        "schemas.jsonl": write_jsonl(os.path.join(a.out, "schemas.jsonl"), schemas),
    }
    manifest = {
        "design": "plan/c4.md",
        "bfcl": {"repository": "https://github.com/ShishirPatil/gorilla", "commit": COMMIT,
                 "folder": "berkeley-function-call-leaderboard", "licence": "Apache-2.0",
                 "note": "at this commit the files carry the prefix BFCL_v4_; the four non-live AST categories are the v3 sets",
                 "categories": counts},
        "jsonschemabench": {"repository": JSB_REPO, "revision": JSB_REV, "licence": "MIT", "split": "test",
                            "seed": SEED, "per_subset": PER_SUBSET, "subsets": sample,
                            "sampling": "per subset a fresh numpy default_rng(20261009); permutation of the row count; first 100 positions; rows kept in index order"},
        "upstream_sha256": upstream,
        "files_sha256": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"bfcl": counts, "schemas": len(schemas), "files": files}, indent=1))


if __name__ == "__main__":
    main()
