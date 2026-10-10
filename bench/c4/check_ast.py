"""Scores the raw C4 rows item by item (plan/c4.md, results/c4/method.md).

    BFCL_SRC=.cache/c4/gorilla/berkeley-function-call-leaderboard \
      .cache/c4/venv/bin/python bench/c4/check_ast.py [results/c4/<arm>/<date>-run1.jsonl ...]

Part A (fc, prompt): BFCL's own `_evaluate_single_ast_entry` with the
decoder of BFCL's OpenAI completions handler (FC: tool calls as
{name: JSON arguments}; prompt: `default_decode_ast_prompting` on the text)
and the AST checker. Nothing of the check is reimplemented here; this file
only feeds the recorded answers in.

Part B (free, constrained): parse as results/c4/method.md says, then
validate with jsonschema (the draft the schema declares, Draft 2020-12 when
it names none, format checking off).

Writes <run>-verdicts.jsonl next to each run file: one row per item and
mode with `valid` (bool) and the reason when not valid. bench/score-cap.py
reads only these verdicts and the raw rows; bench/check-cap.py recomputes
Part B on its own and Part A's totals from the verdicts.
"""
import copy
import glob
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bfcl_shim  # noqa: E402

REFUSED = re.compile(r"(format|schema|grammar)", re.I)


def parse_free(text):
    """First JSON value of the answer: whole text, else a Markdown fence, else first { to last }."""
    if text is None:
        return None, "no content"
    t = text.strip()
    for cand in (t,):
        try:
            return json.loads(cand), None
        except Exception:  # noqa: BLE001
            pass
    m = re.search(r"```(?:json|JSON)?\s*\n(.*?)\n?```", t, re.S)
    if m:
        try:
            return json.loads(m.group(1)), None
        except Exception:  # noqa: BLE001
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), None
        except Exception:  # noqa: BLE001
            pass
    return None, "no JSON value found"


def validator_for(schema):
    import jsonschema  # noqa: PLC0415
    from jsonschema import validators  # noqa: PLC0415
    cls = validators.validator_for(schema, default=jsonschema.Draft202012Validator)
    try:
        cls.check_schema(schema)
    except Exception as e:  # noqa: BLE001
        return None, f"schema invalid for {cls.__name__}: {str(e)[:200]}"
    return cls(schema), None


def check_b(row, schema_text):
    schema = json.loads(schema_text)
    v, bad = validator_for(schema)
    if bad:
        return {"valid": None, "reason": bad, "schema_invalid": True}
    if row["mode"] == "constrained" and row.get("error") and REFUSED.search(row["error"]):
        return {"valid": None, "reason": "not constrainable: " + row["error"][:200], "not_constrainable": True}
    if row.get("error"):
        return {"valid": False, "reason": "call failed: " + row["error"][:200]}
    val, why = parse_free(row.get("content"))
    if why:
        return {"valid": False, "reason": why, "parsed": False}
    errs = list(v.iter_errors(val))
    return {"valid": not errs, "parsed": True, "reason": None if not errs else str(errs[0].message)[:200]}


def main():
    b = bfcl_shim.load()
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_ast_entry  # noqa: PLC0415
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: PLC0415
    handlers = {m: types.SimpleNamespace(is_fc_model=(m == "fc")) for m in ("fc", "prompt")}
    for h in handlers.values():
        h.decode_ast = types.MethodType(OpenAICompletionsHandler.decode_ast, h)
    cdir = os.path.join(ROOT, "cases", "c4")
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(cdir, "bfcl.jsonl"), encoding="utf-8")}
    gold = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(os.path.join(cdir, "bfcl-answers.jsonl"), encoding="utf-8")}
    schemas = {json.loads(l)["id"]: json.loads(l)["schema"] for l in open(os.path.join(cdir, "schemas.jsonl"), encoding="utf-8")}
    files = sys.argv[1:] or sorted(f for f in glob.glob(os.path.join(ROOT, "results", "c4", "*", "*-run1.jsonl")))
    for f in files:
        out = []
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            v = {"id": r["id"], "mode": r["mode"]}
            if r["mode"] in ("fc", "prompt"):
                it = items[r["id"]]
                if r["mode"] == "fc":
                    result = [{tc["name"]: json.dumps(tc["arguments"]) if not isinstance(tc["arguments"], str) else tc["arguments"]}
                              for tc in r["tool_calls"]] if r["tool_calls"] else (r.get("content") or "")
                    model = bfcl_shim.MODEL_FC
                else:
                    result = r.get("content") or ""
                    model = bfcl_shim.MODEL_PROMPT
                if r.get("error"):
                    v.update(valid=False, error_type="call_failed", reason=r["error"][:200])
                else:
                    res = _evaluate_single_ast_entry(handlers[r["mode"]], r["id"], copy.deepcopy(result), gold[r["id"]],
                                                     copy.deepcopy(it), model, it["category"], b.Language.PYTHON,
                                                     b.ReturnFormat.PYTHON, False)
                    v["valid"] = bool(res["valid"])
                    if not res["valid"]:
                        v["error_type"] = res.get("error_type")
                        v["reason"] = str(res.get("error"))[:300]
                        v["decoded"] = res.get("error_type", "").startswith("ast_decoder") is False
            else:
                v.update(check_b(r, schemas[r["id"]]))
            out.append(v)
        dest = f[: -len(".jsonl")] + "-verdicts.jsonl"
        with open(dest, "w", encoding="utf-8", newline="\n") as fo:
            for v in out:
                fo.write(json.dumps(v, ensure_ascii=False) + "\n")
        by = {}
        for v in out:
            s = by.setdefault(v["mode"], [0, 0, 0])
            s[0] += v["valid"] is True
            s[1] += v["valid"] is not None
            s[2] += 1
        print(os.path.relpath(f, ROOT), {m: f"{a}/{n} valid ({t - n} excluded)" for m, (a, n, t) in by.items()})


if __name__ == "__main__":
    main()
