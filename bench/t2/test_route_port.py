"""bench/lib/route_local.py against the Node originals (run in CI with pytest).

The raw prompts and the shuffle come from the real modules
(bench/lib/templates.mjs, bench/normalize.mjs). stateText, lettered and the
user turn live inside bench/run-local.mjs, which runs on import; their source
lines are copied below verbatim and the test first checks that run-local.mjs
still contains each of them, so a change there fails here.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "bench", "lib"))
import route_local as rl  # noqa: E402

COPIED = [
    "    if (v && typeof v === 'object') parts.push(`\\`${k}\\`:\\n${Object.entries(v).map(([id, t]) => `${k}.${id}: ${t}`).join('\\n')}`);",
    "    else parts.push(`\\`${k}\\`:\\n${v}`);",
    "  if (q.type === 'noul') return [{ key: 'yes', label: 'Yes' }, { key: 'no', label: 'No' }];",
    "  if (q.type === 'score') return q.criteria.map((c, i) => ({ key: String(i), label: c }));",
    "  return Object.entries(q.criteria).map(([k, v]) => ({ key: k, label: v }));",
    "  const user = `${stateText(state)}\\n\\nQuestion: ${q.instructions}\\n${opts.map((o, i) => `${LETTERS[i]}) ${o.label}`).join('\\n')}`;",
    "  route: 'You classify a short text. Answer each multiple-choice question with the letter of exactly one option.',",
    "    const request = shuffle ? withShuffledOptions(req.request, 1000 + run) : req.request;",
]

NODE = r"""
import fs from 'node:fs';
import { rawPrompt, TEMPLATES } from './bench/lib/templates.mjs';
import { withShuffledOptions, shuffled } from './bench/normalize.mjs';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const SYSTEM = 'You classify a short text. Answer each multiple-choice question with the letter of exactly one option.';
function stateText(state) {
  const parts = [];
  for (const [k, v] of Object.entries(state)) {
    if (v && typeof v === 'object') parts.push(`\`${k}\`:\n${Object.entries(v).map(([id, t]) => `${k}.${id}: ${t}`).join('\n')}`);
    else parts.push(`\`${k}\`:\n${v}`);
  }
  return parts.join('\n\n');
}
function lettered(q) {
  if (q.type === 'noul') return [{ key: 'yes', label: 'Yes' }, { key: 'no', label: 'No' }];
  if (q.type === 'score') return q.criteria.map((c, i) => ({ key: String(i), label: c }));
  return Object.entries(q.criteria).map(([k, v]) => ({ key: k, label: v }));
}
const reqs = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(l => JSON.parse(l));
const out = [];
for (const req of reqs) for (const seed of [null, 1002]) for (const t of TEMPLATES) {
  const request = seed == null ? req.request : withShuffledOptions(req.request, seed);
  for (const [qid, q] of Object.entries(request.questions)) {
    const opts = lettered(q);
    const user = `${stateText(request.state)}\n\nQuestion: ${q.instructions}\n${opts.map((o, i) => `${LETTERS[i]}) ${o.label}`).join('\n')}`;
    out.push({ line: req.line, seed, t, qid, keys: opts.map(o => o.key), prompt: rawPrompt(t, SYSTEM, user, seed ? 'Answer: **' : '') });
  }
}
out.push({ shuffle100: shuffled([...Array(40).keys()], 7) });
console.log(JSON.stringify(out));
"""


def sample_requests(tmp_path):
    lines = open(os.path.join(ROOT, "requests", "route1", "requests.jsonl"), encoding="utf-8").read().splitlines()
    pick = lines[:40] + lines[3000:3040] + lines[-40:]
    p = tmp_path / "reqs.jsonl"
    p.write_text("\n".join(pick) + "\n", encoding="utf-8")
    return p, [json.loads(x) for x in pick]


def test_copied_lines_still_in_run_local():
    src = open(os.path.join(ROOT, "bench", "run-local.mjs"), encoding="utf-8").read()
    for line in COPIED:
        assert line in src, f"run-local.mjs changed, port and test need a look: {line!r}"


def test_prompts_and_shuffle_match_node(tmp_path):
    p, reqs = sample_requests(tmp_path)
    js = tmp_path / "dump.mjs"
    js.write_text(NODE.replace("./bench/", (ROOT + "/bench/").replace("\\", "/")), encoding="utf-8")
    res = subprocess.run(["node", str(js), str(p)], capture_output=True, text=True, check=True)
    node = json.loads(res.stdout)
    want_shuffle = node.pop()["shuffle100"]
    assert rl.shuffled(list(range(40)), 7) == want_shuffle
    py = []
    for req in reqs:
        for seed in (None, 1002):
            for t in rl.TEMPLATES:
                request = req["request"] if seed is None else rl.with_shuffled_options(req["request"], seed)
                for qid, q in request["questions"].items():
                    opts = rl.lettered(q)
                    py.append({"line": req["line"], "seed": seed, "t": t, "qid": qid, "keys": [k for k, _ in opts],
                               "prompt": rl.prompt(t, request["state"], q, opts, "Answer: **" if seed else "")})
    assert len(py) == len(node)
    for a, b in zip(py, node):
        assert a == b, (a["line"], a["t"], a["qid"])
