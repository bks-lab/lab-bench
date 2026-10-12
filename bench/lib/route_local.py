"""Python port of the route1 path of bench/run-local.mjs, for queue jobs on the
workstation, which has no Node (plan/t2.md, Method). Stdlib only.

Ported one to one, and checked against the Node originals by
bench/t2/test_route_port.py (CI): the raw prompts of bench/lib/templates.mjs,
the state text, the lettered options, the system line of the route family,
and the seeded option shuffle of bench/normalize.mjs (mulberry32).
"""
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_ROUTE = "You classify a short text. Answer each multiple-choice question with the letter of exactly one option."
TEMPLATES = ("chatml", "gemma4", "gemma", "phi4", "mistral", "granite")
M32 = 0xFFFFFFFF


def raw_prompt(template, system, user, prefix=""):
    if template == "gemma4":
        return f"<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{prefix}"
    if template == "gemma":
        return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n{prefix}"
    if template == "phi4":
        return f"<|im_start|>system<|im_sep|>\n{system}<|im_end|>\n<|im_start|>user<|im_sep|>\n{user}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n{prefix}"
    if template == "mistral":
        return f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]{prefix}"
    if template == "granite":
        return (f"<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{user}"
                f"<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{prefix}")
    if template == "chatml":
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
    raise ValueError(f"unknown template {template}")


def state_text(state):
    parts = []
    for k, v in state.items():
        if isinstance(v, dict):
            parts.append(f"`{k}`:\n" + "\n".join(f"{k}.{i}: {t}" for i, t in v.items()))
        else:
            parts.append(f"`{k}`:\n{v}")
    return "\n\n".join(parts)


def options_of(q):
    """normalize.mjs optionsOf: the order the question defined them."""
    if q["type"] == "noul":
        return ["no", "yes"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q.get("criteria") or []))]
    return list((q.get("criteria") or {}).keys())


def lettered(q):
    """[(key, label)] in the order the question is asked (run-local.mjs lettered)."""
    if q["type"] == "noul":
        return [("yes", "Yes"), ("no", "No")]
    if q["type"] == "score":
        return [(str(i), c) for i, c in enumerate(q["criteria"])]
    return list(q["criteria"].items())


def prompt(template, state, q, opts, prefix="", system=SYSTEM_ROUTE):
    user = f"{state_text(state)}\n\nQuestion: {q['instructions']}\n" + "\n".join(f"{LETTERS[i]}) {lab}" for i, (_, lab) in enumerate(opts))
    return raw_prompt(template, system, user, prefix)


def _imul(a, b):
    return (a * b) & M32


def shuffled(arr, seed):
    """normalize.mjs shuffled(): Fisher-Yates driven by mulberry32(seed)."""
    t = seed & M32

    def rnd():
        nonlocal t
        t = (t + 0x6D2B79F5) & M32
        x = t
        x = _imul(x ^ (x >> 15), x | 1)
        x = x ^ ((x + _imul(x ^ (x >> 7), x | 61)) & M32)
        return ((x ^ (x >> 14)) & M32) / 4294967296

    a = list(arr)
    for i in range(len(a) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def with_shuffled_options(request, seed):
    """normalize.mjs withShuffledOptions(): the options of every choice question in shuffled order."""
    questions = {}
    for qid, q in request["questions"].items():
        if q["type"] == "choice":
            keys = shuffled(list(q["criteria"].keys()), seed + len(qid))
            questions[qid] = {**q, "criteria": {k: q["criteria"][k] for k in keys}}
        else:
            questions[qid] = q
    return {**request, "questions": questions}


def letter_probs(top_logprobs, opts):
    """run-local.mjs askQuestion: summed probability per option letter, renormalised; (pByKey, mass)."""
    import math  # noqa: PLC0415
    by = {}
    for t in top_logprobs or []:
        s = (t.get("token") or "").strip()
        if len(s) == 1 and s in LETTERS:
            by[s] = by.get(s, 0.0) + math.exp(t["logprob"])
    raw = [by.get(LETTERS[i], 0.0) for i in range(len(opts))]
    mass = sum(raw)
    return {k: (raw[i] / mass if mass else 0.0) for i, (k, _) in enumerate(opts)}, mass


def r4(x):
    """Math.round(x * 1e4) / 1e4 as in run-local.mjs (round half up, not banker's rounding)."""
    import math  # noqa: PLC0415
    return math.floor(x * 1e4 + 0.5) / 1e4
