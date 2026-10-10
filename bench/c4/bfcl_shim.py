"""Imports the official BFCL code (gorilla repository at the pinned commit,
plan/c4.md) without its API client dependencies.

bfcl_eval.constants.model_config imports a handler class for every hosted
model it knows (Anthropic, Cohere, Gemini, ...). The AST checker reads only
one flag from that table, `underscore_to_dot`, so this module puts a small
stand-in table into sys.modules before anything else of bfcl_eval loads.
Everything else (dataset loader, function doc processing, tool conversion,
system prompt, decoder, checker) is the unchanged upstream code.

Set BFCL_SRC to the folder `berkeley-function-call-leaderboard` of the
checkout, or pass it to load().
"""
import os
import sys
import types
from dataclasses import dataclass

# Two names for the two modes of plan/c4.md. "lab-fc" follows BFCL's OpenAI
# compatible handlers: dots in function names become "_" in the tool list and
# the checker maps them back (underscore_to_dot). "lab-prompt" keeps the names.
MODEL_FC = "lab-fc"
MODEL_PROMPT = "lab-prompt"


@dataclass
class _Cfg:
    model_name: str
    underscore_to_dot: bool
    is_fc_model: bool


def load(src=None):
    src = src or os.environ.get("BFCL_SRC")
    if not src or not os.path.isdir(os.path.join(src, "bfcl_eval")):
        raise SystemExit("set BFCL_SRC to gorilla/berkeley-function-call-leaderboard at commit 6ea57973c7a6")
    if src not in sys.path:
        sys.path.insert(0, src)
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.ModelConfig = _Cfg
    stub.MODEL_CONFIG_MAPPING = {
        MODEL_FC: _Cfg(MODEL_FC, True, True),
        MODEL_PROMPT: _Cfg(MODEL_PROMPT, False, False),
    }
    sys.modules["bfcl_eval.constants.model_config"] = stub
    import bfcl_eval.utils as bu  # noqa: F401,PLC0415
    from bfcl_eval.constants.enums import Language, ModelStyle, ReturnFormat  # noqa: PLC0415
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: PLC0415
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: PLC0415
    from bfcl_eval.model_handler import utils as mu  # noqa: PLC0415
    return types.SimpleNamespace(utils=bu, mutils=mu, ast_checker=ast_checker, Language=Language,
                                 ModelStyle=ModelStyle, ReturnFormat=ReturnFormat,
                                 GORILLA_TO_OPENAPI=GORILLA_TO_OPENAPI)
