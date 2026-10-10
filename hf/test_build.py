"""Tests for hf/build.py, the export of the aggregated results to the
Hugging Face dataset bks-lab/lab-bench-results."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hf"))
import build  # noqa: E402


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("hf")
    build.build(ROOT, d, commit="0" * 40)
    return d


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def test_writes_exactly_the_card_and_two_tables(out):
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["README.md", "data/capabilities.jsonl", "data/toolmap.jsonl"]


def test_toolmap_has_one_row_per_task_and_family(out):
    tm = json.loads((ROOT / "results/toolmap.json").read_text(encoding="utf-8"))
    expected = sum(len(r["cells"]) for r in tm["rows"])
    got = rows(out / "data/toolmap.jsonl")
    assert len(got) == expected
    assert {r["family"] for r in got} <= set(tm["families"])
    for r in got:
        assert r["task_id"] and r["state"] in tm["states"]


def test_capabilities_has_one_row_per_capability(out):
    cap = json.loads((ROOT / "results/capabilities.json").read_text(encoding="utf-8"))
    got = rows(out / "data/capabilities.jsonl")
    assert [r["id"] for r in got] == [r["id"] for r in cap["rows"]]


def test_no_per_question_output_reaches_the_dataset(out):
    """Jev's per-question rows may not be published for training
    (LICENSE-JEV-OUTPUTS). Only aggregates leave the repository."""
    allowed = set(build.TOOLMAP_FIELDS) | {"extra_json"}
    for r in rows(out / "data/toolmap.jsonl"):
        assert set(r) <= allowed
    blob = (out / "data/toolmap.jsonl").read_text(encoding="utf-8")
    assert "results/pv1/jev/" not in blob and "results/route1/jev/" not in blob


def test_card_names_commit_licence_and_jev_exclusion(out):
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "0" * 40 in card
    assert "license: cc-by-4.0" in card
    assert "LICENSE-JEV-OUTPUTS" in card
    assert "—" not in card and "–" not in card


def test_links_point_to_github_at_the_commit(out):
    for r in rows(out / "data/capabilities.jsonl"):
        assert r["report_url"].startswith(
            "https://github.com/bks-lab/lab-bench/blob/" + "0" * 40 + "/")


def test_cli_refuses_a_dirty_or_unknown_commit(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT / "hf/build.py"), "--out", str(tmp_path),
                        "--commit", "not-a-sha"], capture_output=True, text=True)
    assert p.returncode != 0


def test_every_column_has_one_type_for_the_hub_viewer(out):
    """The Hub's dataset viewer builds an Arrow schema per column; a column
    that mixes numbers and strings (p values like 0.11 and "<0.001") fails."""
    for name in ("data/toolmap.jsonl", "data/capabilities.jsonl"):
        kinds = {}
        for r in rows(out / name):
            for k, v in r.items():
                if v is None:
                    continue
                t = "number" if isinstance(v, (int, float)) and not isinstance(v, bool) else type(v).__name__
                kinds.setdefault(k, set()).add(t)
        mixed = {k: v for k, v in kinds.items() if len(v) > 1}
        assert not mixed, f"{name}: {mixed}"
