"""Tests for hf/c3/build.py, the export of the C3 invoices with gold fields
and recorded model outputs to the Hugging Face dataset
bks-lab/xrechnung-invoice-fields."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("c3_build", ROOT / "hf/c3/build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

SHA = "0" * 40


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("c3")
    build.build(ROOT, d, commit=SHA)
    return d


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def gold():
    return rows(ROOT / "reference/c3/gold.jsonl")


def test_every_page_image_is_listed_once(out):
    meta = rows(out / "pages/metadata.jsonl")
    pngs = sorted(p.name for p in (out / "pages").glob("*.png"))
    assert sorted(m["file_name"] for m in meta) == pngs
    assert len(pngs) == len(list((ROOT / "cases/c3/png").glob("*.png")))


def test_invoices_match_gold_and_point_to_their_files(out):
    inv = rows(out / "invoices.jsonl")
    assert [r["id"] for r in inv] == [g["id"] for g in gold()]
    for r in inv:
        assert (out / r["xml_file"]).is_file() and (out / r["pdf_file"]).is_file()
        assert r["pages"] == len(list((out / "pages").glob(f"{r['id']}-*.png")))
        assert set(build.FIELDS) <= set(r["gold"])


def test_predictions_cover_finished_arms_with_hub_model_ids(out):
    pred = rows(out / "predictions.jsonl")
    summary = json.loads((ROOT / "results/c3/summary.json").read_text(encoding="utf-8"))
    finished = {a["arm"] for a in summary["arms"] if a.get("finished")}
    assert {p["arm"] for p in pred} == finished
    n = len(gold())
    for arm in finished:
        assert sum(p["arm"] == arm for p in pred) == n
    for p in pred:
        assert set(p["correct"]) == set(build.FIELDS)
        assert p["n_correct"] == sum(p["correct"].values())


def test_recorded_accuracy_matches_the_published_summary(out):
    pred = rows(out / "predictions.jsonl")
    summary = json.loads((ROOT / "results/c3/summary.json").read_text(encoding="utf-8"))
    for a in summary["arms"]:
        if not a.get("finished"):
            continue
        mine = [p for p in pred if p["arm"] == a["arm"]]
        acc = sum(p["n_correct"] for p in mine) / (len(mine) * len(build.FIELDS))
        assert round(acc, 4) == a["quality"]["value"], a["arm"]


def test_licences_travel_with_the_data(out):
    for f in ["LICENSE-xrechnung-testsuite", "LICENSE-xrechnung-visualization",
              "LICENSE-SourceSerifPro-OFL.txt"]:
        assert (out / f).is_file(), f
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "license: apache-2.0" in card and SHA in card
    assert "—" not in card and "–" not in card
