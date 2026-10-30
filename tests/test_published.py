"""The hosted dashboard reads `published/`, so it must be complete, intact and in step with the results it was built from."""
import hashlib
import json

import pandas as pd
import pytest

from kronosquant.config import ROOT
from kronosquant.dashboard_data import experiment_table, load_manifest, scorecard
from kronosquant.experiments import EXPERIMENTS

PUBLISHED = ROOT / "published"
NEEDED = {f for e in EXPERIMENTS for f in e["results"]} | {"equity_curves.csv", "kronos_window_returns.csv", "datasets_inventory.csv", "splits.csv", "paper_account_examples.csv"}


def test_every_file_the_dashboard_needs_is_published_and_matches_its_hash():
    manifest = load_manifest(PUBLISHED)
    assert manifest and set(manifest["files"]) == NEEDED
    for name, info in manifest["files"].items():
        data = (PUBLISHED / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], f"{name} was edited after publishing; rerun scripts/publish_results.py"


def test_published_dashboard_tables_have_every_experiment_completed_and_the_expected_scorecard():
    table = experiment_table(PUBLISHED)
    assert (table["state"] == "Completed").all() and len(table) == len(EXPERIMENTS)
    sc = scorecard(PUBLISHED)
    assert len(sc) == 19 and int(sc["passed"].sum()) == 1  # 1 marginal classical pass, nothing involving Kronos
    assert sc[sc["passed"]]["exp_id"].tolist() == ["exp_stage13_signal_screen"]
    assert table["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()  # dates come from the manifest, not file times


def test_published_snapshot_holds_no_raw_candles():
    """Raw exchange candles must not be redistributed: no published file may carry open/high/low/close columns."""
    for path in PUBLISHED.glob("*.csv"):
        cols = {c.lower() for c in pd.read_csv(path, nrows=1).columns}
        assert not {"open", "high", "low", "close"} <= cols, f"{path.name} looks like raw candles"


def test_published_results_equal_the_local_outputs_when_present():
    """Catches a stale snapshot: if you changed a result in outputs/, republish before committing."""
    outputs = ROOT / "outputs"
    manifest = json.loads((PUBLISHED / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if info["source"].startswith("outputs/") and (outputs / name).exists():
            assert (outputs / name).read_bytes() == (PUBLISHED / name).read_bytes(), f"{name} differs from outputs/; rerun scripts/publish_results.py"
    if not any(outputs.glob("*.csv")):
        pytest.skip("no local outputs/ to compare against")
