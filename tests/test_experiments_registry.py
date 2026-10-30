import pytest

from kronosquant.config import ROOT
from kronosquant.dashboard_data import declared_rule, scorecard
from kronosquant.experiments import EXPERIMENTS
from kronosquant.glossary import GLOSSARY, definition

REQUIRED = {"id", "name", "stage", "model", "dataset", "script", "results", "period", "config", "seed", "costs", "baselines", "command"}


def test_registry_entries_are_complete_and_ids_unique():
    assert len({e["id"] for e in EXPERIMENTS}) == len(EXPERIMENTS)
    for e in EXPERIMENTS:
        assert REQUIRED <= set(e), f"{e['id']} is missing {REQUIRED - set(e)}"
        assert e["results"] and all(f.endswith(".csv") for f in e["results"])


def test_every_registered_script_exists_and_the_command_runs_it():
    for e in EXPERIMENTS:
        assert (ROOT / e["script"]).exists(), f"{e['script']} not found"
        assert e["script"] in e["command"], f"{e['id']}: command does not run {e['script']}"


@pytest.mark.parametrize("exp", [e for e in EXPERIMENTS if e["id"] != "exp_stage3_quality"], ids=lambda e: e["id"])
def test_experiments_with_a_pass_rule_declare_it_before_running(exp):
    """The Reports page shows the declared rule from the script header, so each script must contain one."""
    assert declared_rule(ROOT / exp["script"]).startswith("DECLARED BEFORE")


def test_scorecard_experiment_ids_are_all_registered(tmp_path):
    """A scorecard row for an unregistered experiment would never be listed on the Experiments page."""
    import pandas as pd

    pd.DataFrame([dict(kind="paired_diff", name="COMB-BASE", cost_bps=10, mean_bps=1.0, ci_lo99_bps=-1.0, ci_hi99_bps=3.0, primary=True)]).to_csv(tmp_path / "stage13b_incremental.csv", index=False)
    assert set(scorecard(tmp_path)["exp_id"]) <= {e["id"] for e in EXPERIMENTS}


def test_glossary_entries_are_real_sentences_and_lookup_fails_loudly():
    assert all(len(text) > 40 for text in GLOSSARY.values())
    assert definition("Drawdown") == GLOSSARY["Drawdown"]
    with pytest.raises(KeyError):
        definition("Drawdwon")
