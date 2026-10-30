"""Models: Kronos-small and its variants against the random walk. Forecast quality only, no costs."""
import streamlit as st

import charts
import theme
from common import OUT, ROOT, show_chart
from kronosquant.dashboard_data import load_table, pinned_revisions, status_of
from kronosquant.glossary import GLOSSARY


def _result_rows(table, fine_tuned) -> list[list[str]]:
    """One display row per setting, including the fine-tuned model; the IC result uses the same three-state rule as everywhere else."""
    records = [(r.variant, r) for r in table.itertuples()]
    if fine_tuned is not None:
        records.append(("fine-tuned", fine_tuned.iloc[0]))
    return [[name, f"{r.IC:+.3f}", f"[{r.IC_lo99:+.3f}, {r.IC_hi99:+.3f}]", f"{r.dir_acc:.1%}", f"{r.base_rate_up:.1%}", f"{r.path_MAE:.4f}", f"{r.rw_path_MAE:.4f}", status_of(r.IC_lo99, r.IC_hi99)]
            for name, r in records]


def _model_card() -> None:
    rev = pinned_revisions(ROOT)
    theme.card_title("Kronos-small", help="An open-source model that reads candlestick charts like a language model reads words. Revisions are pinned so results stay reproducible.")
    rows = [("Parameters", "24.7 million"), ("Context window", "512 candles"), ("Tokenizer", "NeoQuasar/Kronos-Tokenizer-base"), ("Tokenizer revision", rev["tokenizer"][:12]),
            ("Model revision", rev["model"][:12]), ("Upstream code commit", rev["kronos_commit"][:12]), ("Hardware used", "CPU (Apple M1, 8 GB)"), ("Measured speed", "about 4 s per forecast (5 sampled paths, 400 bars of history)")]
    st.markdown(theme.kv_table(rows), unsafe_allow_html=True)


def _error_chart(grid, tuned) -> None:
    theme.card_title("Path error versus the random walk (12-bar horizon, lower is better)", help=GLOSSARY["Random walk baseline"])
    if grid is None:
        theme.empty_state("No grid results yet.", ".venv/bin/python -I scripts/stage3_grid.py")
        return
    same = grid[~grid["variant"].str.contains("horizon4")]  # the 4-bar variant is on a different scale
    labels, values = list(same["variant"]), [float(v) for v in same["path_MAE"]]
    if tuned is not None:
        labels.append("fine-tuned")
        values.append(float(tuned.iloc[0]["path_MAE"]))
    show_chart(charts.comparison_bars(labels, values, float(same.iloc[0]["rw_path_MAE"]), "random walk", "Mean absolute path error"))


def render() -> None:
    theme.page_header("Models", "Kronos-small and its variants against the random walk. Forecast quality only, no costs.", chip="BTC 1h validation split")
    grid, tuned = load_table(OUT / "grid_BTCUSDT_1h.csv"), load_table(OUT / "stage9_evaluate_val.csv")
    left, right = st.columns([1, 1.5], gap="medium")
    with left, st.container(border=True):
        _model_card()
    with right, st.container(border=True):
        _error_chart(grid, tuned)
    with st.container(border=True):
        theme.card_title("Comparison results", help=GLOSSARY["Information coefficient (IC)"])
        if grid is None:
            theme.empty_state("No comparison results yet.", ".venv/bin/python -I scripts/stage3_grid.py")
            return
        headers = ["Setting", "IC", "99% interval", "Direction", "Base rate up", "Path error", "Random walk", "IC result"]
        st.markdown(theme.rows_table(headers, _result_rows(grid, tuned), badge_cols=(7,), num_cols=(1, 3, 4, 5, 6)), unsafe_allow_html=True)
        theme.note("Direction = how often the sign of the forecast matched the sign of the move, to be read against the share of windows that really went up. "
                   "V4 uses a 4-bar horizon, so its path error is on a different scale from the others. In every setting the path error was higher than the random walk, "
                   "and the 99% interval for the excess error excluded zero (see the result tables in outputs/).")
