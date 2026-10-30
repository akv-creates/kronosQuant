"""Helpers shared by every dashboard page: paths, cached data loaders, number formatting and friendly error handling."""
import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import theme  # noqa: E402
from kronosquant.dashboard_data import STRATEGIES, load_table, validation_candles  # noqa: E402
from kronosquant.data import DataError, load_candles  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402

OUT, DATA = ROOT / "outputs", ROOT / "data"
FETCH_HINT = ".venv/bin/python -I scripts/fetch_binance.py --symbol BTCUSDT --interval 1h --years 3 --end 2026-10-28"
CHART_CONFIG = {"displayModeBar": False}


@st.cache_data(show_spinner="Loading validation data…")
def val_data(symbol: str) -> pd.DataFrame:
    """Validation-split candles for one symbol. The train and test rows are never returned (see dashboard_data.validation_candles)."""
    return validation_candles(load_candles(DATA / f"{symbol}_1h.csv"))


@st.cache_data(show_spinner="Running the paper account…")
def paper_run(asset: str, strategy: str, fee: float, slip: float, max_pos: float, max_order: float, max_day: float, max_dd: float):
    """One paper-account run on the validation split. Cached, so moving a slider back to an earlier value is instant."""
    limits = RiskLimits(max_position=max_pos, max_order=max_order, max_daily_loss=max_day, max_drawdown=max_dd)
    res = run_paper(val_data(asset), STRATEGIES[strategy], limits=limits, fee_bps=fee, slippage_bps=slip)
    return res.frame, res.events, res.metrics


def period_text() -> str:
    """Date range of the BTC validation split, for page headers."""
    if not (DATA / "BTCUSDT_1h.csv").exists():
        return "validation split"
    v = val_data("BTCUSDT")
    return f"{v['timestamps'].iloc[0]:%b %d, %Y} – {v['timestamps'].iloc[-1]:%b %d, %Y}"


def pct(x: float) -> str:
    """Signed percentage with a proper minus sign, or a dash for missing values."""
    return "—" if pd.isna(x) else f"{x:+.1%}".replace("-", "−")


def num(x: float, fmt: str = "{:.2f}") -> str:
    return "—" if pd.isna(x) else fmt.format(x).replace("-", "−")


def show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def table_or_empty(filename: str, fix: str) -> pd.DataFrame | None:
    """Load outputs/<filename>; if it does not exist yet, show a friendly box saying how to create it and return None."""
    table = load_table(OUT / filename)
    if table is None:
        theme.empty_state(f"No results yet: this view reads outputs/{filename}, which is created when the experiment is run.", fix)
    return table


def guarded(render):
    """Wrap a page so a missing file or bad data shows an explanation and a next step instead of a Python traceback."""

    @functools.wraps(render)
    def page():
        try:
            render()
        except FileNotFoundError as err:
            theme.empty_state(f"This page needs a file that is not on this machine yet ({Path(str(err.filename)).name}). Data and results are not stored in the repository.", FETCH_HINT)
        except DataError as err:
            st.error(f"The market data failed its safety checks, so nothing was drawn: {err}. Download the file again to fix it.")
        except Exception as err:  # noqa: BLE001  last-resort screen for a research tool; details stay available
            st.error("Something unexpected went wrong while building this page. The research results themselves are not affected.")
            with st.expander("Technical details"):
                st.exception(err)

    return page
