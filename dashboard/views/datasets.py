"""Datasets: downloaded market data and how it is split."""
import streamlit as st

import theme
from common import DATA, FETCH_HINT
from kronosquant.dashboard_data import dataset_inventory
from kronosquant.data import load_candles
from kronosquant.glossary import GLOSSARY
from kronosquant.splits import chronological_split


def _split_card() -> None:
    theme.card_title("Chronological split (BTC 1h)", help=GLOSSARY["Validation split"])
    path = DATA / "BTCUSDT_1h.csv"
    if path.exists():
        ts = load_candles(path)["timestamps"]
        labels = {"train": "Train", "val": "Validation", "test": "Test"}
        rows = [[labels.get(name, name), f"{ts.iloc[r.start]:%Y-%m-%d}", f"{ts.iloc[r.stop - 1]:%Y-%m-%d}", f"{len(r):,}", "Untouched" if name == "test" else ""]
                for name, r in chronological_split(len(ts)).items()]
        st.markdown(theme.rows_table(["Split", "From", "To", "Rows", "Status"], rows, badge_cols=(4,), num_cols=(3,)), unsafe_allow_html=True)
    else:
        theme.empty_state("BTCUSDT hourly data is not on this machine yet, so the split cannot be shown.", FETCH_HINT)
    theme.note("Splits are consecutive and never shuffled. The test split is opened only after a validation pass, and only once. Known limits: one exchange, "
               "a hand-picked 12-asset universe (survivorship bias), costs assumed rather than verified.")


def _inventory_card() -> None:
    inv = dataset_inventory(DATA)
    theme.card_title("Inventory", help="Every downloaded file with its row count, date range, number of missing candles (gaps) and a fingerprint that changes if any value changes.")
    if inv.empty:
        theme.empty_state("No datasets found in data/.", FETCH_HINT)
        return
    rows = [[r.symbol, r.interval, f"{r.rows:,}", r.first, r.last, str(r.gaps), r.fingerprint, r.fetched_utc] for r in inv.itertuples()]
    st.markdown(theme.rows_table(["Symbol", "Interval", "Rows", "First", "Last", "Gaps", "Fingerprint", "Fetched"], rows, num_cols=(2, 5)), unsafe_allow_html=True)


def render() -> None:
    theme.page_header("Datasets", "Downloaded market data and how it is split. Public Binance data, no keys.", chip="Binance public klines")
    with st.container(border=True):
        _split_card()
    st.write("")
    with st.container(border=True):
        _inventory_card()
