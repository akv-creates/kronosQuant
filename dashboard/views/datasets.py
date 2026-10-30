"""Datasets: downloaded market data and how it is split. On the hosted dashboard these come from the published snapshot (summary numbers only)."""
import pandas as pd
import streamlit as st

import theme
from common import DATA, FETCH_HINT, HAS_LOCAL_DATA, OUT, PUBLIC_MODE
from kronosquant.dashboard_data import dataset_inventory, load_table, split_table
from kronosquant.data import load_candles
from kronosquant.glossary import GLOSSARY

SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def _split_data() -> pd.DataFrame | None:
    """The BTC 1h chronological split, from the local candles when they exist, else from the published snapshot."""
    path = DATA / "BTCUSDT_1h.csv"
    if HAS_LOCAL_DATA and path.exists():
        return split_table(load_candles(path)["timestamps"])
    return load_table(OUT / "splits.csv")


def _split_card() -> None:
    theme.card_title("Chronological split (BTC 1h)", help=GLOSSARY["Validation split"])
    splits = _split_data()
    if splits is None:
        theme.empty_state("BTCUSDT hourly data is not on this machine yet, so the split cannot be shown.", FETCH_HINT)
    else:
        rows = [[SPLIT_LABELS.get(name, name), first, last, f"{n:,}", "Untouched" if name == "test" else ""] for name, first, last, n in zip(splits["split"], splits["from"], splits["to"], splits["rows"])]
        st.markdown(theme.rows_table(["Split", "From", "To", "Rows", "Status"], rows, badge_cols=(4,), num_cols=(3,)), unsafe_allow_html=True)
    theme.note("Splits are consecutive and never shuffled. The test split is opened only after a validation pass, and only once. Known limits: one exchange, "
               "a hand-picked 12-asset universe (survivorship bias), costs assumed rather than verified.")


def _inventory_card() -> None:
    inv = dataset_inventory(DATA) if HAS_LOCAL_DATA else load_table(OUT / "datasets_inventory.csv")
    theme.card_title("Inventory", help="Every dataset used, with its row count, date range, number of missing candles (gaps) and a fingerprint that changes if any value changes.")
    if inv is None or inv.empty:
        theme.empty_state("No datasets found in data/.", FETCH_HINT)
        return
    rows = [[r.symbol, r.interval, f"{r.rows:,}", r.first, r.last, str(r.gaps), r.fingerprint, r.fetched_utc] for r in inv.itertuples()]
    st.markdown(theme.rows_table(["Symbol", "Interval", "Rows", "First", "Last", "Gaps", "Fingerprint", "Fetched"], rows, num_cols=(2, 5)), unsafe_allow_html=True)
    if PUBLIC_MODE:
        theme.note("Only this summary is published. The raw candles come from Binance's public market data and are not redistributed here; the Reports page shows how to download them.")


def render() -> None:
    theme.page_header("Datasets", "Market data used and how it is split. Public Binance data, no keys.", chip="Binance public klines")
    with st.container(border=True):
        _split_card()
    st.write("")
    with st.container(border=True):
        _inventory_card()
