"""Look and feel for the dashboard: palette, CSS and small HTML helpers (cards, badges, tables). Palette follows the design reference:
primary #3B82F6, secondary #6366F1, success #10B981, warning #F59E0B, error #EF4444, navy #0F172A, slate #64748B, light #E2E8F0.
Status is always shown as text in a badge, never by colour alone."""
import html

import streamlit as st

PRIMARY, SECONDARY, SUCCESS, WARNING, ERROR = "#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EF4444"
NAVY, SLATE, LIGHT = "#0F172A", "#64748B", "#E2E8F0"

BADGES = {  # label -> (background, text colour)
    "Completed": ("#D1FAE5", "#047857"), "Passed": ("#D1FAE5", "#047857"), "Pending": ("#FEF3C7", "#B45309"), "Failed": ("#FEE2E2", "#B91C1C"),
    "Inconclusive": ("#E0E7FF", "#4338CA"), "Exploratory": ("#F1F5F9", "#475569"), "Placeholder": ("#F1F5F9", "#475569"), "Untouched": ("#D1FAE5", "#047857"),
}

CSS = f"""
<style>
  html, body, [class*="css"], .stApp {{ font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding: 1.4rem 2.2rem 3rem 2.2rem; max-width: 1440px; }}
  .page-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap; margin-bottom:14px; }}
  .page-title {{ font-size:1.6rem; font-weight:650; color:{NAVY}; line-height:1.25; }}
  .page-sub {{ color:{SLATE}; font-size:0.88rem; margin-top:2px; }}
  .chip {{ display:inline-block; border:1px solid {LIGHT}; background:#fff; color:#475569; border-radius:8px; padding:6px 12px; font-size:0.78rem; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border:1px solid {LIGHT}; border-radius:12px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }}
  .card-title {{ font-size:0.95rem; font-weight:650; color:{NAVY}; margin:0 0 8px 0; }}
  .kpi {{ background:#fff; border:1px solid {LIGHT}; border-radius:12px; padding:14px 18px; box-shadow:0 1px 2px rgba(15,23,42,0.04); height:100%; }}
  .kpi .l {{ color:{SLATE}; font-size:0.8rem; }}
  .kpi .v {{ color:{NAVY}; font-size:2rem; font-weight:650; line-height:1.2; font-variant-numeric:tabular-nums; }}
  .kpi .s {{ color:{SLATE}; font-size:0.78rem; }}
  .kpi .s.good {{ color:#047857; }} .kpi .s.bad {{ color:#B91C1C; }} .kpi .s.warn {{ color:#B45309; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; white-space:nowrap; }}
  .tbl-wrap {{ width:100%; max-width:100%; overflow-x:auto; }}
  .rows {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
  .rows th {{ text-align:left; color:{SLATE}; font-weight:600; font-size:0.74rem; padding:8px 10px; border-bottom:1px solid {LIGHT}; }}
  .rows td {{ padding:10px 10px; border-bottom:1px solid #EEF2F7; color:{NAVY}; vertical-align:middle; }}
  .rows td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .kv {{ width:100%; border-collapse:collapse; font-size:0.86rem; }}
  .kv td {{ padding:9px 6px; border-bottom:1px solid #EEF2F7; vertical-align:top; }}
  .kv td:first-child {{ color:{SLATE}; width:34%; }}
  .kv td:last-child {{ color:{NAVY}; word-break:break-word; }}
  .note {{ color:{SLATE}; font-size:0.78rem; }}
  .stButton > button[kind="primary"] {{ border-radius:8px; font-weight:600; }}
  button[data-baseweb="tab"] {{ font-size:0.9rem; font-weight:500; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(label: str) -> str:
    bg, fg = BADGES.get(label, ("#F1F5F9", "#475569"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{html.escape(label)}</span>'


def page_header(title: str, subtitle: str, chip: str = "", badge_label: str = "") -> None:
    right = f'<div class="chip">{html.escape(chip)}</div>' if chip else ""
    st.markdown(f'<div class="page-head"><div><div class="page-title">{html.escape(title)} {badge(badge_label) if badge_label else ""}</div>'
                f'<div class="page-sub">{html.escape(subtitle)}</div></div>{right}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div><div class="s {tone}">{html.escape(sub)}&nbsp;</div></div>'


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards), gap="medium")
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def card_title(text: str) -> None:
    st.markdown(f'<div class="card-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def kv_table(pairs: list[tuple[str, str]]) -> str:
    return '<table class="kv">' + "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in pairs) + "</table>"


def rows_table(headers: list[str], rows: list[list[str]], badge_cols: tuple[int, ...] = (), num_cols: tuple[int, ...] = (), wrap_cols: tuple[int, ...] = (0,)) -> str:
    """HTML table; cells in badge_cols are rendered as status badges, num_cols are right-aligned, only wrap_cols may wrap onto several lines.
    All text is escaped."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        cells = ""
        for i, v in enumerate(r):
            inner = badge(str(v)) if i in badge_cols else html.escape(str(v))
            style = "" if i in wrap_cols else ' style="white-space:nowrap"'
            cells += f'<td class="{"num" if i in num_cols else ""}"{style}>{inner}</td>'
        body += f"<tr>{cells}</tr>"
    return f'<div class="tbl-wrap"><table class="rows"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
