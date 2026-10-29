"""Plotly figure builders for the dashboard (light theme, design-reference palette).

Colours were checked with the dataviz validator on a white surface: blue #3B82F6, green #10B981 and indigo #6366F1 pass every hard gate
(green is under 3:1 contrast on white, so every series is also named in a legend and labelled at its end, and each page has a table view).
Candles: hollow blue = up, filled navy = down, so direction never rests on colour alone and never on red/green. Red is used only for
drawdown and limit trips, always with a text label."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SURFACE = "#FFFFFF"
PRIMARY, SECONDARY, SUCCESS, WARNING, ERROR = "#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EF4444"
NAVY, SLATE, MUTED, GRID, AXIS = "#0F172A", "#475569", "#64748B", "#E8EDF4", "#CBD5E1"
NEUTRAL = "#94A3B8"
FONT = dict(family='Inter, system-ui, -apple-system, "Segoe UI", sans-serif', color=SLATE, size=12)


def _style(fig: go.Figure, height: int, right_margin: int = 24) -> None:
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=FONT, height=height, margin=dict(l=60, r=right_margin, t=44, b=32),
        hovermode="x unified", hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=NAVY, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=SLATE, size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=11), showspikes=True, spikecolor=NEUTRAL,
                     spikethickness=1, spikedash="dot", spikemode="across", rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=11))


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def performance_figure(series: dict[str, pd.Series], colors: dict[str, str], height: int = 340) -> go.Figure:
    """Cumulative net return in percent for several strategies on one axis. `series` maps a name to equity (1.0 = start) indexed by time.

    The first series is the focus; the last is drawn dotted as the benchmark. Every line is labelled at its end."""
    fig = go.Figure()
    names = list(series)
    for i, name in enumerate(names):
        s = (series[name] - 1.0) * 100
        dash = "dot" if i == len(names) - 1 and len(names) > 1 else "solid"
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=name, line=dict(color=colors[name], width=2.2 if i == 0 else 1.8, dash=dash),
                                 hovertemplate="%{y:+.1f}%<extra>" + name + "</extra>"))
        fig.add_annotation(x=s.index[-1], y=s.iloc[-1], text=f"{s.iloc[-1]:+.0f}%", showarrow=False, xanchor="left", xshift=6, font=dict(color=SLATE, size=11))
    _style(fig, height, right_margin=56)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor=AXIS)
    return fig


def trading_figure(candles: pd.DataFrame, frame: pd.DataFrame, benchmark: pd.Series, markers: pd.DataFrame, trips: pd.DataFrame, kill_level: float) -> go.Figure:
    """Four stacked panels on one time axis: price with trade markers, position, equity versus buy & hold, drawdown.

    `candles`, `frame` and `benchmark` must cover the same bars (the caller slices them to the chosen window)."""
    ts = candles["timestamps"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.5, 0.12, 0.2, 0.18])

    fig.add_trace(go.Candlestick(x=ts, open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"], name="Price",
                                 increasing=dict(line=dict(color=PRIMARY, width=1), fillcolor=SURFACE), decreasing=dict(line=dict(color=NAVY, width=1), fillcolor=NAVY)), row=1, col=1)
    for side, symbol, label in [("buy", "triangle-up", "Buy (position up)"), ("sell", "triangle-down", "Sell (position down)")]:
        m = markers[markers["side"] == side]
        if m.empty:
            continue
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["price"], mode="markers", name=label, marker=dict(symbol=symbol, size=10, color=SECONDARY, line=dict(color=SURFACE, width=2)),
                                 customdata=m[["position_change", "position_after"]], hovertemplate=f"{side} %{{y:,.2f}}  (position %{{customdata[1]:+.2f}}, change %{{customdata[0]:+.2f}})<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=frame.index, y=frame["position"], mode="lines", line=dict(color=PRIMARY, width=1.5, shape="hv"), fill="tozeroy", fillcolor=_rgba(PRIMARY, 0.2),
                             name="Position", hovertemplate="%{y:+.2f}<extra>Position</extra>", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["equity"], mode="lines", line=dict(color=PRIMARY, width=2.2), name="Paper account", hovertemplate="%{y:.4f}<extra>Paper account</extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, mode="lines", line=dict(color=NEUTRAL, width=1.6, dash="dot"), name="Buy & hold (no costs)", hovertemplate="%{y:.4f}<extra>Buy & hold</extra>"), row=3, col=1)
    dd = frame["equity"] / frame["equity"].cummax() - 1.0  # drawdown within the visible window's running peak
    fig.add_trace(go.Scatter(x=frame.index, y=dd, mode="lines", line=dict(color=ERROR, width=1), fill="tozeroy", fillcolor=_rgba(ERROR, 0.22),
                             name="Drawdown", hovertemplate="%{y:.1%}<extra>Drawdown</extra>", showlegend=False), row=4, col=1)

    for y_end in (frame["equity"].iloc[-1], benchmark.iloc[-1]):
        fig.add_annotation(x=frame.index[-1], y=y_end, xref="x3", yref="y3", text=f"{y_end:.3f}", showarrow=False, xanchor="left", xshift=6, font=dict(color=SLATE, size=11))
    fig.add_hline(y=-kill_level, row=4, col=1, line=dict(color=ERROR, width=1, dash="dash"))
    fig.add_annotation(x=frame.index[0], y=-kill_level, xref="x4", yref="y4", text=f"kill switch at −{kill_level:.0%}", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=SLATE, size=11))

    for r in trips.itertuples():
        text = "KILL SWITCH" if r.rule == "kill_switch_tripped" else "DAILY HALT"
        x = pd.Timestamp(r.timestamp).to_pydatetime()
        fig.add_shape(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1, line=dict(color=ERROR, width=1, dash="dot"))
        fig.add_annotation(x=x, y=1, xref="x", yref="paper", text=f"✕ {text}", showarrow=False, xanchor="left", yanchor="top", xshift=4, yshift=-4, font=dict(color="#B91C1C", size=11),
                           bgcolor=_rgba(ERROR, 0.12))

    _style(fig, 760, right_margin=64)
    fig.update_yaxes(title_text="Price", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="Position", row=2, col=1, range=[-1.05, 1.05], tickvals=[-1, 0, 1])
    fig.update_yaxes(title_text="Equity", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown", row=4, col=1, tickformat=".0%")
    fig.update_xaxes(range=[ts.iloc[0], ts.iloc[-1]])
    lo, hi = float(candles["low"].min()), float(candles["high"].max())
    pad = (hi - lo) * 0.04 or 1.0
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    return fig


def forest_figure(rows: pd.DataFrame, xlabel: str) -> go.Figure:
    """One row per comparison: dot = estimate, bar = 99% interval, dashed line = no skill. Filled dot = passed, hollow = not passed."""
    rows = rows.reset_index(drop=True)
    labels = [f"{e.split(':')[0]} · {s}" for e, s in zip(rows["experiment"], rows["signal"])]
    fig = go.Figure(go.Scatter(
        x=rows["estimate"], y=labels, mode="markers",
        marker=dict(size=10, symbol=["circle" if p else "circle-open" for p in rows["passed"]], color=PRIMARY, line=dict(color=PRIMARY, width=2)),
        error_x=dict(type="data", symmetric=False, array=rows["ci_hi"] - rows["estimate"], arrayminus=rows["estimate"] - rows["ci_lo"], color=NEUTRAL, thickness=1.5, width=4),
        customdata=rows[["ci_lo", "ci_hi"]], hovertemplate="%{y}<br>estimate %{x:+.4f}  [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<extra></extra>", showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dash"))
    _style(fig, 60 + 34 * len(rows))
    fig.update_layout(hovermode="closest", margin=dict(l=24, r=24, t=16, b=44))
    fig.update_xaxes(title_text=xlabel, title_font=dict(color=MUTED, size=11), showspikes=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=SLATE, size=11), automargin=True)
    return fig


def distribution_figure(values: np.ndarray, mean: float, lo: float, hi: float, xlabel: str, level: float = 0.99) -> go.Figure:
    """Histogram of per-window results with the mean (dashed) and its confidence interval (shaded band)."""
    fig = go.Figure(go.Histogram(x=values, nbinsx=30, marker=dict(color=_rgba(PRIMARY, 0.55), line=dict(color=SURFACE, width=1)), name="Windows",
                                 hovertemplate="%{x:.0f}: %{y} windows<extra></extra>"))
    fig.add_vrect(x0=lo, x1=hi, fillcolor=_rgba(SECONDARY, 0.15), line_width=0)  # the shaded band is the interval; its numbers are in the caption
    fig.add_vline(x=mean, line=dict(color=NAVY, width=1.5, dash="dash"), annotation_text=f"mean {mean:+.1f}", annotation_position="top right", annotation_font=dict(color=NAVY, size=11))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    _style(fig, 300)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.05, margin=dict(l=48, r=24, t=36, b=44))
    fig.update_xaxes(title_text=xlabel, title_font=dict(color=MUTED, size=11), showspikes=False)
    fig.update_yaxes(title_text="Windows")
    return fig


def comparison_bars(labels: list[str], values: list[float], reference: float, reference_label: str, ytitle: str) -> go.Figure:
    """Bars for several settings against one reference line (e.g. path error versus the random walk). Lower is better."""
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=PRIMARY, cornerradius=4), text=[f"{v:.4f}" for v in values], textposition="outside",
                           textfont=dict(color=SLATE, size=11), hovertemplate="%{x}: %{y:.5f}<extra></extra>"))
    fig.add_hline(y=reference, line=dict(color=NAVY, width=1.5, dash="dash"), annotation_text=reference_label, annotation_position="top left", annotation_font=dict(color=NAVY, size=11))
    _style(fig, 320)
    fig.update_layout(hovermode="closest", showlegend=False, margin=dict(l=56, r=24, t=36, b=40))
    fig.update_yaxes(title_text=ytitle, rangemode="tozero")
    fig.update_xaxes(showspikes=False)
    return fig


def cost_sensitivity_figure(costs: list[float], net_returns: list[float]) -> go.Figure:
    """Net return of one strategy as the one-way cost assumption rises."""
    fig = go.Figure(go.Scatter(x=costs, y=[r * 100 for r in net_returns], mode="lines+markers", line=dict(color=PRIMARY, width=2.2), marker=dict(size=9, color=PRIMARY, line=dict(color=SURFACE, width=2)),
                               hovertemplate="%{x} bps: %{y:+.1f}%<extra></extra>"))
    _style(fig, 300)
    fig.update_layout(hovermode="closest", showlegend=False, margin=dict(l=56, r=24, t=24, b=44))
    fig.update_xaxes(title_text="One-way cost (bps)", title_font=dict(color=MUTED, size=11), showspikes=False)
    fig.update_yaxes(title_text="Net return", ticksuffix="%", zeroline=True, zerolinecolor=AXIS)
    return fig
