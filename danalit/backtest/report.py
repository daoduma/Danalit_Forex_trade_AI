"""HTML backtest report (plotly): equity + drawdown, monthly returns, trade
distributions, and the cost breakdown showing exactly how much spread/swap ate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from danalit.backtest.metrics import monthly_returns


def _fmt(v, pct=False, digits=2) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{100 * v:.{digits}f}%"
    return f"{v:,.{digits}f}"


def build_report(
    results: dict,
    summary: dict,
    title: str,
    out_path: Path,
    extra_sections: Optional[list[str]] = None,
) -> Path:
    import plotly.graph_objects as go
    import plotly.io as pio

    eq: pd.DataFrame = results["equity_curve"]
    trades = results["trades"]
    parts: list[str] = [f"<h1>{title}</h1>"]

    # headline table
    rows = [
        ("Net profit", _fmt(summary.get("net_profit"))),
        ("Return", _fmt(summary.get("return_pct"), pct=True)),
        ("CAGR", _fmt(summary.get("cagr"), pct=True)),
        ("Profit factor", _fmt(summary.get("profit_factor"))),
        ("Expectancy / trade", _fmt(summary.get("expectancy"), digits=4)),
        ("Win rate", _fmt(summary.get("win_rate"), pct=True)),
        ("Max DD (equity)", _fmt(summary.get("max_drawdown_equity"), pct=True)),
        ("Max DD (balance)", _fmt(summary.get("max_drawdown_balance"), pct=True)),
        ("Sharpe", _fmt(summary.get("sharpe"))),
        ("Sortino", _fmt(summary.get("sortino"))),
        ("Trades", f"{summary.get('n_trades', 0):,}"),
        ("Max consecutive losses", str(summary.get("consecutive_losses", "—"))),
        ("Avg duration (h)", _fmt(summary.get("avg_duration_h"))),
    ]
    parts.append("<table border=1 cellpadding=4><tr>" +
                 "".join(f"<th>{k}</th>" for k, _ in rows) + "</tr><tr>" +
                 "".join(f"<td>{v}</td>" for _, v in rows) + "</tr></table>")

    if not eq.empty:
        dd = eq["equity"] / eq["equity"].cummax() - 1
        fig = go.Figure()
        fig.add_scatter(x=eq.index, y=eq["equity"], name="equity", line=dict(width=1))
        fig.add_scatter(x=eq.index, y=eq["balance"], name="balance", line=dict(width=1, dash="dot"))
        fig.update_layout(title="Equity curve", height=350, margin=dict(l=40, r=20, t=40, b=30))
        parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        fig2 = go.Figure()
        fig2.add_scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="drawdown %")
        fig2.update_layout(title="Drawdown (%)", height=220, margin=dict(l=40, r=20, t=40, b=30))
        parts.append(pio.to_html(fig2, include_plotlyjs=False, full_html=False))

    mt = monthly_returns(eq)
    if not mt.empty:
        parts.append("<h2>Monthly returns (%)</h2>" + mt.to_html(border=1, na_rep=""))

    if trades:
        pnls = [t.net_pnl for t in trades]
        fig3 = go.Figure()
        fig3.add_histogram(x=pnls, nbinsx=60)
        fig3.update_layout(title="Trade net P&L distribution", height=250,
                           margin=dict(l=40, r=20, t=40, b=30))
        parts.append(pio.to_html(fig3, include_plotlyjs=False, full_html=False))

        gross = summary.get("gross_pnl", 0.0)
        com = summary.get("total_commission", 0.0)
        swap = summary.get("total_swap", 0.0)
        spread = summary.get("total_spread_cost", 0.0)
        net = summary.get("net_profit", 0.0)
        parts.append(
            "<h2>Cost breakdown — what execution ate</h2>"
            "<table border=1 cellpadding=4>"
            f"<tr><th>Gross P&L (post-spread fills)</th><td>{_fmt(gross)}</td></tr>"
            f"<tr><th>Spread paid (implicit in fills)</th><td>{_fmt(spread)}</td></tr>"
            f"<tr><th>Commission</th><td>{_fmt(-com)}</td></tr>"
            f"<tr><th>Swap</th><td>{_fmt(swap)}</td></tr>"
            f"<tr><th><b>Net</b></th><td><b>{_fmt(net)}</b></td></tr>"
            "</table>"
            "<p><i>Gross already includes the spread (paid at fill); the spread row shows "
            "how much that cost so gross-if-spread-free can be reconstructed.</i></p>"
        )

    for section in extra_sections or []:
        parts.append(section)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<html><head><meta charset='utf-8'><title>" + title + "</title></head><body>"
        + "\n".join(parts) + "</body></html>",
        encoding="utf-8",
    )
    return out_path
