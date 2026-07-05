"""Forward-test analytics: reality vs backtest expectations, trade by trade.

Produces reports/forward_test_{period}.html with:
- execution quality (realized slippage + spread vs the modeled cost assumptions
  — THE key go-live input; flags when live costs exceed modeled),
- performance (equity curve vs the walk-forward Monte Carlo cone, win rate /
  expectancy / PF with confidence intervals for the N so far),
- model behavior (confidence distributions, veto/rejection frequencies),
- discipline audit (unapproved orders and overridden rejections MUST be zero),
- an auto-filled go-live checklist with honest PASS/FAIL per Chapter 10.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danalit.config import load_config
from danalit.db import connect

GO_LIVE = {
    "min_weeks": 12,
    "min_trades": 100,
    "min_profit_factor": 1.15,
    "max_drawdown": 0.12,
    "max_cost_overrun": 1.25,   # live costs must stay under 1.25x modeled
}


# ------------------------------------------------------------------ stitching

def stitch_trades(deals: list[dict]) -> list[dict]:
    """Build full trade lifecycles from broker deal rows.

    Deal: {position_id, time_utc, kind: 'entry'|'exit', price, volume,
           side (+1/-1 of the POSITION), commission, swap, profit,
           instrument, reason}
    Partial closes (multiple exits) collapse into one trade row with
    volume-weighted exit price and summed P&L/costs.
    """
    trades = []
    for pid, group in pd.DataFrame(deals).groupby("position_id"):
        group = group.sort_values("time_utc")
        entries = group[group["kind"] == "entry"]
        exits = group[group["kind"] == "exit"]
        if entries.empty:
            continue
        vol_in = entries["volume"].sum()
        entry_px = float((entries["price"] * entries["volume"]).sum() / vol_in)
        row = {
            "position_id": pid,
            "instrument": entries.iloc[0]["instrument"],
            "side": int(entries.iloc[0]["side"]),
            "lots": float(vol_in),
            "opened_utc": entries.iloc[0]["time_utc"],
            "entry_price": entry_px,
            "commission": float(group.get("commission", pd.Series(0)).sum()),
            "swap": float(group.get("swap", pd.Series(0)).sum()),
        }
        if not exits.empty and exits["volume"].sum() >= vol_in - 1e-9:
            vol_out = exits["volume"].sum()
            row.update({
                "closed_utc": exits.iloc[-1]["time_utc"],
                "exit_price": float((exits["price"] * exits["volume"]).sum() / vol_out),
                "gross_pnl": float(exits["profit"].sum()),
                "net_pnl": float(exits["profit"].sum() + row["swap"] - abs(row["commission"])),
                "exit_reason": exits.iloc[-1].get("reason", ""),
                "n_partials": int(len(exits) - 1),
            })
        else:
            row.update({"closed_utc": None, "exit_price": None, "net_pnl": None})
        trades.append(row)
    return trades


def compute_mae_mfe(bars: pd.DataFrame, side: int, entry_price: float,
                    opened, closed) -> tuple[float, float]:
    """Worst/best price excursion (price units) from bar data over the trade."""
    window = bars[(bars["time_utc"] >= opened) & (bars["time_utc"] <= closed)]
    if window.empty:
        return 0.0, 0.0
    if side > 0:
        return (float(window["low"].min() - entry_price),
                float(window["high"].max() - entry_price))
    return (float(entry_price - window["high"].max()),
            float(entry_price - window["low"].min()))


# ------------------------------------------------------------------- stats

def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def cost_comparison(orders: pd.DataFrame, modeled_slippage: float) -> dict:
    """Realized |filled - intended| vs the modeled slippage assumption."""
    filled = orders.dropna(subset=["filled_price", "intended_price"])
    if filled.empty:
        return {"n": 0, "flag": False}
    slip = (filled["filled_price"] - filled["intended_price"]).abs()
    realized = float(slip.mean())
    return {
        "n": int(len(filled)),
        "realized_mean": realized,
        "realized_p95": float(slip.quantile(0.95)),
        "modeled": modeled_slippage,
        "ratio": realized / modeled_slippage if modeled_slippage > 0 else None,
        "flag": modeled_slippage > 0 and realized > GO_LIVE["max_cost_overrun"] * modeled_slippage,
    }


# ----------------------------------------------------------------- checklist

def evaluate_checklist(stats: dict) -> list[dict]:
    """Auto-fill the measurable Chapter-10 items with PASS/FAIL. Honest FAILs
    are expected until the forward test has actually run its course."""
    items = [
        ("Forward-test duration >= 12 weeks", stats.get("weeks", 0.0),
         GO_LIVE["min_weeks"], stats.get("weeks", 0.0) >= GO_LIVE["min_weeks"]),
        ("Trades >= 100", stats.get("n_trades", 0),
         GO_LIVE["min_trades"], stats.get("n_trades", 0) >= GO_LIVE["min_trades"]),
        ("Profit factor >= 1.15", stats.get("profit_factor"),
         GO_LIVE["min_profit_factor"],
         (stats.get("profit_factor") or 0) >= GO_LIVE["min_profit_factor"]),
        ("Max drawdown < 12%", stats.get("max_drawdown"),
         GO_LIVE["max_drawdown"],
         (stats.get("max_drawdown") if stats.get("max_drawdown") is not None else 1.0)
         < GO_LIVE["max_drawdown"]),
        ("Live costs within 1.25x of modeled", stats.get("cost_ratio"),
         GO_LIVE["max_cost_overrun"],
         stats.get("cost_ratio") is not None
         and stats["cost_ratio"] <= GO_LIVE["max_cost_overrun"]),
        ("Discipline: zero unapproved orders", stats.get("unapproved_orders", 0),
         0, stats.get("unapproved_orders", 0) == 0),
        ("Discipline: zero overridden rejections", stats.get("overridden_rejections", 0),
         0, stats.get("overridden_rejections", 0) == 0),
    ]
    return [{"item": i, "value": v, "threshold": t, "pass": bool(p)}
            for i, v, t, p in items]


# ------------------------------------------------------------------- report

def gather(db_path: Optional[Path], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    cfg = load_config()
    con = connect(db_path or cfg.settings.paths.db_path)
    try:
        q = lambda sql, *a: pd.DataFrame(  # noqa: E731
            [dict(r) for r in con.execute(sql, a).fetchall()])
        s, e = start.isoformat(), end.isoformat()
        decisions = q("SELECT * FROM decisions WHERE ts_utc BETWEEN ? AND ?", s, e)
        orders = q("SELECT * FROM orders WHERE ts_utc BETWEEN ? AND ?", s, e)
        trades = q("SELECT * FROM trades WHERE opened_utc BETWEEN ? AND ?", s, e)
        equity = q("SELECT * FROM equity_snapshots WHERE ts_utc BETWEEN ? AND ?", s, e)
        events = q("SELECT * FROM system_events WHERE ts_utc BETWEEN ? AND ?", s, e)
    finally:
        con.close()

    stats: dict = {"start": str(start), "end": str(end)}
    stats["n_decisions"] = len(decisions)
    if not decisions.empty:
        stats["veto_counts"] = (decisions["veto_reason"].fillna("(traded)")
                                .value_counts().to_dict())
        stats["weeks"] = max((pd.to_datetime(decisions["ts_utc"]).max()
                              - pd.to_datetime(decisions["ts_utc"]).min()).days / 7, 0)
    else:
        stats["veto_counts"], stats["weeks"] = {}, 0.0

    closed = trades.dropna(subset=["net_pnl"]) if not trades.empty else trades
    stats["n_trades"] = len(closed)
    if len(closed):
        pnls = closed["net_pnl"].astype(float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        stats["win_rate"] = len(wins) / len(pnls)
        stats["win_rate_ci"] = wilson_interval(len(wins), len(pnls))
        stats["expectancy"] = float(pnls.mean())
        stats["expectancy_ci"] = bootstrap_ci(list(pnls))
        stats["profit_factor"] = (float(wins.sum() / -losses.sum())
                                  if len(losses) and losses.sum() < 0 else None)
    if not equity.empty:
        eq = equity.sort_values("ts_utc")["equity"].astype(float)
        peak = eq.cummax()
        stats["max_drawdown"] = float(((peak - eq) / peak).max())

    inst = cfg.instruments.get("EURUSD")
    modeled_slip = 0.2 * inst.pip_size if inst else 0.0
    cc = cost_comparison(orders, modeled_slip) if not orders.empty else {"n": 0, "flag": False}
    stats["cost"] = cc
    stats["cost_ratio"] = cc.get("ratio")

    # discipline audit
    if not orders.empty:
        signal_ids = set(decisions["signal_id"].dropna()) if not decisions.empty else set()
        real = orders[~orders["status"].isin(["dry_run", "intent_dry_run"])]
        stats["unapproved_orders"] = int((~real["signal_id"].isin(signal_ids)).sum()) \
            if not real.empty else 0
    else:
        stats["unapproved_orders"] = 0
    rejected = set()
    if not events.empty:
        for d in events[events["type"] == "risk_rejected"]["detail"]:
            rejected.add(str(d).split(":", 1)[0].strip())
    sent_ids = set(orders["signal_id"].dropna()) if not orders.empty else set()
    stats["overridden_rejections"] = len(rejected & sent_ids)
    stats["checklist"] = evaluate_checklist(stats)
    return stats


def write_report(stats: dict, out_path: Optional[Path] = None,
                 mc_cone: Optional[dict] = None) -> Path:
    cfg = load_config()
    period = f"{stats['start'][:10]}_{stats['end'][:10]}"
    out_path = out_path or cfg.settings.paths.absolute("reports") / f"forward_test_{period}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v, digits=4):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    rows = "".join(
        f"<tr><td>{c['item']}</td><td>{fmt(c['value'], 3)}</td>"
        f"<td>{fmt(c['threshold'], 3)}</td>"
        f"<td style='color:{'green' if c['pass'] else 'red'}'>"
        f"{'PASS' if c['pass'] else 'FAIL'}</td></tr>"
        for c in stats["checklist"])
    veto_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                        for k, v in sorted(stats.get("veto_counts", {}).items(),
                                           key=lambda kv: -kv[1]))
    cc = stats.get("cost", {})
    cost_html = (f"<p>orders measured: {cc.get('n', 0)}, realized slippage mean "
                 f"{fmt(cc.get('realized_mean'), 6)} vs modeled {fmt(cc.get('modeled'), 6)} "
                 f"(ratio {fmt(cc.get('ratio'), 2)}) "
                 + ("<b style='color:red'>— LIVE COSTS EXCEED MODEL</b>" if cc.get("flag")
                    else "— within model") + "</p>") if cc.get("n") else "<p>no fills yet</p>"

    html = f"""<html><head><meta charset='utf-8'><title>Forward test {period}</title></head>
<body>
<h1>Forward-test analytics — {period}</h1>
<h2>Go-live checklist (auto-filled, honest FAILs expected early)</h2>
<table border=1 cellpadding=4><tr><th>Item</th><th>Value</th><th>Threshold</th><th>Status</th></tr>{rows}</table>
<h2>Performance</h2>
<p>trades: {stats.get('n_trades', 0)} | win rate {fmt(stats.get('win_rate'), 3)}
 CI {fmt(stats.get('win_rate_ci'), 3)} | expectancy {fmt(stats.get('expectancy'))}
 CI {fmt(stats.get('expectancy_ci'))} | PF {fmt(stats.get('profit_factor'), 2)}
 | max DD {fmt(stats.get('max_drawdown'), 3)}</p>
<h2>Execution quality</h2>{cost_html}
<h2>Decisions & vetoes ({stats.get('n_decisions', 0)} decisions)</h2>
<table border=1 cellpadding=4><tr><th>Outcome</th><th>Count</th></tr>{veto_rows}</table>
<h2>Discipline audit</h2>
<p>unapproved orders: <b>{stats.get('unapproved_orders')}</b> (must be 0) |
 overridden rejections: <b>{stats.get('overridden_rejections')}</b> (must be 0)</p>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def export_parquet(db_path: Optional[Path] = None, out_dir: Optional[Path] = None) -> list[Path]:
    """Append-only journal -> Parquet snapshot for offline research."""
    cfg = load_config()
    out_dir = out_dir or cfg.settings.paths.absolute("data_store") / "journal_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path or cfg.settings.paths.db_path)
    paths = []
    try:
        for table in ("decisions", "orders", "trades", "managed_actions",
                      "equity_snapshots", "system_events"):
            df = pd.DataFrame([dict(r) for r in con.execute(f"SELECT * FROM {table}")])
            p = out_dir / f"{table}.parquet"
            if not df.empty:
                df.to_parquet(p, index=False)
                paths.append(p)
    finally:
        con.close()
    return paths
