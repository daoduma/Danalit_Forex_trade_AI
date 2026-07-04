"""Data quality report per instrument: gaps, dupes, OHLC sanity, weekend data, spreads."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

WEEKEND_GAP_START = (4, 20, 0)  # Friday 20:00 UTC or later ...
WEEKEND_GAP_END = (6, 23, 59)  # ... through Sunday: gaps spanning this window are expected


def analyze(df: pd.DataFrame, timeframe_minutes: int = 1, gap_factor: int = 10) -> dict:
    """Compute quality metrics for a bar frame (must be sorted by time_utc)."""
    m: dict = {"n_bars": len(df)}
    if df.empty:
        return m
    ts = pd.to_datetime(df["time_utc"]).dt.tz_convert("UTC")
    m["start"] = str(ts.iloc[0])
    m["end"] = str(ts.iloc[-1])
    m["duplicates"] = int(ts.duplicated().sum())
    m["zero_volume_bars"] = int((df["tick_volume"] == 0).sum()) if "tick_volume" in df else 0
    bad_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    m["ohlc_violations"] = int(bad_ohlc.sum())
    # Weekend bars: Saturday all day, Sunday before 21:00 UTC (session reopen ~21-22 UTC)
    wd, hour = ts.dt.weekday, ts.dt.hour
    m["weekend_bars"] = int(((wd == 5) | ((wd == 6) & (hour < 21))).sum())

    # Gap scan: a delta much larger than the bar interval, not explained by the weekend break.
    deltas = ts.diff().dt.total_seconds().div(60).fillna(timeframe_minutes)
    threshold = timeframe_minutes * gap_factor
    gaps = []
    for i in deltas[deltas > threshold].index:
        prev_t, cur_t = ts.loc[i - 1] if i > 0 else ts.iloc[0], ts.loc[i]
        if _spans_weekend(prev_t, cur_t):
            continue
        gaps.append({"from": str(prev_t), "to": str(cur_t), "minutes": float(deltas.loc[i])})
    m["gaps"] = gaps
    m["gap_count"] = len(gaps)

    if "spread" in df.columns and df["spread"].notna().any():
        med = float(df["spread"].median())
        m["spread_median"] = med
        m["spread_outliers"] = int((df["spread"] > 10 * med).sum()) if med > 0 else 0
    return m


def _spans_weekend(prev_t: pd.Timestamp, cur_t: pd.Timestamp) -> bool:
    """True if the gap [prev_t, cur_t] plausibly contains the Fri-close→Sun-open break."""
    if cur_t - prev_t > pd.Timedelta(days=3):
        return False  # too long even for a weekend
    fri_ok = prev_t.weekday() == 4 and prev_t.hour >= WEEKEND_GAP_START[1]
    sat = prev_t.weekday() == 5
    sun_resume = cur_t.weekday() == 6 or (cur_t.weekday() == 0 and cur_t.hour <= 1)
    return (fri_ok or sat) and (sun_resume or cur_t.weekday() == 0)


def write_report(instrument: str, metrics: dict, out_path: Optional[Path] = None) -> Path:
    from danalit.config import load_config

    out_path = out_path or (
        load_config().settings.paths.absolute("reports") / f"data_quality_{instrument}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Data quality — {instrument}", ""]
    for key in ("n_bars", "start", "end", "duplicates", "zero_volume_bars",
                "ohlc_violations", "weekend_bars", "gap_count", "spread_median", "spread_outliers"):
        if key in metrics:
            lines.append(f"- **{key}**: {metrics[key]}")
    gaps = metrics.get("gaps", [])
    if gaps:
        lines += ["", f"## Gaps (first {min(len(gaps), 50)} of {len(gaps)})", ""]
        lines += [f"- {g['from']} → {g['to']}  ({g['minutes']:.0f} min)" for g in gaps[:50]]
    else:
        lines += ["", "No unexplained gaps detected."]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
