"""Synthetic M1 bar generator — used by tests and the pipeline demo mode.

Produces a weekday-only geometric random walk with configurable volatility,
constant-ish spread and Poisson tick volume. Deterministic under a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_m1(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    s0: float = 1.10,
    annual_vol: float = 0.08,
    spread: float = 0.00012,
    seed: int = 7,
    trend_per_year: float = 0.0,
) -> pd.DataFrame:
    """Generate M1 bars between start and end (UTC), weekdays only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(_utc(start), _utc(end), freq="1min", tz="UTC", inclusive="left")
    # Keep Mon-Fri plus Sunday 22:00+ (market open), drop Saturday and most of Sunday.
    keep = (idx.weekday < 5) | ((idx.weekday == 6) & (idx.hour >= 22))
    keep &= ~((idx.weekday == 4) & (idx.hour >= 21))  # Friday close 21:00 UTC
    idx = idx[keep]
    n = len(idx)
    if n == 0:
        raise ValueError("no timestamps in range")

    minutes_per_year = 365.25 * 24 * 60
    sigma = annual_vol / np.sqrt(minutes_per_year)
    drift = trend_per_year / minutes_per_year
    rets = rng.normal(drift, sigma, n)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[s0], close[:-1]])
    wiggle = np.abs(rng.normal(0, sigma * s0, n))
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle

    return pd.DataFrame(
        {
            "time_utc": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.poisson(40, n).astype("int64") + 1,
            "spread": spread * (1 + 0.2 * rng.random(n)),
        }
    )


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
